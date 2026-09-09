use super::ClickEvent;
use super::schema::{Feature, FeatureInputs, FeatureType, Monotonicity};
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use jiff::Span;
use std::path::Path;

// ============================================================================
// Time windows
// ============================================================================
//
// Every window here is a plain rolling window counted backwards from the moment
// being scored: "the last 24 hours", never "since midnight". Nothing in this
// file is timezone aware, and that is deliberate - see "Time windows are
// rolling, not calendar days" in how-it-works.md.
//
// The short version: a calendar day needs a timezone, and resolving one cost
// more than every other feature combined (~50ms on the first ranking pass,
// because 170 files hit a cold timezone cache in parallel). A rolling window
// needs one subtraction, means the same thing in every timezone, and cannot
// disagree with itself when a past session ran somewhere else.
//
// Consequence worth knowing: clicks from 11pm still count as "last 24 hours" at
// 10pm the next day, and stop counting at midnight-plus-23-hours rather than at
// midnight. For a relevance signal that is fine, arguably better - it degrades
// smoothly instead of falling off a cliff when the date rolls over.

const SECONDS_PER_HOUR: i64 = 60 * 60;
const SECONDS_PER_DAY: i64 = 24 * SECONDS_PER_HOUR;

/// Count events falling in `(now - window_seconds, now]`.
fn count_in_window(events: Option<&Vec<ClickEvent>>, now: i64, window_seconds: i64) -> f64 {
    assert!(
        window_seconds > 0,
        "A time window must be positive, got {}",
        window_seconds
    );

    let cutoff = now - window_seconds;
    events
        .map(|events| {
            events
                .iter()
                .filter(|event| event.timestamp >= cutoff && event.timestamp <= now)
                .count()
        })
        .unwrap_or(0) as f64
}

/// Engagements with this exact file in the last `window_seconds`.
fn clicks_for_file(inputs: &FeatureInputs, window_seconds: i64) -> f64 {
    let full_path = inputs.full_path.to_string_lossy();
    count_in_window(
        inputs.clicks_by_file.get(full_path.as_ref()),
        inputs.current_timestamp,
        window_seconds,
    )
}

/// Engagements with anything in this file's parent directory, same window.
fn clicks_for_parent_dir(inputs: &FeatureInputs, window_seconds: i64) -> f64 {
    let Some(parent_dir) = inputs.full_path.parent() else {
        return 0.0;
    };

    count_in_window(
        inputs.clicks_by_parent_dir.get(parent_dir),
        inputs.current_timestamp,
        window_seconds,
    )
}

// ============================================================================
// Feature: filename_starts_with_query
// ============================================================================

pub struct FilenameStartsWithQuery;

impl Feature for FilenameStartsWithQuery {
    fn name(&self) -> &'static str {
        "filename_starts_with_query"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Binary
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        let filename = Path::new(inputs.file_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        if !inputs.query.is_empty()
            && filename
                .to_lowercase()
                .starts_with(&inputs.query.to_lowercase())
        {
            1.0
        } else {
            0.0
        }
    }
}

// ============================================================================
// Feature: clicks_last_30_days
// ============================================================================

pub struct ClicksLast30Days;

impl Feature for ClicksLast30Days {
    fn name(&self) -> &'static str {
        "clicks_last_30_days"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        clicks_for_file(inputs, 30 * SECONDS_PER_DAY)
    }
}

// ============================================================================
// Feature: modified_last_24h
// ============================================================================

pub struct ModifiedLast24h;

impl Feature for ModifiedLast24h {
    fn name(&self) -> &'static str {
        "modified_last_24h"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Binary
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        let Some(mtime) = inputs.mtime else {
            return 0.0;
        };

        let modified_within_a_day = inputs.current_timestamp - mtime < SECONDS_PER_DAY;
        if modified_within_a_day { 1.0 } else { 0.0 }
    }
}

// ============================================================================
// Feature: is_under_cwd
// ============================================================================

pub struct IsUnderCwd;

impl Feature for IsUnderCwd {
    fn name(&self) -> &'static str {
        "is_under_cwd"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Binary
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        // Files from walker are guaranteed to be under cwd
        if inputs.is_from_walker {
            return 1.0;
        }

        // Historical files have already been canonicalized at startup
        // so we can do a simple prefix check
        if inputs.full_path.starts_with(inputs.cwd) {
            1.0
        } else {
            0.0
        }
    }
}

// ============================================================================
// Feature: is_hidden
// ============================================================================

pub struct IsHidden;

impl Feature for IsHidden {
    fn name(&self) -> &'static str {
        "is_hidden"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Binary
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        // Check if any component in the path starts with a dot (hidden)
        let has_hidden_component = inputs.full_path.components().any(|component| {
            component
                .as_os_str()
                .to_str()
                .map(|s| s.starts_with('.'))
                .unwrap_or(false)
        });

        if has_hidden_component { 1.0 } else { 0.0 }
    }
}

// ============================================================================
// Feature: log_file_size
// ============================================================================

/// How big the file is, on a log scale.
///
/// Raw byte counts are close to a unique id per file, so a tree can memorise
/// "the 47,312-byte one is the one they click" and score that as skill; under
/// the old random split it did, ranking third by gain. What size legitimately
/// carries is order of magnitude - tiny configs against huge logs and binaries -
/// and the log keeps that while collapsing the lookup table.
pub struct LogFileSize;

impl Feature for LogFileSize {
    fn name(&self) -> &'static str {
        "log_file_size"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        // 1 + size, so an empty file is 0 rather than -inf.
        ((1 + inputs.file_size.unwrap_or(0)) as f64).log2()
    }
}

// ============================================================================
// Feature: clicks_last_week_parent_dir
// ============================================================================

pub struct ClicksLastWeekParentDir;

impl Feature for ClicksLastWeekParentDir {
    fn name(&self) -> &'static str {
        "clicks_last_week_parent_dir"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Increasing)
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        clicks_for_parent_dir(inputs, 7 * SECONDS_PER_DAY)
    }
}

// ============================================================================
// Feature: clicks_last_hour
// ============================================================================

pub struct ClicksLastHour;

impl Feature for ClicksLastHour {
    fn name(&self) -> &'static str {
        "clicks_last_hour"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Increasing)
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        clicks_for_file(inputs, SECONDS_PER_HOUR)
    }
}

// ============================================================================
// Feature: clicks_last_24h
// ============================================================================

pub struct ClicksLast24h;

impl Feature for ClicksLast24h {
    fn name(&self) -> &'static str {
        "clicks_last_24h"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Increasing)
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        clicks_for_file(inputs, SECONDS_PER_DAY)
    }
}

// ============================================================================
// Feature: clicks_last_7_days
// ============================================================================

pub struct ClicksLast7Days;

impl Feature for ClicksLast7Days {
    fn name(&self) -> &'static str {
        "clicks_last_7_days"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Increasing)
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        clicks_for_file(inputs, 7 * SECONDS_PER_DAY)
    }
}

// ============================================================================
// Feature: modified_age
// ============================================================================

pub struct ModifiedAge;

impl Feature for ModifiedAge {
    fn name(&self) -> &'static str {
        "modified_age"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Decreasing)
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        if let Some(mtime) = inputs.mtime {
            let seconds_since_mod = inputs.current_timestamp - mtime;
            seconds_since_mod as f64
        } else {
            // If mtime is not available, return a large age
            Span::new().days(365).get_seconds() as f64
        }
    }
}

// ============================================================================
// Feature: clicks_for_this_query
// ============================================================================

pub struct ClicksForThisQuery;

impl Feature for ClicksForThisQuery {
    fn name(&self) -> &'static str {
        "clicks_for_this_query"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Increasing)
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        let full_path_str = inputs.full_path.to_string_lossy().to_string();
        let key = (inputs.query.to_string(), full_path_str);

        let clicks = inputs
            .clicks_by_query_and_file
            .get(&key)
            .map(|clicks| clicks.len())
            .unwrap_or(0);

        clicks as f64
    }
}

// ============================================================================
// Feature: engagements_in_episode_with_query
// ============================================================================

pub struct EngagementsInEpisodeWithQuery;

impl Feature for EngagementsInEpisodeWithQuery {
    fn name(&self) -> &'static str {
        "engagements_in_episode_with_query"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Increasing)
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        let full_path_str = inputs.full_path.to_string_lossy().to_string();
        let key = (inputs.query.to_string(), full_path_str);

        let engagements = inputs
            .engagements_by_episode_query_and_file
            .get(&key)
            .map(|engagements| engagements.len())
            .unwrap_or(0);

        engagements as f64
    }
}

// ============================================================================
// Feature: is_dir
// ============================================================================

pub struct IsDir;

impl Feature for IsDir {
    fn name(&self) -> &'static str {
        "is_dir"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Binary
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        if inputs.is_dir { 1.0 } else { 0.0 }
    }
}

// ============================================================================
// Feature: fuzzy_score
// ============================================================================

pub struct FuzzyScore;

impl Feature for FuzzyScore {
    fn name(&self) -> &'static str {
        "fuzzy_score"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Increasing) // Higher fuzzy score = better match
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        // Return 0.0 for empty queries (no fuzzy match signal)
        if inputs.query.is_empty() {
            return 0.0;
        }

        // Compute fuzzy match score using SkimMatcherV2
        let matcher = SkimMatcherV2::default();
        let score = matcher
            .fuzzy_match(inputs.file_path, inputs.query)
            .unwrap_or(0); // Return 0 if no match

        score as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_700_086_400;

    fn events(offsets_in_seconds: &[i64]) -> Vec<ClickEvent> {
        offsets_in_seconds
            .iter()
            .map(|offset| ClickEvent {
                timestamp: NOW - offset,
            })
            .collect()
    }

    #[test]
    fn test_counts_only_events_inside_the_window() {
        let events = events(&[
            60,          // a minute ago
            2 * 3600,    // 2 hours ago
            36 * 3600,   // a day and a half ago
            40 * 86_400, // well outside every window
        ]);

        assert_eq!(count_in_window(Some(&events), NOW, SECONDS_PER_HOUR), 1.0);
        assert_eq!(count_in_window(Some(&events), NOW, SECONDS_PER_DAY), 2.0);
        assert_eq!(
            count_in_window(Some(&events), NOW, 7 * SECONDS_PER_DAY),
            3.0
        );
        assert_eq!(
            count_in_window(Some(&events), NOW, 30 * SECONDS_PER_DAY),
            3.0,
            "The 40-day-old event is outside even the widest window"
        );
    }

    #[test]
    fn test_window_boundary_is_inclusive() {
        // An event exactly one window old counts; one second older does not.
        let on_the_boundary = events(&[SECONDS_PER_DAY]);
        let just_past_it = events(&[SECONDS_PER_DAY + 1]);

        assert_eq!(
            count_in_window(Some(&on_the_boundary), NOW, SECONDS_PER_DAY),
            1.0
        );
        assert_eq!(
            count_in_window(Some(&just_past_it), NOW, SECONDS_PER_DAY),
            0.0
        );
    }

    #[test]
    fn test_future_events_do_not_count() {
        // Clock skew, or feature generation replaying an impression from before a
        // later click: neither should count toward the window.
        let ahead_of_now = events(&[-60]);
        assert_eq!(
            count_in_window(Some(&ahead_of_now), NOW, SECONDS_PER_DAY),
            0.0
        );
    }

    #[test]
    fn test_no_events_for_this_file() {
        assert_eq!(count_in_window(None, NOW, SECONDS_PER_DAY), 0.0);
        assert_eq!(
            count_in_window(Some(&Vec::new()), NOW, SECONDS_PER_DAY),
            0.0
        );
    }

    #[test]
    fn test_windows_are_the_lengths_they_claim() {
        assert_eq!(SECONDS_PER_HOUR, 3_600);
        assert_eq!(SECONDS_PER_DAY, 86_400);
        assert_eq!(7 * SECONDS_PER_DAY, 604_800);
        assert_eq!(30 * SECONDS_PER_DAY, 2_592_000);
    }
}
