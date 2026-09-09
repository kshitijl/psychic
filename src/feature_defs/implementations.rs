use super::ClickEvent;
use super::schema::{Feature, FeatureInputs, FeatureType, Monotonicity};
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
        let full_path = inputs.full_path.to_string_lossy();
        inputs
            .clicks_for_query
            .and_then(|by_path| by_path.get(full_path.as_ref()))
            .map(|clicks| clicks.len())
            .unwrap_or(0) as f64
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
        let full_path = inputs.full_path.to_string_lossy();
        let engagements = inputs
            .engagements_for_query
            .and_then(|by_path| by_path.get(full_path.as_ref()))
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
        // The caller has already matched this file against the query - the
        // filter did it to decide the file was a candidate at all. Redoing it
        // here meant building a fresh SkimMatcherV2 per file per keystroke and
        // running the match twice.
        if inputs.query.is_empty() {
            return 0.0; // no fuzzy match signal
        }
        inputs.fuzzy_score as f64
    }
}

// ============================================================================
// Feature: seconds_since_last_click, seconds_since_last_click_parent_dir
// ============================================================================

/// What a file with no clicks at all reports, in log-seconds.
///
/// `ln(1 + ten years)`, to 2dp. The click index only ever holds 30 days, whose
/// log is 14.77, so every real gap sits well below this: "never" stays clearly
/// separated from "a long time ago" while living on the same axis, which is
/// what lets one monotone split tell the two apart.
const NEVER_CLICKED: f64 = 19.57;

/// How long since the most recent event in `events`, as `ln(1 + seconds)`.
///
/// Log-scaled because what matters is the order of magnitude: a minute against
/// an hour is a real difference, an hour against an hour and a minute is not.
/// The click-count windows can say a file was clicked today; only this can say
/// it was clicked a moment ago.
fn log_seconds_since_last(events: Option<&Vec<ClickEvent>>, now: i64) -> f64 {
    let Some(events) = events else {
        return NEVER_CLICKED;
    };

    // Events after `now` come from a clock that moved backwards; treat them as
    // just-happened rather than letting a negative age take the log.
    let most_recent = events.iter().map(|event| event.timestamp).max();
    match most_recent {
        Some(timestamp) => ((1 + (now - timestamp).max(0)) as f64).ln(),
        None => NEVER_CLICKED,
    }
}

pub struct SecondsSinceLastClick;

impl Feature for SecondsSinceLastClick {
    fn name(&self) -> &'static str {
        "seconds_since_last_click"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Decreasing) // longer ago -> less relevant
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        let full_path = inputs.full_path.to_string_lossy();
        log_seconds_since_last(
            inputs.clicks_by_file.get(full_path.as_ref()),
            inputs.current_timestamp,
        )
    }
}

pub struct SecondsSinceLastClickParentDir;

impl Feature for SecondsSinceLastClickParentDir {
    fn name(&self) -> &'static str {
        "seconds_since_last_click_parent_dir"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Decreasing)
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        let Some(parent_dir) = inputs.full_path.parent() else {
            return NEVER_CLICKED;
        };
        log_seconds_since_last(
            inputs.clicks_by_parent_dir.get(parent_dir),
            inputs.current_timestamp,
        )
    }
}

// ============================================================================
// Feature: visits_last_7_days, visits_last_30_days
// ============================================================================

/// How often the user has changed into this directory in `window_seconds`.
///
/// Zero for files: a file is never `cd`'d into, and letting it inherit its
/// directory's count would make every file in a busy directory look visited.
fn visits_for_dir(inputs: &FeatureInputs, window_seconds: i64) -> f64 {
    if !inputs.is_dir {
        return 0.0;
    }

    let full_path = inputs.full_path.to_string_lossy();
    count_in_window(
        inputs.visits_by_dir.get(full_path.as_ref()),
        inputs.current_timestamp,
        window_seconds,
    )
}

pub struct VisitsLast7Days;

impl Feature for VisitsLast7Days {
    fn name(&self) -> &'static str {
        "visits_last_7_days"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Increasing)
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        visits_for_dir(inputs, 7 * SECONDS_PER_DAY)
    }
}

pub struct VisitsLast30Days;

impl Feature for VisitsLast30Days {
    fn name(&self) -> &'static str {
        "visits_last_30_days"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Increasing)
    }

    fn compute(&self, inputs: &FeatureInputs) -> f64 {
        visits_for_dir(inputs, 30 * SECONDS_PER_DAY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;
    use std::path::PathBuf;

    const NOW: i64 = 1_700_086_400;

    fn events(offsets_in_seconds: &[i64]) -> Vec<ClickEvent> {
        offsets_in_seconds
            .iter()
            .map(|offset| ClickEvent {
                timestamp: NOW - offset,
            })
            .collect()
    }

    /// FeatureInputs for one row, with everything empty but the visit index.
    fn inputs_for<'a>(
        full_path: &'a Path,
        is_dir: bool,
        visits: &'a FxHashMap<String, Vec<ClickEvent>>,
        empty_clicks: &'a FxHashMap<String, Vec<ClickEvent>>,
        empty_dirs: &'a FxHashMap<PathBuf, Vec<ClickEvent>>,
    ) -> FeatureInputs<'a> {
        FeatureInputs {
            query: "",
            file_path: "row",
            full_path,
            mtime: None,
            file_size: None,
            cwd: Path::new("/tmp"),
            clicks_by_file: empty_clicks,
            visits_by_dir: visits,
            clicks_by_parent_dir: empty_dirs,
            clicks_for_query: None,
            engagements_for_query: None,
            current_timestamp: NOW,
            is_from_walker: true,
            is_dir,
            fuzzy_score: 0,
        }
    }

    #[test]
    fn test_recency_reports_the_most_recent_click_not_the_first() {
        // Three clicks, one of them a minute ago: what matters is the newest.
        let events = events(&[60, 3600, 40 * 86_400]);
        let seconds = log_seconds_since_last(Some(&events), NOW);

        assert!(
            (seconds - (61.0f64).ln()).abs() < 1e-12,
            "expected ln(1 + 60), got {}",
            seconds
        );
    }

    #[test]
    fn test_never_clicked_sits_above_anything_the_index_can_hold() {
        // The click index holds 30 days. "Never" has to be clear of that, or a
        // split cannot separate "no history" from "old history".
        let oldest_possible = events(&[30 * 86_400]);
        let thirty_days = log_seconds_since_last(Some(&oldest_possible), NOW);

        assert_eq!(log_seconds_since_last(None, NOW), NEVER_CLICKED);
        assert!(
            NEVER_CLICKED > thirty_days + 4.0,
            "never ({}) should be well clear of a 30-day-old click ({})",
            NEVER_CLICKED,
            thirty_days
        );
    }

    #[test]
    fn test_a_minute_and_an_hour_are_further_apart_than_two_similar_hours() {
        // The point of the log: the windows already say "clicked today", so
        // this has to be the feature that tells a minute from an hour.
        let minute = log_seconds_since_last(Some(&events(&[60])), NOW);
        let hour = log_seconds_since_last(Some(&events(&[3600])), NOW);
        let hour_and_a_bit = log_seconds_since_last(Some(&events(&[3660])), NOW);

        assert!(hour - minute > 4.0, "a minute and an hour are far apart");
        assert!(
            hour_and_a_bit - hour < 0.02,
            "an hour and an hour and a minute are nearly the same"
        );
    }

    #[test]
    fn test_a_clock_that_went_backwards_reads_as_just_now() {
        // Timestamps come from wall clocks on machines that adjust them. A
        // future click must not produce ln of a negative number.
        let future = vec![ClickEvent {
            timestamp: NOW + 3600,
        }];
        assert_eq!(log_seconds_since_last(Some(&future), NOW), (1.0f64).ln());
    }

    #[test]
    fn test_visits_count_only_this_directory_and_only_in_window() {
        let mut visits = FxHashMap::default();
        visits.insert(
            "/tmp/project".to_string(),
            events(&[
                60,          // a minute ago
                3 * 86_400,  // three days ago
                20 * 86_400, // twenty days ago
                40 * 86_400, // outside every window
            ]),
        );
        let (clicks, dirs) = (FxHashMap::default(), FxHashMap::default());
        let path = PathBuf::from("/tmp/project");
        let inputs = inputs_for(&path, true, &visits, &clicks, &dirs);

        assert_eq!(VisitsLast7Days.compute(&inputs), 2.0);
        assert_eq!(VisitsLast30Days.compute(&inputs), 3.0);
    }

    #[test]
    fn test_a_directory_nobody_visited_scores_zero() {
        let mut visits = FxHashMap::default();
        visits.insert("/tmp/project".to_string(), events(&[60]));
        let (clicks, dirs) = (FxHashMap::default(), FxHashMap::default());
        let path = PathBuf::from("/tmp/elsewhere");
        let inputs = inputs_for(&path, true, &visits, &clicks, &dirs);

        assert_eq!(VisitsLast30Days.compute(&inputs), 0.0);
    }

    #[test]
    fn test_a_file_never_counts_as_visited() {
        // The visit index is keyed by directory, and a file has none - but a
        // file could share a path with a directory that was later replaced, and
        // more to the point, letting a file inherit its directory's count would
        // make every file in a busy directory look visited.
        let mut visits = FxHashMap::default();
        visits.insert("/tmp/project".to_string(), events(&[60, 120]));
        let (clicks, dirs) = (FxHashMap::default(), FxHashMap::default());
        let path = PathBuf::from("/tmp/project");
        let inputs = inputs_for(&path, false, &visits, &clicks, &dirs);

        assert_eq!(VisitsLast7Days.compute(&inputs), 0.0);
        assert_eq!(VisitsLast30Days.compute(&inputs), 0.0);
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
