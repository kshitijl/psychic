use super::schema::{Feature, FeatureInputs, FeatureType, Monotonicity};
use anyhow::Result;
use jiff::{Span, Timestamp};
use std::path::Path;

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

    fn compute(&self, inputs: &FeatureInputs) -> Result<f64> {
        let filename = Path::new(inputs.file_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        Ok(
            if !inputs.query.is_empty()
                && filename
                    .to_lowercase()
                    .starts_with(&inputs.query.to_lowercase())
            {
                1.0
            } else {
                0.0
            },
        )
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

    fn compute(&self, inputs: &FeatureInputs) -> Result<f64> {
        // Calculate 30 days ago timestamp
        let now_ts = Timestamp::from_second(inputs.current_timestamp)?;
        let session_tz = if let Some(session) = inputs.session {
            jiff::tz::TimeZone::get(&session.timezone).unwrap_or(jiff::tz::TimeZone::system())
        } else {
            jiff::tz::TimeZone::system()
        };
        let now_zoned = now_ts.to_zoned(session_tz);
        let thirty_days_ago = now_zoned.checked_sub(Span::new().days(30))?.timestamp();

        // Count clicks in the time window
        let full_path_str = inputs.full_path.to_string_lossy().to_string();
        let clicks = inputs
            .clicks_by_file
            .get(&full_path_str)
            .map(|clicks| {
                clicks
                    .iter()
                    .filter(|c| {
                        c.timestamp >= thirty_days_ago.as_second()
                            && c.timestamp <= inputs.current_timestamp
                    })
                    .count()
            })
            .unwrap_or(0);

        Ok(clicks as f64)
    }
}

// ============================================================================
// Feature: modified_today
// ============================================================================

pub struct ModifiedToday;

impl Feature for ModifiedToday {
    fn name(&self) -> &'static str {
        "modified_today"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Binary
    }

    fn compute(&self, inputs: &FeatureInputs) -> Result<f64> {
        if let Some(mtime) = inputs.mtime {
            let seconds_since_mod = inputs.current_timestamp - mtime;
            let hours = seconds_since_mod / 3600;
            Ok(if hours < 24 { 1.0 } else { 0.0 })
        } else {
            Ok(0.0)
        }
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

    fn compute(&self, inputs: &FeatureInputs) -> Result<f64> {
        // Files from walker are guaranteed to be under cwd
        if inputs.is_from_walker {
            return Ok(1.0);
        }

        // Historical files have already been canonicalized at startup
        // so we can do a simple prefix check
        Ok(if inputs.full_path.starts_with(inputs.cwd) {
            1.0
        } else {
            0.0
        })
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

    fn compute(&self, inputs: &FeatureInputs) -> Result<f64> {
        // Check if any component in the path starts with a dot (hidden)
        let has_hidden_component = inputs.full_path.components().any(|component| {
            component
                .as_os_str()
                .to_str()
                .map(|s| s.starts_with('.'))
                .unwrap_or(false)
        });

        Ok(if has_hidden_component { 1.0 } else { 0.0 })
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

    fn compute(&self, inputs: &FeatureInputs) -> Result<f64> {
        // Calculate 7 days ago timestamp
        let now_ts = Timestamp::from_second(inputs.current_timestamp)?;
        let session_tz = if let Some(session) = inputs.session {
            jiff::tz::TimeZone::get(&session.timezone).unwrap_or(jiff::tz::TimeZone::system())
        } else {
            jiff::tz::TimeZone::system()
        };
        let now_zoned = now_ts.to_zoned(session_tz);
        let seven_days_ago = now_zoned.checked_sub(Span::new().days(7))?.timestamp();

        // Get parent directory of the current file
        let parent_dir = inputs.full_path.parent();

        if parent_dir.is_none() {
            return Ok(0.0);
        }
        let parent_dir = parent_dir.unwrap();

        // Look up clicks in parent directory from precomputed index
        let clicks = inputs
            .clicks_by_parent_dir
            .get(parent_dir)
            .map(|clicks| {
                clicks
                    .iter()
                    .filter(|c| {
                        c.timestamp >= seven_days_ago.as_second()
                            && c.timestamp <= inputs.current_timestamp
                    })
                    .count()
            })
            .unwrap_or(0);

        Ok(clicks as f64)
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

    fn compute(&self, inputs: &FeatureInputs) -> Result<f64> {
        let now_ts = Timestamp::from_second(inputs.current_timestamp)?;
        let session_tz = if let Some(session) = inputs.session {
            jiff::tz::TimeZone::get(&session.timezone).unwrap_or(jiff::tz::TimeZone::system())
        } else {
            jiff::tz::TimeZone::system()
        };
        let now_zoned = now_ts.to_zoned(session_tz);
        let one_hour_ago = now_zoned.checked_sub(Span::new().hours(1))?.timestamp();

        let full_path_str = inputs.full_path.to_string_lossy().to_string();
        let clicks = inputs
            .clicks_by_file
            .get(&full_path_str)
            .map(|clicks| {
                clicks
                    .iter()
                    .filter(|c| {
                        c.timestamp >= one_hour_ago.as_second()
                            && c.timestamp <= inputs.current_timestamp
                    })
                    .count()
            })
            .unwrap_or(0);
        Ok(clicks as f64)
    }
}

// ============================================================================
// Feature: clicks_today
// ============================================================================

pub struct ClicksToday;

impl Feature for ClicksToday {
    fn name(&self) -> &'static str {
        "clicks_today"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Increasing)
    }

    fn compute(&self, inputs: &FeatureInputs) -> Result<f64> {
        let now_ts = Timestamp::from_second(inputs.current_timestamp)?;
        let session_tz = if let Some(session) = inputs.session {
            jiff::tz::TimeZone::get(&session.timezone).unwrap_or(jiff::tz::TimeZone::system())
        } else {
            jiff::tz::TimeZone::system()
        };
        let start_of_day = now_ts.to_zoned(session_tz).start_of_day()?.timestamp();

        let full_path_str = inputs.full_path.to_string_lossy().to_string();
        let clicks = inputs
            .clicks_by_file
            .get(&full_path_str)
            .map(|clicks| {
                clicks
                    .iter()
                    .filter(|c| {
                        c.timestamp >= start_of_day.as_second()
                            && c.timestamp <= inputs.current_timestamp
                    })
                    .count()
            })
            .unwrap_or(0);
        Ok(clicks as f64)
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

    fn compute(&self, inputs: &FeatureInputs) -> Result<f64> {
        let now_ts = Timestamp::from_second(inputs.current_timestamp)?;
        let session_tz = if let Some(session) = inputs.session {
            jiff::tz::TimeZone::get(&session.timezone).unwrap_or(jiff::tz::TimeZone::system())
        } else {
            jiff::tz::TimeZone::system()
        };
        let now_zoned = now_ts.to_zoned(session_tz);
        let seven_days_ago = now_zoned.checked_sub(Span::new().days(7))?.timestamp();

        let full_path_str = inputs.full_path.to_string_lossy().to_string();
        let clicks = inputs
            .clicks_by_file
            .get(&full_path_str)
            .map(|clicks| {
                clicks
                    .iter()
                    .filter(|c| {
                        c.timestamp >= seven_days_ago.as_second()
                            && c.timestamp <= inputs.current_timestamp
                    })
                    .count()
            })
            .unwrap_or(0);
        Ok(clicks as f64)
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

    fn compute(&self, inputs: &FeatureInputs) -> Result<f64> {
        if let Some(mtime) = inputs.mtime {
            let seconds_since_mod = inputs.current_timestamp - mtime;
            Ok(seconds_since_mod as f64)
        } else {
            // If mtime is not available, return a large age
            Ok(Span::new().days(365).get_seconds() as f64)
        }
    }
}
