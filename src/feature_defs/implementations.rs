use super::schema::{Feature, FeatureInputs, FeatureType};
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
        let full_path_normalized = inputs
            .full_path
            .canonicalize()
            .unwrap_or_else(|_| inputs.full_path.to_path_buf());
        let cwd_normalized = inputs
            .cwd
            .canonicalize()
            .unwrap_or_else(|_| inputs.cwd.to_path_buf());

        Ok(if full_path_normalized.starts_with(&cwd_normalized) {
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

        // Count clicks in parent directory within time window
        let clicks = inputs
            .clicks_by_file
            .iter()
            .filter(|(path, _)| {
                // Check if this click is for a file in the same parent directory
                Path::new(path).parent() == Some(parent_dir)
            })
            .flat_map(|(_, clicks)| clicks)
            .filter(|c| {
                c.timestamp >= seven_days_ago.as_second() && c.timestamp <= inputs.current_timestamp
            })
            .count();

        Ok(clicks as f64)
    }
}
