use super::schema::{Feature, FeatureInputs, FeatureType};
use anyhow::Result;
use jiff::{Span, Timestamp};

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
            jiff::tz::TimeZone::get(&session.timezone)
                .unwrap_or(jiff::tz::TimeZone::system())
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
