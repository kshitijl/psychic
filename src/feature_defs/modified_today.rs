use super::schema::{Feature, FeatureInputs, FeatureType};
use anyhow::Result;

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
