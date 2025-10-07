use super::schema::{Feature, FeatureInputs, FeatureType};
use anyhow::Result;

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
