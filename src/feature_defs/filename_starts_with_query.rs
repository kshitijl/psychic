use super::schema::{Feature, FeatureInputs, FeatureType};
use anyhow::Result;
use std::path::Path;

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

        Ok(if !inputs.query.is_empty()
            && filename
                .to_lowercase()
                .starts_with(&inputs.query.to_lowercase())
        {
            1.0
        } else {
            0.0
        })
    }
}
