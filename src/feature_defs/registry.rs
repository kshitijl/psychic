use super::implementations::{
    ClicksForThisQuery, ClicksLast7Days, ClicksLast30Days, ClicksLastHour, ClicksLastWeekParentDir,
    ClicksToday, FileSizeBytes, FilenameStartsWithQuery, IsDir, IsHidden, IsUnderCwd, ModifiedAge,
    ModifiedToday,
};
use super::schema::{Feature, FeatureType};
use once_cell::sync::Lazy;
use serde_json::json;

/// THE SINGLE SOURCE OF TRUTH FOR ALL FEATURES
/// To add a feature: implement Feature trait, then add to this list
pub static FEATURE_REGISTRY: Lazy<Vec<Box<dyn Feature>>> = Lazy::new(|| {
    vec![
        Box::new(FilenameStartsWithQuery),
        Box::new(ClicksLast30Days),
        Box::new(ModifiedToday),
        Box::new(IsUnderCwd),
        Box::new(IsHidden),
        Box::new(FileSizeBytes),
        Box::new(ClicksLastWeekParentDir),
        Box::new(ClicksLastHour),
        Box::new(ClicksToday),
        Box::new(ClicksLast7Days),
        Box::new(ModifiedAge),
        Box::new(ClicksForThisQuery),
        Box::new(IsDir),
    ]
});

/// Get all feature names in order
pub fn feature_names() -> Vec<&'static str> {
    FEATURE_REGISTRY.iter().map(|f| f.name()).collect()
}

/// Get CSV column names (metadata + features)
pub fn csv_columns() -> Vec<&'static str> {
    let mut cols = vec![
        "label",
        "episode_id",
        "subsession_id",
        "session_id",
        "query",
        "file_path",
    ];
    cols.extend(feature_names());
    cols
}

/// Export feature schema as JSON for Python
pub fn export_json() -> String {
    let features: Vec<_> = FEATURE_REGISTRY
        .iter()
        .map(|f| {
            let mono_val = f.monotonicity().map(|m| match m {
                super::schema::Monotonicity::Increasing => 1,
                super::schema::Monotonicity::Decreasing => -1,
            });

            json!({
                "name": f.name(),
                "type": match f.feature_type() {
                    FeatureType::Binary => "binary",
                    FeatureType::Numeric => "numeric",
                },
                "monotonicity": mono_val,
            })
        })
        .collect();

    let schema = json!({ "features": features });
    serde_json::to_string_pretty(&schema).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn test_export_json_monotonicity() {
        let json_str = export_json();
        let schema: Value = serde_json::from_str(&json_str).unwrap();

        let features = schema["features"].as_array().unwrap();

        let clicks_feature = features
            .iter()
            .find(|f| f["name"] == "clicks_last_hour")
            .unwrap();
        assert_eq!(clicks_feature["monotonicity"], 1);

        let modified_age_feature = features
            .iter()
            .find(|f| f["name"] == "modified_age")
            .unwrap();
        assert_eq!(modified_age_feature["monotonicity"], -1);

        let no_mono_feature = features
            .iter()
            .find(|f| f["name"] == "filename_starts_with_query")
            .unwrap();
        assert!(no_mono_feature["monotonicity"].is_null());
    }
}
