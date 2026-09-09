use super::implementations::{
    ClicksForThisQuery, ClicksLast7Days, ClicksLast24h, ClicksLast30Days, ClicksLastHour,
    ClicksLastWeekParentDir, EngagementsInEpisodeWithQuery, FilenameStartsWithQuery, FuzzyScore,
    IsDir, IsHidden, IsUnderCwd, LogFileSize, ModifiedAge, ModifiedLast24h, SecondsSinceLastClick,
    SecondsSinceLastClickParentDir, VisitsLast7Days, VisitsLast30Days,
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
        Box::new(ModifiedLast24h),
        Box::new(IsUnderCwd),
        Box::new(IsHidden),
        Box::new(LogFileSize),
        Box::new(ClicksLastWeekParentDir),
        Box::new(ClicksLastHour),
        Box::new(ClicksLast24h),
        Box::new(ClicksLast7Days),
        Box::new(ModifiedAge),
        Box::new(ClicksForThisQuery),
        Box::new(EngagementsInEpisodeWithQuery),
        Box::new(IsDir),
        Box::new(FuzzyScore),
        // New features go on the end: the position in this list is the
        // position in the model's feature vector, so inserting one in the
        // middle silently repoints every feature after it.
        Box::new(VisitsLast7Days),
        Box::new(VisitsLast30Days),
        Box::new(SecondsSinceLastClick),
        Box::new(SecondsSinceLastClickParentDir),
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
        // When the impression happened. Not a feature - training weights rows
        // by age from it, so recent habits count for more than old ones.
        "timestamp",
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
