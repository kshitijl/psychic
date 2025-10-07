use super::schema::{Feature, FeatureType};
use super::clicks_last_30_days::ClicksLast30Days;
use super::filename_starts_with_query::FilenameStartsWithQuery;
use super::is_under_cwd::IsUnderCwd;
use super::modified_today::ModifiedToday;
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
    ]
});

/// Get all feature names in order
pub fn feature_names() -> Vec<&'static str> {
    FEATURE_REGISTRY.iter().map(|f| f.name()).collect()
}

/// Get names of binary features only
pub fn binary_features() -> Vec<&'static str> {
    FEATURE_REGISTRY
        .iter()
        .filter(|f| matches!(f.feature_type(), FeatureType::Binary))
        .map(|f| f.name())
        .collect()
}

/// Get names of numeric features only
pub fn numeric_features() -> Vec<&'static str> {
    FEATURE_REGISTRY
        .iter()
        .filter(|f| matches!(f.feature_type(), FeatureType::Numeric))
        .map(|f| f.name())
        .collect()
}

/// Get CSV column names (metadata + features)
pub fn csv_columns() -> Vec<&'static str> {
    let mut cols = vec![
        "label",
        "group_id",
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
            json!({
                "name": f.name(),
                "type": match f.feature_type() {
                    FeatureType::Binary => "binary",
                    FeatureType::Numeric => "numeric",
                }
            })
        })
        .collect();

    let schema = json!({
        "features": features,
        "feature_names": feature_names(),
        "binary_features": binary_features(),
        "numeric_features": numeric_features(),
    });
    serde_json::to_string_pretty(&schema).unwrap()
}
