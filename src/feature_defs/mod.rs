// Feature schema and trait definition
pub mod schema;

// Individual feature implementations
mod clicks_last_30_days;
mod filename_starts_with_query;
mod is_under_cwd;
mod modified_today;

// Feature registry - single source of truth
pub mod registry;

// Re-export key types for external use
pub use schema::FeatureInputs;
pub use registry::{FEATURE_REGISTRY, feature_names, csv_columns, export_json};

// These types are shared between parent and this module
#[derive(Debug, Clone)]
pub struct ClickEvent {
    pub timestamp: i64,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub session_id: String,
    pub timezone: String,
    pub cwd: String,
}
