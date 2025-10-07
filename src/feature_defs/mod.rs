// Feature schema and trait definition
pub mod schema;

// All feature implementations in one file
mod implementations;

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
