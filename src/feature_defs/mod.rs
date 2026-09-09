// Feature schema and trait definition
pub mod schema;

// All feature implementations in one file
mod implementations;

// Feature registry - single source of truth
pub mod registry;

// Re-export key types for external use
pub use registry::{FEATURE_REGISTRY, csv_columns, export_json, feature_names};
pub use schema::{FeatureInputs, extension_key};

// These types are shared between parent and this module
#[derive(Debug, Clone, Copy)]
pub struct ClickEvent {
    pub timestamp: i64,
}

/// The parts of a recorded session that feature generation replays.
///
/// The `sessions` table also stores a timezone, but no feature reads it: every
/// time window is a rolling one (see `implementations.rs`), so it is session
/// context for analysis rather than a model input.
#[derive(Debug, Clone)]
pub struct Session {
    pub session_id: String,
    pub cwd: String,
}
