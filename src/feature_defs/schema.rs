use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

// Re-export from parent features module
use crate::features::{ClickEvent, Session};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureType {
    Binary,
    Numeric,
}

/// All inputs a feature might need to compute its value
pub struct FeatureInputs<'a> {
    pub query: &'a str,
    pub file_path: &'a str,
    pub full_path: &'a Path,
    pub mtime: Option<i64>,
    pub cwd: &'a Path,
    pub clicks_by_file: &'a HashMap<String, Vec<ClickEvent>>,
    pub current_timestamp: i64,
    pub session: Option<&'a Session>,
    pub is_from_walker: bool,
}

/// Trait that all features must implement
pub trait Feature: Send + Sync {
    /// Feature name (used in CSV, model, UI)
    fn name(&self) -> &'static str;

    /// Feature type (binary or numeric)
    fn feature_type(&self) -> FeatureType;

    /// Compute feature value from inputs
    /// Used for both training and inference
    fn compute(&self, inputs: &FeatureInputs) -> Result<f64>;
}
