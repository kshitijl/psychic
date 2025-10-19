use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// Re-export from parent features module
use crate::features::{ClickEvent, Session};

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum FeatureType {
    Binary,
    Numeric,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Monotonicity {
    Increasing = 1,
    Decreasing = -1,
}

/// All inputs a feature might need to compute its value
pub struct FeatureInputs<'a> {
    pub query: &'a str,
    pub file_path: &'a str,
    pub full_path: &'a Path,
    pub mtime: Option<i64>,
    pub cwd: &'a Path,
    pub clicks_by_file: &'a HashMap<String, Vec<ClickEvent>>,
    pub clicks_by_parent_dir: &'a HashMap<PathBuf, Vec<ClickEvent>>,
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

    /// Monotonicity constraint for the model
    fn monotonicity(&self) -> Option<Monotonicity> {
        None // Default to no constraint
    }

    /// Compute feature value from inputs
    /// Used for both training and inference
    fn compute(&self, inputs: &FeatureInputs) -> Result<f64>;
}
