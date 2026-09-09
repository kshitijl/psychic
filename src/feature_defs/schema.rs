use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// Re-export from parent features module
use crate::features::ClickEvent;

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

/// All inputs a feature might need to compute its value.
///
/// Everything here is already resolved: features do lookups and arithmetic, never
/// setup. `compute` runs once per file across a parallel loop, so anything that
/// needs initializing belongs on this struct, not inside a feature. See the note
/// above the `par_iter` in `ranker.rs`.
pub struct FeatureInputs<'a> {
    pub query: &'a str,
    pub file_path: &'a str,
    pub full_path: &'a Path,
    pub mtime: Option<i64>,
    pub file_size: Option<i64>,
    pub cwd: &'a Path,
    pub clicks_by_file: &'a FxHashMap<String, Vec<ClickEvent>>,
    pub clicks_by_parent_dir: &'a FxHashMap<PathBuf, Vec<ClickEvent>>,
    /// Clicks on each path *for this query*, resolved once by the caller rather
    /// than by every feature for every file. `None` when the query has never
    /// been engaged with, which is the common case.
    pub clicks_for_query: Option<&'a FxHashMap<String, Vec<ClickEvent>>>,
    /// The same, for engagements anywhere in an episode containing this query.
    pub engagements_for_query: Option<&'a FxHashMap<String, Vec<ClickEvent>>>,
    pub current_timestamp: i64,
    pub is_from_walker: bool,
    pub is_dir: bool,
    /// How well the file matched the query, as the caller already computed it.
    /// Inference gets it from the filter, which has just done this match;
    /// training computes it once per row with a shared matcher.
    pub fuzzy_score: i64,
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
    ///
    /// Infallible on purpose. Every feature is arithmetic over a struct that
    /// already holds everything it needs, so there is nothing here that can
    /// fail - and saying otherwise put a `Result` in a `rayon` loop, where the
    /// only thing to do with it was `.expect()`, which would take the search
    /// worker down without a word.
    fn compute(&self, inputs: &FeatureInputs) -> f64;
}
