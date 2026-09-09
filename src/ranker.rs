use anyhow::{Context, Result};
use jiff::Timestamp;
use lightgbm3::Booster;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

// Import features module
use crate::feature_defs::{ClickEvent, FEATURE_REGISTRY, FeatureInputs, feature_names};
use crate::{db, features};

#[derive(Debug, Clone)]
/// One file to rank, borrowed from the worker's registry.
///
/// It borrows rather than owns because it is built fresh for every keystroke:
/// owning the two strings meant two allocations per file per query, and a query
/// over this developer's home directory ranks 243 of them. Nothing here outlives
/// the `rank_files` call it was made for.
pub struct FileCandidate<'a> {
    pub file_id: usize, // Index into main.rs file registry
    pub relative_path: &'a str,
    pub full_path: &'a Path,
    pub mtime: Option<i64>,
    pub file_size: Option<i64>,
    pub is_from_walker: bool,
    pub is_dir: bool,
    pub fuzzy_score: i64, // Score from fuzzy matcher (higher = better match)
}

#[derive(Debug, Clone)]
pub struct FileScore {
    pub file_id: usize, // Index into main.rs file registry
    pub score: f64,
    pub features: Vec<f64>,         // Feature vector in registry order
    pub simple_score: Option<f64>,  // For debugging: score from simple model
    pub ml_score: Option<f64>,      // For debugging: score from ML model
    pub simple_weight: Option<f64>, // For debugging: weight assigned to simple model
    pub ml_weight: Option<f64>,     // For debugging: weight assigned to ML model
    pub fuzzy_score: i64,           // For debugging: fuzzy match score from skim matcher
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelStats {
    pub trained_at: String,
    pub training_duration_seconds: f64,
    pub num_features: usize,
    pub num_total_examples: usize,
    pub num_positive_examples: usize,
    pub num_negative_examples: usize,
    pub top_3_features: Vec<FeatureImportance>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct FeatureImportance {
    pub feature: String,
    pub importance: f64,
}

pub struct ClickData {
    pub clicks_by_file: FxHashMap<String, Vec<ClickEvent>>,
    /// Directories the user has `cd`'d into, from the shell hook. Kept apart
    /// from clicks: a visit says "I work here", a click says "I opened this".
    pub visits_by_dir: FxHashMap<String, Vec<ClickEvent>>,
    pub clicks_by_parent_dir: FxHashMap<PathBuf, Vec<ClickEvent>>,
    /// query -> path -> events. Nested rather than keyed by `(query, path)`
    /// because a ranking pass has one query and hundreds of paths: the query is
    /// looked up once in `rank_files`, and each file is then a lookup by path
    /// alone. The flat key had to be built - two `String`s - for every file on
    /// every keystroke.
    pub clicks_by_query_and_file: FxHashMap<String, FxHashMap<String, Vec<ClickEvent>>>,
    pub engagements_by_episode_query_and_file:
        FxHashMap<String, FxHashMap<String, Vec<ClickEvent>>>,
}

pub struct Ranker {
    model: Option<Booster>,
    pub clicks: ClickData,
    pub stats: Option<ModelStats>,
}

// Tunable constants for hybrid ranking
const CLICKS_WEIGHT: f64 = 3.0; // Weight for clicks in simple model
const RECENCY_WEIGHT: f64 = 1.0; // Weight for recency in simple model

// Sigmoid parameters for normalizing simple scores to [0, 1] range
// With k=0.1, x0=10.0:
//   raw_score=0  → sigmoid ≈ 0.27
//   raw_score=10 → sigmoid = 0.50
//   raw_score=20 → sigmoid ≈ 0.73
//   raw_score=50 → sigmoid ≈ 0.98
const SIGMOID_K: f64 = 0.1; // Steepness of sigmoid curve
const SIGMOID_X0: f64 = 10.0; // Midpoint (where sigmoid = 0.5)

const TRAIN_PY_SOURCE: &str = include_str!("../train.py");

/// Convert fuzzy score for ML use
/// Maps i64::MAX (empty query) to 0.0, otherwise returns as f64
pub fn fuzzy_score_for_ml(fuzzy_score: i64) -> f64 {
    if fuzzy_score == i64::MAX {
        0.0 // Empty query - no match signal
    } else {
        fuzzy_score as f64
    }
}

/// Normalize fuzzy score for simple model scoring
/// Applies basic normalization + sigmoid to map to [0, 1] range
fn fuzzy_score_for_simple_model(fuzzy_score: i64) -> f64 {
    let basic = fuzzy_score_for_ml(fuzzy_score); // Reuses i64::MAX → 0 logic

    if basic == 0.0 {
        0.0 // Don't sigmoid-normalize zero (empty query or no match)
    } else {
        // Sigmoid normalization: x0=100 (midpoint), k=0.02 (steepness)
        // Maps: score 0 → ~0, score 100 → 0.5, score 200+ → ~1.0
        1.0 / (1.0 + (-0.02 * (basic - 100.0)).exp())
    }
}

/// Does a model trained with `model_features` inputs fit today's registry?
///
/// Adding a feature changes the length of the vector `rank_files` builds, and a
/// booster asked to predict from the wrong number of columns fails on every
/// call. The first launch after an upgrade always hits this: `model.txt` was
/// written by the previous version, and the retrain that replaces it has not
/// finished yet.
pub fn model_fits_registry(model_features: usize) -> bool {
    model_features == FEATURE_REGISTRY.len()
}

impl Ranker {
    pub fn new(model_path: &Path, db: &db::Database) -> Result<Self> {
        let model_load_start = std::time::Instant::now();
        let model = Booster::from_file(model_path.to_str().unwrap())
            .context("Failed to load LightGBM model")?;

        let model_features = model.num_features() as usize;
        if !model_fits_registry(model_features) {
            anyhow::bail!(
                "model expects {} features but this build computes {}",
                model_features,
                FEATURE_REGISTRY.len()
            );
        }
        log::info!(
            "TIMING {{\"op\":\"booster_from_file\",\"ms\":{}}}",
            model_load_start.elapsed().as_secs_f64() * 1000.0
        );

        let clicks_load_start = std::time::Instant::now();
        let clicks = Self::load_clicks(db)?;
        log::info!(
            "TIMING {{\"op\":\"load_clicks\",\"ms\":{}}}",
            clicks_load_start.elapsed().as_secs_f64() * 1000.0
        );

        // Load model stats from same directory as model
        let stats_path = model_path
            .parent()
            .map(|p| p.join("model_stats.json"))
            .unwrap_or_else(|| PathBuf::from("model_stats.json"));
        let stats = Self::load_stats(&stats_path);

        Ok(Ranker {
            model: Some(model),
            clicks,
            stats,
        })
    }

    /// Is an ML model loaded, or is ranking running on the simple model alone?
    ///
    /// Only the fallback tests in `search_worker.rs` need to ask; production code
    /// does not branch on it, since a missing model already means weight 0.
    #[cfg(test)]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    pub fn new_empty(db: &db::Database) -> Result<Self> {
        // Load clicks even when there's no model (needed for simple scoring)
        let clicks_load_start = std::time::Instant::now();
        let clicks = Self::load_clicks(db)?;
        log::info!(
            "TIMING {{\"op\":\"load_clicks\",\"ms\":{}}}",
            clicks_load_start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(Ranker {
            model: None,
            clicks,
            stats: None,
        })
    }

    fn load_stats(stats_path: &PathBuf) -> Option<ModelStats> {
        if stats_path.exists() {
            match std::fs::read_to_string(stats_path) {
                Ok(contents) => match serde_json::from_str::<ModelStats>(&contents) {
                    Ok(stats) => Some(stats),
                    Err(e) => {
                        log::warn!("Failed to parse model stats: {}", e);
                        None
                    }
                },
                Err(e) => {
                    log::warn!("Failed to read model stats: {}", e);
                    None
                }
            }
        } else {
            // Only reached with a model already loaded, and train.py writes both
            // files in the same run. Without stats the blend has no idea how
            // much the model was trained on and falls back to ~2% weight, so a
            // model that ranks fine is nearly ignored - say so rather than
            // silently ranking worse.
            log::warn!(
                "Model loaded but no stats at {}; ranking will lean on the \
                 simple model until the next retrain writes them",
                stats_path.display()
            );
            None
        }
    }

    /// Load click events from last 30 days from database
    pub fn load_clicks(db: &db::Database) -> Result<ClickData> {
        let total_start = std::time::Instant::now();

        let now_ts = Timestamp::now().as_second();
        // Simple arithmetic: 30 days = 30 * 24 * 60 * 60 seconds
        let thirty_days_ago_ts = now_ts - (30 * 24 * 60 * 60);

        // Pre-allocate with reasonable capacity to avoid rehashing
        let mut clicks_by_file: FxHashMap<String, Vec<ClickEvent>> =
            FxHashMap::with_capacity_and_hasher(128, Default::default());
        let mut clicks_by_query_and_file: FxHashMap<String, FxHashMap<String, Vec<ClickEvent>>> =
            FxHashMap::with_capacity_and_hasher(256, Default::default());
        let mut engagements_by_episode_query_and_file: FxHashMap<
            String,
            FxHashMap<String, Vec<ClickEvent>>,
        > = FxHashMap::with_capacity_and_hasher(256, Default::default());

        let query_start = std::time::Instant::now();
        let rows = db.engagements_since(thirty_days_ago_ts)?;
        let visits = db.visits_since(thirty_days_ago_ts)?;
        log::info!(
            "TIMING {{\"op\":\"collect_rows\",\"ms\":{},\"count\":{}}}",
            query_start.elapsed().as_secs_f64() * 1000.0,
            rows.len()
        );

        let indexing_start = std::time::Instant::now();
        let row_count = rows.len();
        for db::Engagement {
            full_path: path,
            timestamp,
            query,
            episode_queries: episode_queries_json,
        } in rows
        {
            let click_event = ClickEvent { timestamp };

            // Index by query, then by path
            clicks_by_query_and_file
                .entry(query)
                .or_default()
                .entry(path.clone())
                .or_default()
                .push(click_event);

            // Index by file path only (reuse path without clone)
            clicks_by_file
                .entry(path.clone())
                .or_default()
                .push(click_event);

            // Build episode query index if episode_queries is present
            if let Some(episode_json) = episode_queries_json
                && let Ok(episode_queries) = serde_json::from_str::<Vec<String>>(&episode_json)
            {
                for episode_query in episode_queries {
                    engagements_by_episode_query_and_file
                        .entry(episode_query)
                        .or_default()
                        .entry(path.clone())
                        .or_default()
                        .push(click_event);
                }
            }
        }
        log::info!(
            "TIMING {{\"op\":\"process_click_rows\",\"ms\":{},\"count\":{}}}",
            indexing_start.elapsed().as_secs_f64() * 1000.0,
            row_count
        );

        // Index visits by the directory visited
        let mut visits_by_dir: FxHashMap<String, Vec<ClickEvent>> =
            FxHashMap::with_capacity_and_hasher(128, Default::default());
        for db::Visit {
            full_path,
            timestamp,
        } in visits
        {
            visits_by_dir
                .entry(full_path)
                .or_default()
                .push(ClickEvent { timestamp });
        }

        // Build parent directory index
        let parent_dir_start = std::time::Instant::now();
        let mut clicks_by_parent_dir: FxHashMap<PathBuf, Vec<ClickEvent>> = FxHashMap::default();
        for (path, clicks) in &clicks_by_file {
            if let Some(parent) = Path::new(path).parent() {
                clicks_by_parent_dir
                    .entry(parent.to_path_buf())
                    .or_default()
                    .extend(clicks.iter().copied());
            }
        }
        log::info!(
            "TIMING {{\"op\":\"build_parent_dir_index\",\"ms\":{}}}",
            parent_dir_start.elapsed().as_secs_f64() * 1000.0
        );

        log::info!(
            "TIMING {{\"op\":\"load_clicks_total\",\"ms\":{}}}",
            total_start.elapsed().as_secs_f64() * 1000.0
        );
        log::debug!("Loaded {} total engagements from last 30 days", row_count);
        log::debug!(
            "Loaded {} files with engagement history from last 30 days",
            clicks_by_file.len()
        );
        log::debug!("Indexed {} parent directories", clicks_by_parent_dir.len());
        log::debug!(
            "Indexed {} (query, file) pairs",
            clicks_by_query_and_file.len()
        );
        log::debug!(
            "Indexed {} (episode_query, file) pairs",
            engagements_by_episode_query_and_file.len()
        );

        log::debug!("Loaded visits to {} directories", visits_by_dir.len());

        Ok(ClickData {
            clicks_by_file,
            visits_by_dir,
            clicks_by_parent_dir,
            clicks_by_query_and_file,
            engagements_by_episode_query_and_file,
        })
    }

    /// Apply sigmoid function to normalize raw scores to [0, 1] range
    /// Formula: 1 / (1 + exp(-k * (x - x0)))
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-SIGMOID_K * (x - SIGMOID_X0)).exp())
    }

    /// Compute simple score for cold-start ranking
    /// Raw formula: CLICKS_WEIGHT * clicks_last_7_days + RECENCY_WEIGHT / (1 + modified_age_in_days)
    ///               + 2.0 * fuzzy_score
    /// Then normalized to [0, 1] using sigmoid function
    fn compute_simple_score(&self, file: &FileCandidate<'_>, current_timestamp: i64) -> f64 {
        // Count clicks in last 7 days
        let seven_days_ago = current_timestamp - (7 * 24 * 60 * 60);
        // Borrowed: this runs for every file on every keystroke, and an owned
        // copy of the path bought nothing.
        let full_path_str = file.full_path.to_string_lossy();
        let clicks_last_7_days = self
            .clicks
            .clicks_by_file
            .get(full_path_str.as_ref())
            .map(|clicks| {
                clicks
                    .iter()
                    .filter(|c| c.timestamp >= seven_days_ago && c.timestamp <= current_timestamp)
                    .count()
            })
            .unwrap_or(0) as f64;

        // Compute modified age in days
        let modified_age_in_days = if let Some(mtime) = file.mtime {
            let seconds_since_mod = current_timestamp - mtime;
            (seconds_since_mod as f64) / (24.0 * 60.0 * 60.0)
        } else {
            // Large age for files with no mtime
            365.0
        };

        // Normalize fuzzy score to [0, 1] range using helper function
        let fuzzy_score_normalized = fuzzy_score_for_simple_model(file.fuzzy_score);

        // Combine: clicks weighted 3x, recency 1x, fuzzy match 2x
        // Fuzzy match gets weight of 2.0 to make it significant but not dominating
        let raw_score = CLICKS_WEIGHT * clicks_last_7_days
            + RECENCY_WEIGHT / (1.0 + modified_age_in_days)
            + 2.0 * fuzzy_score_normalized;

        // Normalize to [0, 1] using sigmoid
        Self::sigmoid(raw_score)
    }

    /// Compute blend weights with a tanh ramp over how much the model was
    /// trained on.
    ///
    /// `num_positive_examples` is the count of clicked rows in the training
    /// data, from `model_stats.json`. It answers the only question the gate
    /// needs to ask - has this model seen enough to be trusted - and it does not
    /// move until the next retrain.
    ///
    /// It used to ramp over engagements in the last 30 days, which handed
    /// ranking back to the simple model after a quiet month on an installation
    /// with years of history behind it. Staleness is already handled where it
    /// belongs: the click *data* is a rolling 30-day window, and the model's
    /// features carry no file identity, so the model itself does not go stale.
    ///
    /// Crossover is at `k * l` = 30 positives, saturating around 60.
    ///
    /// Returns: (w_simple, w_lightgbm) where weights sum to 1.0
    fn compute_blend_weights(num_positive_examples: usize) -> (f64, f64) {
        let k = 15f64;
        let l = 2f64;

        let x = num_positive_examples as f64;
        let ml_weight = (1.0 + (x / k - l).tanh()) / 2.0;

        let simple_weight = 1.0 - ml_weight;
        (simple_weight, ml_weight)
    }

    pub fn rank_files(
        &mut self,
        query: &str,
        files: &[FileCandidate<'_>],
        current_timestamp: i64,
        cwd: &Path,
    ) -> Result<Ranking> {
        if files.is_empty() {
            return Ok(Ranking {
                scores: Vec::new(),
                timings: RankTimings::empty(),
            });
        }

        // Compute simple scores for all files (used for cold-start or blending)
        let simple_start = Instant::now();
        let simple_scores: Vec<f64> = files
            .iter()
            .map(|file| self.compute_simple_score(file, current_timestamp))
            .collect();
        let simple_ms = ms_since(simple_start);

        // Always compute features for all files in parallel (for debugging visibility)
        let compute_start = Instant::now();

        let clicks = QueryClicks::resolve(&self.clicks, query);

        // Careful with lazily-initialized globals inside this loop. Feature code
        // runs once per file across every core, so the *first* call after launch
        // has ~170 threads arriving at any cold global at the same moment, all
        // queueing on whatever lock guards its initialization. That is a startup
        // cliff, not a per-call cost, and it does not show up in a warm benchmark:
        // timezone lookups here once cost 4.4ms per file on the first ranking pass
        // and 2us on every one after, which was 75ms of a 97ms time-to-first-result.
        // If a feature needs something expensive to set up, resolve it once outside
        // this loop and pass it in.
        // fold/reduce rather than map/collect: each rayon chunk keeps one timing
        // accumulator and folds its files into it, so the per-feature totals cost
        // one `Vec<Duration>` per chunk instead of a map per file. The chunks are
        // contiguous and are recombined in order, so the features come back
        // aligned with `files`.
        let feature_count = FEATURE_REGISTRY.len();
        let (all_features, per_feature_totals) = files
            .par_iter()
            .fold(
                || (Vec::new(), vec![Duration::ZERO; feature_count]),
                |(mut features, mut per_feature), file| {
                    features.push(compute_features_into(
                        query,
                        file,
                        current_timestamp,
                        cwd,
                        &clicks,
                        &mut per_feature,
                    ));
                    (features, per_feature)
                },
            )
            .reduce(
                || (Vec::new(), vec![Duration::ZERO; feature_count]),
                |(mut features, mut per_feature), (more_features, more_per_feature)| {
                    features.extend(more_features);
                    for (total, add) in per_feature.iter_mut().zip(more_per_feature) {
                        *total += add;
                    }
                    (features, per_feature)
                },
            );
        assert_eq!(
            all_features.len(),
            files.len(),
            "every file must come back with a feature vector, in order"
        );
        let features_ms = ms_since(compute_start);
        let per_feature_ms: Vec<f64> = per_feature_totals
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();

        // If no model, use only simple scores (but keep the computed features for debugging)
        if self.model.is_none() {
            let mut scored_files: Vec<FileScore> = files
                .iter()
                .zip(all_features)
                .enumerate()
                .map(|(idx, (file, features))| FileScore {
                    file_id: file.file_id,
                    score: simple_scores[idx],
                    features,
                    simple_score: Some(simple_scores[idx]),
                    ml_score: None,
                    simple_weight: Some(1.0), // 100% simple model when no ML model
                    ml_weight: Some(0.0),
                    fuzzy_score: file.fuzzy_score,
                })
                .collect();
            scored_files.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            return Ok(Ranking {
                scores: scored_files,
                timings: RankTimings {
                    simple_ms,
                    features_ms,
                    predict_ms: 0.0,
                    blend_ms: 0.0,
                    per_feature_ms,
                },
            });
        }

        // Batch predict all files at once (we have a model)
        // Flatten features into a single vector for batch prediction
        let num_features = if all_features.is_empty() {
            0
        } else {
            all_features[0].len()
        };
        let flat_features: Vec<f64> = all_features.iter().flatten().copied().collect();

        let predict_start = Instant::now();
        let prediction_results = self
            .model
            .as_ref()
            .unwrap()
            .predict_with_params(&flat_features, num_features as i32, true, "num_threads=8")
            .context("Failed to batch predict with model")?;
        let predict_ms = ms_since(predict_start);

        // Blend the two models by how much the model was trained on
        let blend_start = Instant::now();
        let num_positive_examples = self
            .stats
            .as_ref()
            .map(|s| s.num_positive_examples)
            .unwrap_or(0);
        let (w_simple, w_lightgbm) = Self::compute_blend_weights(num_positive_examples);
        log::debug!(
            "Hybrid ranking weights: simple={:.4}, lightgbm={:.4} (num_positive_examples={})",
            w_simple,
            w_lightgbm,
            num_positive_examples
        );

        // Build scored files with hybrid blending
        // Both simple_score and ml_score are now in [0, 1] range
        let mut scored_files = Vec::with_capacity(files.len());
        for (idx, (file, features)) in files.iter().zip(all_features).enumerate() {
            let simple_score = simple_scores[idx]; // Already normalized via sigmoid
            let ml_score = prediction_results[idx]; // Binary classification probability [0, 1]
            let blended_score = w_simple * simple_score + w_lightgbm * ml_score;

            scored_files.push(FileScore {
                file_id: file.file_id,
                score: blended_score,
                features,
                simple_score: Some(simple_score),
                ml_score: Some(ml_score),
                simple_weight: Some(w_simple),
                ml_weight: Some(w_lightgbm),
                fuzzy_score: file.fuzzy_score,
            });
        }
        let blend_ms = ms_since(blend_start);

        // Sort by score descending (higher scores first)
        scored_files.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(Ranking {
            scores: scored_files,
            timings: RankTimings {
                simple_ms,
                features_ms,
                predict_ms,
                blend_ms,
                per_feature_ms,
            },
        })
    }
}

/// Convert feature vector to HashMap (for display purposes)
pub fn features_to_map(features: &[f64]) -> FxHashMap<String, f64> {
    feature_names()
        .iter()
        .zip(features.iter())
        .map(|(name, value)| (name.to_string(), *value))
        .collect()
}

/// The click history with one query's slice already found.
///
/// Resolved once per `rank_files` rather than once per file. The two
/// query-keyed indexes are nested query -> path -> events, so with the query
/// resolved a feature is a lookup by path; before this, each feature built a
/// `(String, String)` key for every file on every keystroke.
struct QueryClicks<'a> {
    all: &'a ClickData,
    clicks_for_query: Option<&'a FxHashMap<String, Vec<ClickEvent>>>,
    engagements_for_query: Option<&'a FxHashMap<String, Vec<ClickEvent>>>,
}

impl<'a> QueryClicks<'a> {
    fn resolve(clicks: &'a ClickData, query: &str) -> Self {
        QueryClicks {
            all: clicks,
            clicks_for_query: clicks.clicks_by_query_and_file.get(query),
            engagements_for_query: clicks.engagements_by_episode_query_and_file.get(query),
        }
    }
}

/// Milliseconds elapsed since `start`, the unit every TIMING field is in.
fn ms_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Where the time went inside one `rank_files` call.
///
/// Timings are returned rather than logged here: the worker logs one line per
/// query, after the results are on their way to the UI. `per_feature_ms` is
/// indexed by position in `FEATURE_REGISTRY` and totals every file.
pub struct RankTimings {
    pub simple_ms: f64,
    pub features_ms: f64,
    pub predict_ms: f64,
    pub blend_ms: f64,
    pub per_feature_ms: Vec<f64>,
}

/// What one ranking pass produced: the scored files, and where the time went.
impl RankTimings {
    /// What a ranking pass that did no work took: nothing, everywhere.
    fn empty() -> Self {
        RankTimings {
            simple_ms: 0.0,
            features_ms: 0.0,
            predict_ms: 0.0,
            blend_ms: 0.0,
            per_feature_ms: vec![0.0; FEATURE_REGISTRY.len()],
        }
    }
}

/// What one ranking pass produced: the scored files, and where the time went.
pub struct Ranking {
    pub scores: Vec<FileScore>,
    pub timings: RankTimings,
}

#[cfg(test)]
/// Compute features for a file (for tests, discarding the timings)
fn compute_features(
    query: &str,
    file: &FileCandidate<'_>,
    current_timestamp: i64,
    cwd: &Path,
    clicks: &ClickData,
) -> Vec<f64> {
    let mut per_feature = vec![Duration::ZERO; FEATURE_REGISTRY.len()];
    compute_features_into(
        query,
        file,
        current_timestamp,
        cwd,
        &QueryClicks::resolve(clicks, query),
        &mut per_feature,
    )
}

/// Compute every feature for one file, adding each one's time into `per_feature`.
///
/// `per_feature` is indexed by position in `FEATURE_REGISTRY` and is a running
/// total across files, so a whole ranking pass costs one accumulator per rayon
/// chunk rather than a map of 15 freshly allocated `String` keys per file.
fn compute_features_into(
    query: &str,
    file: &FileCandidate<'_>,
    current_timestamp: i64,
    cwd: &Path,
    clicks: &QueryClicks<'_>,
    per_feature: &mut [Duration],
) -> Vec<f64> {
    assert_eq!(
        per_feature.len(),
        FEATURE_REGISTRY.len(),
        "the timing accumulator has one slot per registered feature"
    );

    // Create FeatureInputs for inference
    let inputs = FeatureInputs {
        query,
        file_path: file.relative_path,
        full_path: file.full_path,
        mtime: file.mtime,
        file_size: file.file_size,
        cwd,
        clicks_by_file: &clicks.all.clicks_by_file,
        visits_by_dir: &clicks.all.visits_by_dir,
        clicks_by_parent_dir: &clicks.all.clicks_by_parent_dir,
        clicks_for_query: clicks.clicks_for_query,
        engagements_for_query: clicks.engagements_for_query,
        current_timestamp,
        is_from_walker: file.is_from_walker,
        is_dir: file.is_dir,
        fuzzy_score: file.fuzzy_score,
    };

    // Compute all features using the registry, tracking time for each
    let mut features = Vec::with_capacity(FEATURE_REGISTRY.len());

    for (idx, feature) in FEATURE_REGISTRY.iter().enumerate() {
        let start = Instant::now();
        let value = feature.compute(&inputs);
        per_feature[idx] += start.elapsed();

        features.push(value);
    }

    features
}

/// Ensure train.py is materialized in the data directory and return its path.
fn ensure_train_py(data_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(data_dir)
        .with_context(|| format!("Failed to create data directory {}", data_dir.display()))?;

    let train_py_path = data_dir.join("train.py");
    let needs_write = match std::fs::read_to_string(&train_py_path) {
        Ok(existing) => existing != TRAIN_PY_SOURCE,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => true,
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "Failed to read existing train.py at {}",
                    train_py_path.display()
                )
            });
        }
    };

    if needs_write {
        std::fs::write(&train_py_path, TRAIN_PY_SOURCE).with_context(|| {
            format!(
                "Failed to write embedded train.py to {}",
                train_py_path.display()
            )
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&train_py_path)
                .with_context(|| format!("Failed to stat {}", train_py_path.display()))?
                .permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&train_py_path, perms).with_context(|| {
                format!(
                    "Failed to set executable permissions on {}",
                    train_py_path.display()
                )
            })?;
        }

        log::info!("Wrote embedded train.py to {:?}", train_py_path);
    }

    Ok(train_py_path)
}

/// Retrain the model by generating features and running train.py
/// Runs in a spawned thread and blocks until complete
///
/// If `training_log_path` is provided, training output is appended to that file.
/// Otherwise, output is printed to stdout.
pub fn retrain_model(data_dir: &Path, training_log_path: Option<PathBuf>) -> Result<()> {
    let data_dir = data_dir.to_path_buf();

    // Spawn a thread for the training process
    let handle = thread::spawn(move || -> Result<()> {
        let total_start = Instant::now();

        // Step 1: Generate features
        log::info!("Generating features...");
        let feature_start = Instant::now();
        let db_path = db::Database::get_db_path(&data_dir);

        let features_csv = data_dir.join("features.csv");
        let schema_json = data_dir.join("feature_schema.json");

        std::fs::create_dir_all(&data_dir)?;

        let summary = features::generate_features(
            &db_path,
            &features_csv,
            &schema_json,
            features::OutputFormat::Csv,
        )?;
        let feature_duration = feature_start.elapsed();

        // A fresh install has impressions but nothing clicked, so there is
        // nothing for the model to learn from. Training anyway means a Python
        // traceback and an ERROR in the log on the very first launch, for a
        // state that is entirely normal: ranking runs on the simple model
        // until the user has clicked something.
        if summary.positives == 0 {
            log::info!(
                "Nothing to train on yet: {} impressions, none of them clicked. \
                 Ranking stays on the simple model.",
                summary.rows
            );
            return Ok(());
        }
        log::info!(
            "Features generated at {:?} ({:.2}s)",
            features_csv,
            feature_duration.as_secs_f64()
        );

        // Step 2: Ensure train.py is available
        let train_py = ensure_train_py(&data_dir)?;
        log::info!("Using train.py at {:?}", train_py);

        // Step 3: Run training
        log::info!("Training model...");
        let training_start = Instant::now();
        let output_prefix = data_dir.join("model");
        let output_prefix_str = output_prefix
            .to_str()
            .context("Failed to convert output prefix to string")?;

        let data_dir_str = data_dir
            .to_str()
            .context("Failed to convert data_dir to string")?;

        let output = Command::new("uv")
            .arg("run")
            .arg(&train_py)
            .arg(&features_csv)
            .arg(output_prefix_str)
            .arg("--data-dir")
            .arg(data_dir_str)
            .output()
            .context("Failed to run train.py with uv")?;

        // Handle training output - either to file or stdout
        if let Some(log_path) = training_log_path {
            // Append to log file
            use std::fs::OpenOptions;
            use std::io::Write;

            let mut log_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
                .context("Failed to open training log file")?;

            // Write timestamp header
            let now = Timestamp::now();
            let tz = jiff::tz::TimeZone::system();
            let zoned = now.to_zoned(tz);
            let timestamp = zoned.strftime("%Y-%m-%d %H:%M:%S");
            writeln!(log_file, "\n=== Training run at {} ===", timestamp)?;

            if !output.stdout.is_empty() {
                log_file.write_all(&output.stdout)?;
            }
            if !output.stderr.is_empty() {
                log_file.write_all(&output.stderr)?;
            }

            log::info!("Training output appended to {:?}", log_path);
        } else {
            // Print to stdout/stderr
            if !output.stdout.is_empty() {
                let stdout_str = String::from_utf8_lossy(&output.stdout);
                print!("{}", stdout_str);
            }

            if !output.stderr.is_empty() {
                let stderr_str = String::from_utf8_lossy(&output.stderr);
                eprint!("{}", stderr_str);
            }
        }

        if !output.status.success() {
            anyhow::bail!("Training failed with exit code: {:?}", output.status.code());
        }

        let training_duration = training_start.elapsed();
        log::info!(
            "Training complete! ({:.2}s)",
            training_duration.as_secs_f64()
        );
        log::info!("Model saved at {:?}", output_prefix.with_extension("txt"));

        let total_duration = total_start.elapsed();
        log::info!(
            "Total retraining time: {:.2}s",
            total_duration.as_secs_f64()
        );

        Ok(())
    });

    // Wait for the thread to complete and return its result
    handle.join().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;

    #[test]
    fn test_ranker_basic() {
        // Skip if model doesn't exist
        let model_path = PathBuf::from("output.txt");
        if !model_path.exists() {
            eprintln!("Skipping test - output.txt not found");
            return;
        }

        let data_dir = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp"))
            .join(".local")
            .join("share")
            .join("psychic");
        let db_path = Database::get_db_path(&data_dir);

        let db = Database::new(&db_path).expect("open db");
        let ranker = Ranker::new(&model_path, &db);
        match &ranker {
            Ok(_) => println!("✓ Ranker loaded successfully"),
            Err(e) => {
                eprintln!("✗ Failed to load ranker: {}", e);
                panic!("Ranker failed to load: {}", e);
            }
        }

        let mut ranker = ranker.unwrap();

        // Test with simple data
        let test_path = PathBuf::from("/tmp/test.md");
        let test_files = vec![FileCandidate {
            file_id: 0,
            relative_path: "test.md",
            full_path: &test_path,
            mtime: Some(1234567890),
            file_size: Some(2048),
            is_from_walker: true,
            is_dir: false,
            fuzzy_score: 100,
        }];

        let result = ranker.rank_files(
            "test",
            &test_files,
            Timestamp::now().as_second(),
            &PathBuf::from("/tmp"),
        );
        match &result {
            Ok(ranking) => {
                println!("✓ Ranking succeeded");
                for fs in &ranking.scores {
                    println!(
                        "  file_id {} - score: {:.4}, features: {:?}",
                        fs.file_id, fs.score, fs.features
                    );
                }
            }
            Err(e) => {
                eprintln!("✗ Ranking failed: {}", e);
                eprintln!("  Full error chain:");
                let mut source = e.source();
                while let Some(err) = source {
                    eprintln!("    caused by: {}", err);
                    source = err.source();
                }
                panic!("Ranking failed: {}", e);
            }
        }
    }

    #[test]
    fn test_feature_computation() {
        // This test verifies the exact feature vector computed for a known file
        let query = "test";
        let current_timestamp = 1700086400i64; // Nov 15, 2023
        let cwd = PathBuf::from("/tmp");

        // Create file candidate
        let file = FileCandidate {
            file_id: 0,
            relative_path: "foo/bar.txt",
            full_path: &PathBuf::from("/tmp/foo/bar.txt"),
            mtime: Some(1700000000i64), // Nov 14, 2023
            file_size: Some(12_288),
            is_from_walker: true,
            is_dir: false,
            // What the filter would have produced for this pair: "foo/bar.txt"
            // does not match "test", so the matcher returns None and the
            // candidate scores 0. The fixture used to say 100 while the feature
            // recomputed the match and reported 0 - which went unnoticed only
            // because nothing read the candidate's own score.
            fuzzy_score: 0,
        };

        // Create synthetic click data
        let mut clicks_by_file = FxHashMap::default();
        // Add 3 clicks to bar.txt
        clicks_by_file.insert(
            "/tmp/foo/bar.txt".to_string(),
            vec![
                ClickEvent {
                    timestamp: 1700000000,
                },
                ClickEvent {
                    timestamp: 1700010000,
                },
                ClickEvent {
                    timestamp: 1700020000,
                },
            ],
        );
        // Add some clicks to a different file in the same directory (for parent_dir feature)
        clicks_by_file.insert(
            "/tmp/foo/other.txt".to_string(),
            vec![ClickEvent {
                timestamp: 1700000000,
            }],
        );

        // Build parent directory index
        let mut clicks_by_parent_dir = FxHashMap::default();
        for (path, clicks) in &clicks_by_file {
            if let Some(parent) = Path::new(path).parent() {
                clicks_by_parent_dir
                    .entry(parent.to_path_buf())
                    .or_insert_with(Vec::new)
                    .extend(clicks.iter().copied());
            }
        }

        // Build query -> path index - add 2 query-specific clicks
        let mut for_this_query = FxHashMap::default();
        for_this_query.insert(
            "/tmp/foo/bar.txt".to_string(),
            vec![
                ClickEvent {
                    timestamp: 1700000000,
                },
                ClickEvent {
                    timestamp: 1700010000,
                },
            ],
        );
        let mut clicks_by_query_and_file = FxHashMap::default();
        clicks_by_query_and_file.insert(query.to_string(), for_this_query);

        // Build episode engagement index
        let engagements_by_episode_query_and_file = FxHashMap::default();

        // Compute features using the standalone function
        let features = compute_features(
            query,
            &file,
            current_timestamp,
            &cwd,
            &ClickData {
                clicks_by_file,
                visits_by_dir: FxHashMap::default(),
                clicks_by_parent_dir,
                clicks_by_query_and_file,
                engagements_by_episode_query_and_file,
            },
        );

        // Format as string for expect-test style comparison
        let actual = format!("{:?}", features);

        // Expected output: [filename_starts_with_query, clicks_last_30_days, modified_last_24h, is_under_cwd, is_hidden, log_file_size, clicks_last_week_parent_dir, clicks_last_hour, clicks_last_24h, clicks_last_7_days, modified_age, clicks_for_this_query, engagements_in_episode_with_query, is_dir, fuzzy_score]
        // filename_starts_with_query=0 (bar.txt doesn't start with "test")
        // clicks_last_30_days=3 (3 clicks on bar.txt itself)
        // modified_last_24h=0 (mtime is exactly 24h before current_timestamp, so outside)
        // is_under_cwd=1 (is_from_walker=true, so guaranteed to be under cwd)
        // is_hidden=0 (no dot-prefixed components)
        // log_file_size=13.585079902767108 (log2 of 1 + 12288, a 12 KB file)
        // clicks_last_week_parent_dir=4 (3 clicks on bar.txt + 1 click on other.txt in /tmp/foo/)
        // clicks_last_hour=0 (oldest click is 18.4h before current_timestamp)
        // clicks_last_24h=3 (all 3 clicks land in the window; the earliest sits exactly
        //   on the 24h boundary, which counts - see test_window_boundary_is_inclusive)
        // clicks_last_7_days=3 (all 3 clicks are within the last 7 days of the test timestamp)
        // modified_age=86400 (1 day in seconds)
        // clicks_for_this_query=2 (2 query-specific clicks for "test" + bar.txt)
        // engagements_in_episode_with_query=0 (no episode engagement data in test)
        // is_dir=0 (this is a file, not a directory)
        // fuzzy_score=0 (foo/bar.txt doesn't match "test" - fuzzy matcher returns None)
        // visits_last_7_days=0, visits_last_30_days=0 (this is a file, not a
        //   directory, and files are never visited)
        // seconds_since_last_click=11.1035 = ln(1 + 66400), the most recent of
        //   the three clicks being 18.4h old - the same event the "oldest click
        //   is 18.4h before current_timestamp" note above refers to
        // seconds_since_last_click_parent_dir=11.1035 (same event: the most
        //   recent click in /tmp/foo is the one on bar.txt itself)
        //
        // This expectation used to depend on the machine's timezone: when
        // clicks_last_24h was "clicks_today" it counted clicks since local midnight,
        // giving 0 in America/New_York, 2 in UTC and 3 in Asia/Kolkata for exactly
        // this data. Rolling windows are the same number everywhere.
        let expected = "[0.0, 3.0, 0.0, 1.0, 0.0, 13.585079902767108, 4.0, 0.0, 3.0, 3.0, 86400.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.103467395592086, 11.103467395592086]";

        assert_eq!(actual, expected, "Feature vector mismatch");
    }

    #[test]
    fn test_features_come_back_aligned_with_their_files() {
        // The per-file feature work runs as a rayon fold/reduce so the timing
        // accumulator can be per chunk rather than per file. That only holds
        // together if the chunks are recombined in order, so pin it: give each
        // file a size no other file has, and check every scored file carries its
        // own size back.
        let ranker = Ranker {
            model: None,
            clicks: ClickData {
                clicks_by_file: FxHashMap::default(),
                visits_by_dir: FxHashMap::default(),
                clicks_by_parent_dir: FxHashMap::default(),
                clicks_by_query_and_file: FxHashMap::default(),
                engagements_by_episode_query_and_file: FxHashMap::default(),
            },
            stats: None,
        };

        // Enough files that rayon splits them across more than one chunk.
        let sizes: Vec<i64> = (0..512).map(|i| 1024 + i * 7).collect();
        // Candidates borrow, so the names have to outlive them.
        let names: Vec<(String, PathBuf)> = (0..sizes.len())
            .map(|i| {
                (
                    format!("file{}.txt", i),
                    PathBuf::from(format!("/tmp/file{}.txt", i)),
                )
            })
            .collect();
        let files: Vec<FileCandidate> = names
            .iter()
            .zip(&sizes)
            .enumerate()
            .map(|(i, ((relative_path, full_path), &size))| FileCandidate {
                file_id: i,
                relative_path,
                full_path,
                mtime: Some(1_700_500_000),
                file_size: Some(size),
                is_from_walker: true,
                is_dir: false,
                fuzzy_score: 50,
            })
            .collect();

        let mut ranker = ranker;
        let ranking = ranker
            .rank_files("file", &files, 1_700_604_800, &PathBuf::from("/tmp"))
            .expect("ranking without a model");

        assert_eq!(ranking.scores.len(), files.len());

        let size_idx = feature_names()
            .iter()
            .position(|name| *name == "log_file_size")
            .expect("log_file_size is a registered feature");

        for score in &ranking.scores {
            let expected = ((1 + sizes[score.file_id]) as f64).log2();
            assert_eq!(
                score.features[size_idx], expected,
                "file_id {} carried the wrong file's features",
                score.file_id
            );
        }

        assert_eq!(
            ranking.timings.per_feature_ms.len(),
            FEATURE_REGISTRY.len(),
            "one timing slot per registered feature"
        );
    }

    #[test]
    fn test_a_model_from_a_different_feature_set_is_rejected() {
        // The first launch after a feature is added loads a model.txt written
        // by the previous build. A booster asked for the wrong number of
        // columns fails on every predict, so this has to be caught at load
        // time, where the fallback to the simple model lives.
        assert!(model_fits_registry(FEATURE_REGISTRY.len()));
        assert!(!model_fits_registry(FEATURE_REGISTRY.len() - 1));
        assert!(!model_fits_registry(FEATURE_REGISTRY.len() + 1));
        assert!(!model_fits_registry(0));
    }

    #[test]
    fn test_compute_blend_weights() {
        // The ramp runs over training positives - clicked rows the model was
        // trained on, from model_stats.json - not over recent activity.

        // A model trained on nothing, or one whose stats would not load: the
        // simple model should dominate (ML ≈ 2%)
        let (w_simple, w_lightgbm) = Ranker::compute_blend_weights(0);
        assert!(
            w_simple > 0.97 && w_simple < 0.99,
            "With no training positives, w_simple should be ~0.98, got {}",
            w_simple
        );
        assert!(
            w_lightgbm > 0.01 && w_lightgbm < 0.03,
            "With no training positives, w_lightgbm should be ~0.02, got {}",
            w_lightgbm
        );
        assert!(
            (w_simple + w_lightgbm - 1.0).abs() < 0.0001,
            "Weights should sum to 1.0, got {}",
            w_simple + w_lightgbm
        );

        // 100 positives: ML model should dominate (simple nearly zero)
        let (w_simple, w_lightgbm) = Ranker::compute_blend_weights(100);
        assert!(
            w_simple < 0.01,
            "At 100 positives, w_simple should be near 0.0, got {}",
            w_simple
        );
        assert!(
            w_lightgbm > 0.99,
            "At 100 positives, w_lightgbm should be near 1.0, got {}",
            w_lightgbm
        );

        // 1000 positives: ML model should dominate completely
        let (w_simple, w_lightgbm) = Ranker::compute_blend_weights(1000);
        assert!(
            w_simple < 0.001,
            "At 1000 positives, w_simple should be ~0.0, got {}",
            w_simple
        );
        assert!(
            w_lightgbm > 0.999,
            "At 1000 positives, w_lightgbm should be ~1.0, got {}",
            w_lightgbm
        );

        // 30 positives is the documented crossover, k * l, where the two models
        // carry equal weight.
        let (w_simple, w_lightgbm) = Ranker::compute_blend_weights(30);
        assert!(
            (w_simple - 0.5).abs() < 0.0001 && (w_lightgbm - 0.5).abs() < 0.0001,
            "At the crossover the weights should both be 0.5, got {} and {}",
            w_simple,
            w_lightgbm
        );
    }

    #[test]
    fn test_sigmoid_function() {
        // Test that sigmoid function produces expected values in [0, 1] range
        // With k=0.1, x0=10.0

        // Raw score = 0 (no clicks, old file)
        let score_0 = Ranker::sigmoid(0.0);
        assert!(
            score_0 > 0.26 && score_0 < 0.28,
            "sigmoid(0) should be ~0.27, got {}",
            score_0
        );

        // Raw score = 10 (midpoint, should be exactly 0.5)
        let score_10 = Ranker::sigmoid(10.0);
        assert!(
            (score_10 - 0.5).abs() < 0.0001,
            "sigmoid(10) should be 0.5, got {}",
            score_10
        );

        // Raw score = 20 (moderately popular file)
        let score_20 = Ranker::sigmoid(20.0);
        assert!(
            score_20 > 0.73 && score_20 < 0.74,
            "sigmoid(20) should be ~0.73, got {}",
            score_20
        );

        // Raw score = 50 (very popular file)
        let score_50 = Ranker::sigmoid(50.0);
        assert!(
            score_50 > 0.98 && score_50 < 0.99,
            "sigmoid(50) should be ~0.98, got {}",
            score_50
        );

        // Raw score = 100 (extremely popular file)
        let score_100 = Ranker::sigmoid(100.0);
        assert!(
            score_100 > 0.9998 && score_100 < 1.0,
            "sigmoid(100) should be ~0.9999, got {}",
            score_100
        );

        // Negative raw score (should still be > 0, approaches lower asymptote)
        let score_neg = Ranker::sigmoid(-5.0);
        assert!(
            score_neg > 0.18 && score_neg < 0.19,
            "sigmoid(-5) should be ~0.182, got {}",
            score_neg
        );
    }

    #[test]
    fn test_simple_score_with_sigmoid_normalization() {
        // Test that compute_simple_score returns values in [0, 1] range
        // and show actual numeric values for different scenarios

        let mut clicks_by_file = FxHashMap::default();

        // Scenario 1: Popular file with 10 clicks in last week
        clicks_by_file.insert(
            "/tmp/popular.txt".to_string(),
            vec![
                ClickEvent {
                    timestamp: 1700500000,
                },
                ClickEvent {
                    timestamp: 1700510000,
                },
                ClickEvent {
                    timestamp: 1700520000,
                },
                ClickEvent {
                    timestamp: 1700530000,
                },
                ClickEvent {
                    timestamp: 1700540000,
                },
                ClickEvent {
                    timestamp: 1700550000,
                },
                ClickEvent {
                    timestamp: 1700560000,
                },
                ClickEvent {
                    timestamp: 1700570000,
                },
                ClickEvent {
                    timestamp: 1700580000,
                },
                ClickEvent {
                    timestamp: 1700590000,
                },
            ],
        );

        let ranker = Ranker {
            model: None,
            clicks: ClickData {
                clicks_by_file,
                visits_by_dir: FxHashMap::default(),
                clicks_by_parent_dir: FxHashMap::default(),
                clicks_by_query_and_file: FxHashMap::default(),
                engagements_by_episode_query_and_file: FxHashMap::default(),
            },
            stats: None,
        };

        let current_timestamp = 1700604800; // Nov 22, 2023

        // Popular file: 10 clicks + recent mtime + fuzzy score
        // Raw score = 3.0 * 10 + 1.0 / (1 + 1.2) + 2.0 * 0.5 ≈ 31.45
        // sigmoid(31.45) with k=0.1, x0=10 ≈ 0.895
        let popular_file = FileCandidate {
            file_id: 0,
            relative_path: "popular.txt",
            full_path: &PathBuf::from("/tmp/popular.txt"),
            mtime: Some(1700500000), // Recent (1.2 days ago)
            file_size: Some(1024),
            is_from_walker: true,
            is_dir: false,
            fuzzy_score: 100,
        };
        let popular_score = ranker.compute_simple_score(&popular_file, current_timestamp);
        assert!(
            popular_score > 0.0 && popular_score < 1.0,
            "Popular file score should be in [0, 1], got {}",
            popular_score
        );
        assert!(
            popular_score > 0.89 && popular_score < 0.90,
            "Popular file (raw≈31.45) should have score ~0.895, got {}",
            popular_score
        );

        // Recent file with no clicks + fuzzy score
        // Raw score = 3.0 * 0 + 1.0 / (1 + 1.2) + 2.0 * 0.5 ≈ 1.45
        // sigmoid(1.45) ≈ 0.304
        let recent_file = FileCandidate {
            file_id: 1,
            relative_path: "recent.txt",
            full_path: &PathBuf::from("/tmp/recent.txt"),
            mtime: Some(1700500000), // Recent
            file_size: Some(2048),
            is_from_walker: true,
            is_dir: false,
            fuzzy_score: 100,
        };
        let recent_score = ranker.compute_simple_score(&recent_file, current_timestamp);
        assert!(
            recent_score > 0.0 && recent_score < 1.0,
            "Recent file score should be in [0, 1], got {}",
            recent_score
        );
        assert!(
            recent_score > 0.298 && recent_score < 0.305,
            "Recent file with no clicks (raw≈1.45) should have score ~0.304, got {}",
            recent_score
        );

        // Old file with no clicks + fuzzy score
        // Raw score = 3.0 * 0 + 1.0 / (1 + 365) + 2.0 * 0.5 ≈ 1.003
        // sigmoid(1.003) ≈ 0.295
        let old_file = FileCandidate {
            file_id: 2,
            relative_path: "old.txt",
            full_path: &PathBuf::from("/tmp/old.txt"),
            mtime: Some(1600000000), // Very old
            file_size: Some(512),
            is_from_walker: true,
            is_dir: false,
            fuzzy_score: 100,
        };
        let old_score = ranker.compute_simple_score(&old_file, current_timestamp);
        assert!(
            old_score > 0.0 && old_score < 1.0,
            "Old file score should be in [0, 1], got {}",
            old_score
        );
        assert!(
            old_score > 0.288 && old_score < 0.296,
            "Old file with no clicks (raw≈1.003) should have score ~0.295, got {}",
            old_score
        );

        // Verify ordering is preserved: popular > recent > old
        assert!(
            popular_score > recent_score && recent_score > old_score,
            "Score ordering should be preserved: popular({}) > recent({}) > old({})",
            popular_score,
            recent_score,
            old_score
        );
    }

    #[test]
    fn test_compute_simple_score() {
        // Create a ranker with some synthetic click data
        let mut clicks_by_file = FxHashMap::default();
        clicks_by_file.insert(
            "/tmp/foo.txt".to_string(),
            vec![
                ClickEvent {
                    timestamp: 1700000000, // ~7 days ago
                },
                ClickEvent {
                    timestamp: 1700100000, // Recent
                },
            ],
        );

        let ranker = Ranker {
            model: None,
            clicks: ClickData {
                clicks_by_file,
                visits_by_dir: FxHashMap::default(),
                clicks_by_parent_dir: FxHashMap::default(),
                clicks_by_query_and_file: FxHashMap::default(),
                engagements_by_episode_query_and_file: FxHashMap::default(),
            },
            stats: None,
        };

        let current_timestamp = 1700604800; // Nov 22, 2023
        let file = FileCandidate {
            file_id: 0,
            relative_path: "foo.txt",
            full_path: &PathBuf::from("/tmp/foo.txt"),
            mtime: Some(1700500000), // Recent (1 day ago)
            file_size: Some(1024),
            is_from_walker: true,
            is_dir: false,
            fuzzy_score: 100,
        };

        let score = ranker.compute_simple_score(&file, current_timestamp);

        // Raw score: 3.0 * 2 (clicks) + 1.0 / (1 + ~1.2 days) + 2.0 * 0.5 (fuzzy) ≈ 7.45
        // sigmoid(7.45) with k=0.1, x0=10 ≈ 0.437
        assert!(
            score > 0.0 && score < 1.0,
            "Sigmoid-normalized score should be in [0, 1], got {}",
            score
        );
        assert!(
            score > 0.43 && score < 0.44,
            "Sigmoid(7.45) should be ~0.437, got {}",
            score
        );

        // Test file with no clicks and old mtime
        let old_file = FileCandidate {
            file_id: 1,
            relative_path: "bar.txt",
            full_path: &PathBuf::from("/tmp/bar.txt"),
            mtime: Some(1600000000), // Very old
            file_size: Some(2048),
            is_from_walker: true,
            is_dir: false,
            fuzzy_score: 100,
        };

        let score = ranker.compute_simple_score(&old_file, current_timestamp);

        // Raw score: 3.0 * 0 (no clicks) + 1.0 / (1 + ~1160 days) + 2.0 * 0.5 (fuzzy) ≈ 1.001
        // sigmoid(1.001) ≈ 0.289
        assert!(
            score > 0.0 && score < 1.0,
            "Sigmoid-normalized score should be in [0, 1], got {}",
            score
        );
        assert!(
            score > 0.288 && score < 0.291,
            "Sigmoid(~1.001) should be ~0.289, got {}",
            score
        );
    }

    #[test]
    fn test_simple_scoring_without_model() {
        // Test ranking with no model (cold start)
        let mut clicks_by_file = FxHashMap::default();
        clicks_by_file.insert(
            "/tmp/popular.txt".to_string(),
            vec![
                ClickEvent {
                    timestamp: 1700500000,
                },
                ClickEvent {
                    timestamp: 1700510000,
                },
                ClickEvent {
                    timestamp: 1700520000,
                },
            ],
        );

        let mut ranker = Ranker {
            model: None,
            clicks: ClickData {
                clicks_by_file,
                visits_by_dir: FxHashMap::default(),
                clicks_by_parent_dir: FxHashMap::default(),
                clicks_by_query_and_file: FxHashMap::default(),
                engagements_by_episode_query_and_file: FxHashMap::default(),
            },
            stats: None,
        };

        let current_timestamp = 1700604800;
        let (popular, recent) = (
            PathBuf::from("/tmp/popular.txt"),
            PathBuf::from("/tmp/recent.txt"),
        );
        let files = vec![
            FileCandidate {
                file_id: 0,
                relative_path: "popular.txt",
                full_path: &popular,
                mtime: Some(1700000000),
                file_size: Some(1024),
                is_from_walker: true,
                is_dir: false,
                fuzzy_score: 100,
            },
            FileCandidate {
                file_id: 1,
                relative_path: "recent.txt",
                full_path: &recent,
                mtime: Some(1700600000), // Very recent
                file_size: Some(2048),
                is_from_walker: true,
                is_dir: false,
                fuzzy_score: 100,
            },
        ];

        let result = ranker
            .rank_files("", &files, current_timestamp, &PathBuf::from("/tmp"))
            .expect("Ranking should succeed")
            .scores;

        assert_eq!(result.len(), 2, "Should have 2 ranked files");

        // popular.txt should rank first due to clicks
        // Raw: 3 * 3 clicks ≈ 9 → sigmoid(9) ≈ 0.48
        // recent.txt has 0 clicks but very recent mtime
        // Raw: 0 + 1/(1+0.05) ≈ 0.95 → sigmoid(0.95) ≈ 0.29
        // So popular should still rank higher
        assert_eq!(
            result[0].file_id, 0,
            "Popular file should rank first in cold start"
        );

        // Verify debug fields are populated
        assert!(
            result[0].simple_score.is_some(),
            "Simple score should be present"
        );
        assert!(result[0].ml_score.is_none(), "ML score should be None");
    }
}
