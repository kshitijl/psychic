use anyhow::{Context, Result};
use jiff::Timestamp;
use lightgbm3::Booster;
use rayon::prelude::*;
use rusqlite::Connection;
use rustc_hash::FxHashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

// Import features module
use crate::feature_defs::{ClickEvent, FEATURE_REGISTRY, FeatureInputs, feature_names};
use crate::{db, features};

#[derive(Debug, Clone)]
pub struct FileCandidate {
    pub file_id: usize, // Index into main.rs file registry
    pub relative_path: String,
    pub full_path: PathBuf,
    pub mtime: Option<i64>,
    pub file_size: Option<i64>,
    pub is_from_walker: bool,
    pub is_dir: bool,
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
    pub clicks_by_parent_dir: FxHashMap<PathBuf, Vec<ClickEvent>>,
    pub clicks_by_query_and_file: FxHashMap<(String, String), Vec<ClickEvent>>,
    pub engagements_by_episode_query_and_file: FxHashMap<(String, String), Vec<ClickEvent>>,
}

pub struct Ranker {
    model: Option<Booster>,
    pub clicks: ClickData,
    pub stats: Option<ModelStats>,
    pub total_clicks: usize,
}

// Tunable constants for hybrid ranking
const CLICKS_WEIGHT: f64 = 3.0; // Weight for clicks in simple model
const RECENCY_WEIGHT: f64 = 1.0; // Weight for recency in simple model

impl Ranker {
    pub fn new(model_path: &Path, db_path: &Path) -> Result<Self> {
        let model_load_start = std::time::Instant::now();
        let model = Booster::from_file(model_path.to_str().unwrap())
            .context("Failed to load LightGBM model")?;
        log::info!(
            "TIMING {{\"op\":\"booster_from_file\",\"ms\":{}}}",
            model_load_start.elapsed().as_secs_f64() * 1000.0
        );

        let clicks_load_start = std::time::Instant::now();
        let (clicks, total_clicks) = Self::load_clicks(db_path)?;
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
            total_clicks,
        })
    }

    pub fn new_empty(db_path: &Path) -> Result<Self> {
        // Load clicks even when there's no model (needed for simple scoring)
        let clicks_load_start = std::time::Instant::now();
        let (clicks, total_clicks) = Self::load_clicks(db_path)?;
        log::info!(
            "TIMING {{\"op\":\"load_clicks\",\"ms\":{}}}",
            clicks_load_start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(Ranker {
            model: None,
            clicks,
            stats: None,
            total_clicks,
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
            None
        }
    }

    /// Load click events from last 30 days from database
    /// Returns (ClickData, total_clicks)
    pub fn load_clicks(db_path: &Path) -> Result<(ClickData, usize)> {
        let total_start = std::time::Instant::now();

        let timestamp_calc_start = std::time::Instant::now();
        let now = Timestamp::now();
        let now_ts = now.as_second();
        // Simple arithmetic: 30 days = 30 * 24 * 60 * 60 seconds
        let thirty_days_ago_ts = now_ts - (30 * 24 * 60 * 60);
        log::info!(
            "TIMING {{\"op\":\"timestamp_calc\",\"ms\":{}}}",
            timestamp_calc_start.elapsed().as_secs_f64() * 1000.0
        );

        // Pre-allocate with reasonable capacity to avoid rehashing
        let mut clicks_by_file: FxHashMap<String, Vec<ClickEvent>> =
            FxHashMap::with_capacity_and_hasher(128, Default::default());
        let mut clicks_by_query_and_file: FxHashMap<(String, String), Vec<ClickEvent>> =
            FxHashMap::with_capacity_and_hasher(256, Default::default());
        let mut engagements_by_episode_query_and_file: FxHashMap<
            (String, String),
            Vec<ClickEvent>,
        > = FxHashMap::with_capacity_and_hasher(256, Default::default());

        let db_open_start = std::time::Instant::now();
        let conn =
            Connection::open(db_path).context("Failed to open database for preloading clicks")?;

        // Configure SQLite for better concurrency
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;",
        )?;

        log::info!(
            "TIMING {{\"op\":\"db_open\",\"ms\":{}}}",
            db_open_start.elapsed().as_secs_f64() * 1000.0
        );

        let query_start = std::time::Instant::now();
        let mut stmt = conn.prepare(
            "SELECT full_path, timestamp, query, episode_queries
             FROM events
             WHERE action IN ('click', 'scroll')
             AND timestamp >= ?1",
        )?;

        let rows = stmt.query_map([thirty_days_ago_ts], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?;
        log::info!(
            "TIMING {{\"op\":\"db_query_prepare\",\"ms\":{}}}",
            query_start.elapsed().as_secs_f64() * 1000.0
        );

        let collect_start = std::time::Instant::now();
        let rows: Vec<(String, i64, String, Option<String>)> =
            rows.collect::<Result<Vec<_>, _>>()?;
        log::info!(
            "TIMING {{\"op\":\"collect_rows\",\"ms\":{},\"count\":{}}}",
            collect_start.elapsed().as_secs_f64() * 1000.0,
            rows.len()
        );

        let indexing_start = std::time::Instant::now();
        let row_count = rows.len();
        let total_clicks = row_count; // Total number of engagement events (clicks + scrolls)
        for (path, timestamp, query, episode_queries_json) in rows {
            let click_event = ClickEvent { timestamp };

            // Index by (query, file_path) first
            clicks_by_query_and_file
                .entry((query, path.clone()))
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
                        .entry((episode_query, path.clone()))
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
        log::debug!(
            "Loaded {} total engagements from last 30 days",
            total_clicks
        );
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

        Ok((
            ClickData {
                clicks_by_file,
                clicks_by_parent_dir,
                clicks_by_query_and_file,
                engagements_by_episode_query_and_file,
            },
            total_clicks,
        ))
    }

    /// Compute simple score for cold-start ranking
    /// Formula: CLICKS_WEIGHT * clicks_last_7_days + RECENCY_WEIGHT / (1 + modified_age_in_days)
    fn compute_simple_score(&self, file: &FileCandidate, current_timestamp: i64) -> f64 {
        // Count clicks in last 7 days
        let seven_days_ago = current_timestamp - (7 * 24 * 60 * 60);
        let full_path_str = file.full_path.to_string_lossy().to_string();
        let clicks_last_7_days = self
            .clicks
            .clicks_by_file
            .get(&full_path_str)
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

        // Combine: clicks are weighted 3x more than recency
        CLICKS_WEIGHT * clicks_last_7_days + RECENCY_WEIGHT / (1.0 + modified_age_in_days)
    }

    /// Compute blend weights using sigmoid function
    /// Returns: (w_simple, w_lightgbm) where weights sum to 1.0
    fn compute_blend_weights(total_clicks: usize) -> (f64, f64) {
        let k = 20f64;
        let l = 2f64;

        let x = total_clicks as f64;
        let ml_weight = (1.0 + (x / k - l).tanh()) / 2.0;

        let simple_weight = 1.0 - ml_weight;
        (simple_weight, ml_weight)
    }

    pub fn rank_files(
        &mut self,
        query: &str,
        files: &[FileCandidate],
        current_timestamp: i64,
        cwd: &Path,
    ) -> Result<Vec<FileScore>> {
        if files.is_empty() {
            return Ok(Vec::new());
        }

        // Compute simple scores for all files (used for cold-start or blending)
        let simple_start = Instant::now();
        let simple_scores: Vec<f64> = files
            .iter()
            .map(|file| self.compute_simple_score(file, current_timestamp))
            .collect();
        log::info!(
            "TIMING {{\"op\":\"simple_score_compute\",\"ms\":{},\"count\":{}}}",
            simple_start.elapsed().as_secs_f64() * 1000.0,
            files.len()
        );

        // Always compute features for all files in parallel (for debugging visibility)
        let compute_start = Instant::now();

        // Capture what we need for parallel computation
        let clicks_by_file = &self.clicks.clicks_by_file;
        let clicks_by_parent_dir = &self.clicks.clicks_by_parent_dir;
        let clicks_by_query_and_file = &self.clicks.clicks_by_query_and_file;
        let engagements_by_episode_query_and_file =
            &self.clicks.engagements_by_episode_query_and_file;

        let click_indexes = FeatureClickIndexes {
            clicks_by_file,
            clicks_by_parent_dir,
            clicks_by_query_and_file,
            engagements_by_episode_query_and_file,
        };

        let results: Vec<(Vec<f64>, FxHashMap<String, Duration>)> = files
            .par_iter()
            .map(|file| {
                compute_features_with_timing(query, file, current_timestamp, cwd, &click_indexes)
                    .expect("Feature computation failed")
            })
            .collect();

        // Extract features and aggregate timings
        let all_features: Vec<Vec<f64>> = results.iter().map(|(f, _)| f.clone()).collect();

        // Aggregate per-feature timings across all files
        let mut aggregated_timings: FxHashMap<String, Duration> = FxHashMap::default();
        for (_features, timings) in &results {
            for (feature_name, duration) in timings {
                *aggregated_timings
                    .entry(feature_name.clone())
                    .or_insert(Duration::ZERO) += *duration;
            }
        }

        log::info!(
            "TIMING {{\"op\":\"ml_compute_features\",\"ms\":{},\"count\":{}}}",
            compute_start.elapsed().as_secs_f64() * 1000.0,
            files.len()
        );

        // Log per-feature timings (average per file)
        let num_files = files.len() as f64;
        for (feature_name, total_duration) in &aggregated_timings {
            let avg_ms = total_duration.as_secs_f64() * 1000.0 / num_files;
            log::info!(
                "TIMING {{\"op\":\"ml_feature_{}\",\"avg_ms\":{},\"total_ms\":{}}}",
                feature_name,
                avg_ms,
                total_duration.as_secs_f64() * 1000.0
            );
        }

        // If no model, use only simple scores (but keep the computed features for debugging)
        if self.model.is_none() {
            let mut scored_files: Vec<FileScore> = files
                .iter()
                .enumerate()
                .map(|(idx, file)| FileScore {
                    file_id: file.file_id,
                    score: simple_scores[idx],
                    features: all_features[idx].clone(),
                    simple_score: Some(simple_scores[idx]),
                    ml_score: None,
                    simple_weight: Some(1.0), // 100% simple model when no ML model
                    ml_weight: Some(0.0),
                })
                .collect();
            scored_files.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            return Ok(scored_files);
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
        log::info!(
            "TIMING {{\"op\":\"ml_predict\",\"ms\":{},\"count\":{}}}",
            predict_start.elapsed().as_secs_f64() * 1000.0,
            files.len()
        );

        // Compute blend weights using softmax
        let blend_start = Instant::now();
        let (w_simple, w_lightgbm) = Self::compute_blend_weights(self.total_clicks);
        log::debug!(
            "Hybrid ranking weights: simple={:.4}, lightgbm={:.4} (total_clicks={})",
            w_simple,
            w_lightgbm,
            self.total_clicks
        );

        // Build scored files with hybrid blending
        let mut scored_files = Vec::with_capacity(files.len());
        for (idx, file) in files.iter().enumerate() {
            let simple_score = simple_scores[idx];
            let ml_score = prediction_results[idx];
            let blended_score = w_simple * simple_score + w_lightgbm * ml_score;

            scored_files.push(FileScore {
                file_id: file.file_id,
                score: blended_score,
                features: all_features[idx].clone(),
                simple_score: Some(simple_score),
                ml_score: Some(ml_score),
                simple_weight: Some(w_simple),
                ml_weight: Some(w_lightgbm),
            });
        }
        log::info!(
            "TIMING {{\"op\":\"hybrid_blend\",\"ms\":{},\"count\":{}}}",
            blend_start.elapsed().as_secs_f64() * 1000.0,
            files.len()
        );

        // Sort by score descending (higher scores first)
        scored_files.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored_files)
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

struct FeatureClickIndexes<'a> {
    clicks_by_file: &'a FxHashMap<String, Vec<ClickEvent>>,
    clicks_by_parent_dir: &'a FxHashMap<PathBuf, Vec<ClickEvent>>,
    clicks_by_query_and_file: &'a FxHashMap<(String, String), Vec<ClickEvent>>,
    engagements_by_episode_query_and_file: &'a FxHashMap<(String, String), Vec<ClickEvent>>,
}

#[cfg(test)]
/// Compute features for a file (for tests, no timing)
fn compute_features(
    query: &str,
    file: &FileCandidate,
    current_timestamp: i64,
    cwd: &Path,
    clicks_by_file: &FxHashMap<String, Vec<ClickEvent>>,
    clicks_by_parent_dir: &FxHashMap<PathBuf, Vec<ClickEvent>>,
    clicks_by_query_and_file: &FxHashMap<(String, String), Vec<ClickEvent>>,
    engagements_by_episode_query_and_file: &FxHashMap<(String, String), Vec<ClickEvent>>,
) -> Result<Vec<f64>> {
    let indexes = FeatureClickIndexes {
        clicks_by_file,
        clicks_by_parent_dir,
        clicks_by_query_and_file,
        engagements_by_episode_query_and_file,
    };
    let (features, _timings) =
        compute_features_with_timing(query, file, current_timestamp, cwd, &indexes)?;
    Ok(features)
}

/// Compute features with timing for each feature
fn compute_features_with_timing(
    query: &str,
    file: &FileCandidate,
    current_timestamp: i64,
    cwd: &Path,
    click_indexes: &FeatureClickIndexes<'_>,
) -> Result<(Vec<f64>, FxHashMap<String, Duration>)> {
    // Create FeatureInputs for inference
    let inputs = FeatureInputs {
        query,
        file_path: &file.relative_path,
        full_path: &file.full_path,
        mtime: file.mtime,
        file_size: file.file_size,
        cwd,
        clicks_by_file: click_indexes.clicks_by_file,
        clicks_by_parent_dir: click_indexes.clicks_by_parent_dir,
        clicks_by_query_and_file: click_indexes.clicks_by_query_and_file,
        engagements_by_episode_query_and_file: click_indexes.engagements_by_episode_query_and_file,
        current_timestamp,
        session: None,
        is_from_walker: file.is_from_walker,
        is_dir: file.is_dir,
    };

    // Compute all features using the registry, tracking time for each
    let mut features = Vec::with_capacity(FEATURE_REGISTRY.len());
    let mut timings = FxHashMap::default();

    for feature in FEATURE_REGISTRY.iter() {
        let start = Instant::now();
        let value = feature.compute(&inputs)?;
        let elapsed = start.elapsed();

        features.push(value);
        timings.insert(feature.name().to_string(), elapsed);
    }

    Ok((features, timings))
}

/// Find train.py by searching: binary dir -> parent dir -> grandparent dir
fn find_train_py() -> Result<PathBuf> {
    let exe_path = std::env::current_exe().context("Failed to get current executable path")?;

    // Canonicalize to resolve symlinks
    let exe_path = exe_path
        .canonicalize()
        .context("Failed to canonicalize executable path")?;

    let exe_dir = exe_path
        .parent()
        .context("Failed to get executable directory")?;

    // Search in 3 levels: exe_dir, parent, grandparent
    for level in 0..3 {
        let mut search_dir = exe_dir.to_path_buf();
        for _ in 0..level {
            search_dir = search_dir
                .parent()
                .context("No parent directory")?
                .to_path_buf();
        }

        let train_py = search_dir.join("train.py");
        if train_py.exists() {
            log::debug!("Found train.py at {:?}", train_py);
            return Ok(train_py);
        }
    }

    anyhow::bail!(
        "train.py not found in binary directory or 2 levels above. Binary directory: {:?}",
        exe_dir
    )
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

        features::generate_features(
            &db_path,
            &features_csv,
            &schema_json,
            features::OutputFormat::Csv,
        )?;
        let feature_duration = feature_start.elapsed();
        log::info!(
            "Features generated at {:?} ({:.2}s)",
            features_csv,
            feature_duration.as_secs_f64()
        );

        // Step 2: Find train.py
        let train_py = find_train_py()?;
        log::info!("Found train.py at {:?}", train_py);

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

        let ranker = Ranker::new(&model_path, &db_path);
        match &ranker {
            Ok(_) => println!("✓ Ranker loaded successfully"),
            Err(e) => {
                eprintln!("✗ Failed to load ranker: {}", e);
                panic!("Ranker failed to load: {}", e);
            }
        }

        let mut ranker = ranker.unwrap();

        // Test with simple data
        let test_files = vec![FileCandidate {
            file_id: 0,
            relative_path: "test.md".to_string(),
            full_path: PathBuf::from("/tmp/test.md"),
            mtime: Some(1234567890),
            file_size: Some(2048),
            is_from_walker: true,
            is_dir: false,
        }];

        let result = ranker.rank_files(
            "test",
            &test_files,
            Timestamp::now().as_second(),
            &PathBuf::from("/tmp"),
        );
        match &result {
            Ok(file_scores) => {
                println!("✓ Ranking succeeded");
                for fs in file_scores {
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
            relative_path: "foo/bar.txt".to_string(),
            full_path: PathBuf::from("/tmp/foo/bar.txt"),
            mtime: Some(1700000000i64), // Nov 14, 2023
            file_size: Some(12_288),
            is_from_walker: true,
            is_dir: false,
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

        // Build (query, file) index - add 2 query-specific clicks
        let mut clicks_by_query_and_file = FxHashMap::default();
        clicks_by_query_and_file.insert(
            (query.to_string(), "/tmp/foo/bar.txt".to_string()),
            vec![
                ClickEvent {
                    timestamp: 1700000000,
                },
                ClickEvent {
                    timestamp: 1700010000,
                },
            ],
        );

        // Build episode engagement index
        let engagements_by_episode_query_and_file = FxHashMap::default();

        // Compute features using the standalone function
        let features = compute_features(
            query,
            &file,
            current_timestamp,
            &cwd,
            &clicks_by_file,
            &clicks_by_parent_dir,
            &clicks_by_query_and_file,
            &engagements_by_episode_query_and_file,
        )
        .expect("Failed to compute features");

        // Format as string for expect-test style comparison
        let actual = format!("{:?}", features);

        // Expected output: [filename_starts_with_query, clicks_last_30_days, modified_today, is_under_cwd, is_hidden, file_size_bytes, clicks_last_week_parent_dir, clicks_last_hour, clicks_today, clicks_last_7_days, modified_age, clicks_for_this_query, engagements_in_episode_with_query, is_dir]
        // filename_starts_with_query=0 (bar.txt doesn't start with "test")
        // clicks_last_30_days=3 (3 clicks on bar.txt itself)
        // modified_today=0 (mtime is old - Nov 2023, test runs in 2025)
        // is_under_cwd=1 (is_from_walker=true, so guaranteed to be under cwd)
        // is_hidden=0 (no dot-prefixed components)
        // file_size_bytes=12288 (12 KB file)
        // clicks_last_week_parent_dir=4 (3 clicks on bar.txt + 1 click on other.txt in /tmp/foo/)
        // clicks_last_hour=0 (clicks are old)
        // clicks_today=0 (clicks are old)
        // clicks_last_7_days=3 (all 3 clicks are within the last 7 days of the test timestamp)
        // modified_age=86400 (1 day in seconds)
        // clicks_for_this_query=2 (2 query-specific clicks for "test" + bar.txt)
        // engagements_in_episode_with_query=0 (no episode engagement data in test)
        // is_dir=0 (this is a file, not a directory)
        let expected =
            "[0.0, 3.0, 0.0, 1.0, 0.0, 12288.0, 4.0, 0.0, 0.0, 3.0, 86400.0, 2.0, 0.0, 0.0]";

        assert_eq!(actual, expected, "Feature vector mismatch");
    }

    #[test]
    fn test_compute_blend_weights() {
        // Test sigmoid blend weights at different click counts

        // 0 clicks: simple model should dominate (ML ≈ 2%)
        let (w_simple, w_lightgbm) = Ranker::compute_blend_weights(0);
        assert!(
            w_simple > 0.97 && w_simple < 0.99,
            "At 0 clicks, w_simple should be ~0.98, got {}",
            w_simple
        );
        assert!(
            w_lightgbm > 0.01 && w_lightgbm < 0.03,
            "At 0 clicks, w_lightgbm should be ~0.02, got {}",
            w_lightgbm
        );
        assert!(
            (w_simple + w_lightgbm - 1.0).abs() < 0.0001,
            "Weights should sum to 1.0, got {}",
            w_simple + w_lightgbm
        );

        // 100 clicks: ML model should dominate (simple nearly zero)
        let (w_simple, w_lightgbm) = Ranker::compute_blend_weights(100);
        assert!(
            w_simple < 0.01,
            "At 100 clicks, w_simple should be near 0.0, got {}",
            w_simple
        );
        assert!(
            w_lightgbm > 0.99,
            "At 100 clicks, w_lightgbm should be near 1.0, got {}",
            w_lightgbm
        );

        // 1000 clicks: ML model should dominate completely
        let (w_simple, w_lightgbm) = Ranker::compute_blend_weights(1000);
        assert!(
            w_simple < 0.001,
            "At 1000 clicks, w_simple should be ~0.0, got {}",
            w_simple
        );
        assert!(
            w_lightgbm > 0.999,
            "At 1000 clicks, w_lightgbm should be ~1.0, got {}",
            w_lightgbm
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
                clicks_by_parent_dir: FxHashMap::default(),
                clicks_by_query_and_file: FxHashMap::default(),
                engagements_by_episode_query_and_file: FxHashMap::default(),
            },
            stats: None,
            total_clicks: 2,
        };

        let current_timestamp = 1700604800; // Nov 22, 2023
        let file = FileCandidate {
            file_id: 0,
            relative_path: "foo.txt".to_string(),
            full_path: PathBuf::from("/tmp/foo.txt"),
            mtime: Some(1700500000), // Recent (1 day ago)
            file_size: Some(1024),
            is_from_walker: true,
            is_dir: false,
        };

        let score = ranker.compute_simple_score(&file, current_timestamp);

        // Score should be: 3.0 * 2 (clicks) + 1.0 / (1 + ~1.2 days)
        // = 6.0 + ~0.45 = ~6.45
        assert!(
            score > 6.0 && score < 7.0,
            "Simple score should be ~6.45, got {}",
            score
        );

        // Test file with no clicks and old mtime
        let old_file = FileCandidate {
            file_id: 1,
            relative_path: "bar.txt".to_string(),
            full_path: PathBuf::from("/tmp/bar.txt"),
            mtime: Some(1600000000), // Very old
            file_size: Some(2048),
            is_from_walker: true,
            is_dir: false,
        };

        let score = ranker.compute_simple_score(&old_file, current_timestamp);

        // Score should be: 3.0 * 0 (no clicks) + 1.0 / (1 + many days)
        // = 0.0 + very small positive number
        assert!(
            score > 0.0 && score < 0.1,
            "Old file with no clicks should have very low score, got {}",
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
                clicks_by_parent_dir: FxHashMap::default(),
                clicks_by_query_and_file: FxHashMap::default(),
                engagements_by_episode_query_and_file: FxHashMap::default(),
            },
            stats: None,
            total_clicks: 3,
        };

        let current_timestamp = 1700604800;
        let files = vec![
            FileCandidate {
                file_id: 0,
                relative_path: "popular.txt".to_string(),
                full_path: PathBuf::from("/tmp/popular.txt"),
                mtime: Some(1700000000),
                file_size: Some(1024),
                is_from_walker: true,
                is_dir: false,
            },
            FileCandidate {
                file_id: 1,
                relative_path: "recent.txt".to_string(),
                full_path: PathBuf::from("/tmp/recent.txt"),
                mtime: Some(1700600000), // Very recent
                file_size: Some(2048),
                is_from_walker: true,
                is_dir: false,
            },
        ];

        let result = ranker
            .rank_files("", &files, current_timestamp, &PathBuf::from("/tmp"))
            .expect("Ranking should succeed");

        assert_eq!(result.len(), 2, "Should have 2 ranked files");

        // popular.txt should rank first due to clicks (3 * 3 = 9 points)
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
