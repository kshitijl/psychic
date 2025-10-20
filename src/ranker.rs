use anyhow::{Context, Result};
use jiff::Timestamp;
use lightgbm3::Booster;
use rayon::prelude::*;
use rusqlite::Connection;
use std::collections::HashMap;
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
    pub is_from_walker: bool,
    pub is_dir: bool,
}

#[derive(Debug, Clone)]
pub struct FileScore {
    pub file_id: usize, // Index into main.rs file registry
    pub score: f64,
    pub features: Vec<f64>, // Feature vector in registry order
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
    pub clicks_by_file: HashMap<String, Vec<ClickEvent>>,
    pub clicks_by_parent_dir: HashMap<PathBuf, Vec<ClickEvent>>,
    pub clicks_by_query_and_file: HashMap<(String, String), Vec<ClickEvent>>,
}

pub struct Ranker {
    model: Booster,
    pub clicks: ClickData,
    pub stats: Option<ModelStats>,
}

impl Ranker {
    pub fn new(model_path: &Path, db_path: &Path) -> Result<Self> {
        let model_load_start = std::time::Instant::now();
        let model = Booster::from_file(model_path.to_str().unwrap())
            .context("Failed to load LightGBM model")?;
        log::info!("TIMING {{\"op\":\"booster_from_file\",\"ms\":{}}}", model_load_start.elapsed().as_secs_f64() * 1000.0);

        let clicks_load_start = std::time::Instant::now();
        let clicks = Self::load_clicks(db_path)?;
        log::info!("TIMING {{\"op\":\"load_clicks\",\"ms\":{}}}", clicks_load_start.elapsed().as_secs_f64() * 1000.0);

        // Load model stats from same directory as model
        let stats_path = model_path
            .parent()
            .map(|p| p.join("model_stats.json"))
            .unwrap_or_else(|| PathBuf::from("model_stats.json"));
        let stats = Self::load_stats(&stats_path);

        Ok(Ranker {
            model,
            clicks,
            stats,
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
    pub fn load_clicks(db_path: &Path) -> Result<ClickData> {
        let total_start = std::time::Instant::now();

        let timestamp_calc_start = std::time::Instant::now();
        let now = Timestamp::now();
        let now_ts = now.as_second();
        // Simple arithmetic: 30 days = 30 * 24 * 60 * 60 seconds
        let thirty_days_ago_ts = now_ts - (30 * 24 * 60 * 60);
        log::info!("TIMING {{\"op\":\"timestamp_calc\",\"ms\":{}}}", timestamp_calc_start.elapsed().as_secs_f64() * 1000.0);

        let mut clicks_by_file: HashMap<String, Vec<ClickEvent>> = HashMap::new();
        let mut clicks_by_query_and_file: HashMap<(String, String), Vec<ClickEvent>> = HashMap::new();

        let db_open_start = std::time::Instant::now();
        let conn =
            Connection::open(db_path).context("Failed to open database for preloading clicks")?;
        log::info!("TIMING {{\"op\":\"db_open\",\"ms\":{}}}", db_open_start.elapsed().as_secs_f64() * 1000.0);

        let query_start = std::time::Instant::now();
        let mut stmt = conn.prepare(
            "SELECT full_path, timestamp, query
             FROM events
             WHERE action = 'click'
             AND timestamp >= ?1",
        )?;

        let rows = stmt.query_map([thirty_days_ago_ts], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        log::info!("TIMING {{\"op\":\"db_query_prepare\",\"ms\":{}}}", query_start.elapsed().as_secs_f64() * 1000.0);

        let indexing_start = std::time::Instant::now();
        let mut row_count = 0;
        for row in rows {
            let (path, timestamp, query) = row?;
            let click_event = ClickEvent { timestamp };
            row_count += 1;

            // Index by file path only
            clicks_by_file
                .entry(path.clone())
                .or_default()
                .push(click_event);

            // Index by (query, file_path)
            clicks_by_query_and_file
                .entry((query, path))
                .or_default()
                .push(click_event);
        }
        log::info!("TIMING {{\"op\":\"process_click_rows\",\"ms\":{},\"count\":{}}}", indexing_start.elapsed().as_secs_f64() * 1000.0, row_count);

        // Build parent directory index
        let parent_dir_start = std::time::Instant::now();
        let mut clicks_by_parent_dir: HashMap<PathBuf, Vec<ClickEvent>> = HashMap::new();
        for (path, clicks) in &clicks_by_file {
            if let Some(parent) = Path::new(path).parent() {
                clicks_by_parent_dir
                    .entry(parent.to_path_buf())
                    .or_default()
                    .extend(clicks.iter().copied());
            }
        }
        log::info!("TIMING {{\"op\":\"build_parent_dir_index\",\"ms\":{}}}", parent_dir_start.elapsed().as_secs_f64() * 1000.0);

        log::info!("TIMING {{\"op\":\"load_clicks_total\",\"ms\":{}}}", total_start.elapsed().as_secs_f64() * 1000.0);
        log::debug!(
            "Loaded {} files with click history from last 30 days",
            clicks_by_file.len()
        );
        log::debug!("Indexed {} parent directories", clicks_by_parent_dir.len());
        log::debug!(
            "Indexed {} (query, file) pairs",
            clicks_by_query_and_file.len()
        );

        Ok(ClickData {
            clicks_by_file,
            clicks_by_parent_dir,
            clicks_by_query_and_file,
        })
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

        // Compute features for all files in parallel
        let compute_start = Instant::now();

        // Capture what we need for parallel computation
        let clicks_by_file = &self.clicks.clicks_by_file;
        let clicks_by_parent_dir = &self.clicks.clicks_by_parent_dir;
        let clicks_by_query_and_file = &self.clicks.clicks_by_query_and_file;

        let all_features: Vec<Vec<f64>> = files
            .par_iter()
            .map(|file| {
                compute_features(
                    query,
                    file,
                    current_timestamp,
                    cwd,
                    clicks_by_file,
                    clicks_by_parent_dir,
                    clicks_by_query_and_file,
                )
                .expect("Feature computation failed")
            })
            .collect();

        log::debug!(
            "Parallel feature computation for {} files took {:?}",
            files.len(),
            compute_start.elapsed()
        );

        // Batch predict all files at once
        // Flatten features into a single vector for batch prediction
        let flatten_start = Instant::now();
        let num_features = if all_features.is_empty() {
            0
        } else {
            all_features[0].len()
        };
        let flat_features: Vec<f64> = all_features.iter().flatten().copied().collect();
        log::debug!(
            "Flattening features for {} files took {:?}",
            files.len(),
            flatten_start.elapsed()
        );

        let predict_start = Instant::now();
        let prediction_results = self
            .model
            .predict_with_params(&flat_features, num_features as i32, true, "num_threads=8")
            .context("Failed to batch predict with model")?;
        log::debug!(
            "Batch model prediction for {} files took {:?}",
            files.len(),
            predict_start.elapsed()
        );

        // Build scored files
        let mut scored_files = Vec::with_capacity(files.len());
        for (idx, file) in files.iter().enumerate() {
            scored_files.push(FileScore {
                file_id: file.file_id,
                score: prediction_results[idx],
                features: all_features[idx].clone(),
            });
        }

        // Sort by score descending (higher scores first)
        let sort_start = Instant::now();
        scored_files.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        log::debug!(
            "Sorting {} scored files took {:?}",
            scored_files.len(),
            sort_start.elapsed()
        );

        Ok(scored_files)
    }
}

/// Convert feature vector to HashMap (for display purposes)
pub fn features_to_map(features: &[f64]) -> HashMap<String, f64> {
    feature_names()
        .iter()
        .zip(features.iter())
        .map(|(name, value)| (name.to_string(), *value))
        .collect()
}

/// Compute features for a file (for tests, no timing)
fn compute_features(
    query: &str,
    file: &FileCandidate,
    current_timestamp: i64,
    cwd: &Path,
    clicks_by_file: &HashMap<String, Vec<ClickEvent>>,
    clicks_by_parent_dir: &HashMap<PathBuf, Vec<ClickEvent>>,
    clicks_by_query_and_file: &HashMap<(String, String), Vec<ClickEvent>>,
) -> Result<Vec<f64>> {
    let (features, _timings) = compute_features_with_timing(
        query,
        file,
        current_timestamp,
        cwd,
        clicks_by_file,
        clicks_by_parent_dir,
        clicks_by_query_and_file,
    )?;
    Ok(features)
}

/// Compute features with timing for each feature
fn compute_features_with_timing(
    query: &str,
    file: &FileCandidate,
    current_timestamp: i64,
    cwd: &Path,
    clicks_by_file: &HashMap<String, Vec<ClickEvent>>,
    clicks_by_parent_dir: &HashMap<PathBuf, Vec<ClickEvent>>,
    clicks_by_query_and_file: &HashMap<(String, String), Vec<ClickEvent>>,
) -> Result<(Vec<f64>, HashMap<String, Duration>)> {
    // Create FeatureInputs for inference
    let inputs = FeatureInputs {
        query,
        file_path: &file.relative_path,
        full_path: &file.full_path,
        mtime: file.mtime,
        cwd,
        clicks_by_file,
        clicks_by_parent_dir,
        clicks_by_query_and_file,
        current_timestamp,
        session: None,
        is_from_walker: file.is_from_walker,
        is_dir: file.is_dir,
    };

    // Compute all features using the registry, tracking time for each
    let mut features = Vec::with_capacity(FEATURE_REGISTRY.len());
    let mut timings = HashMap::new();

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
            is_from_walker: true,
            is_dir: false,
        };

        // Create synthetic click data
        let mut clicks_by_file = HashMap::new();
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
        let mut clicks_by_parent_dir = HashMap::new();
        for (path, clicks) in &clicks_by_file {
            if let Some(parent) = Path::new(path).parent() {
                clicks_by_parent_dir
                    .entry(parent.to_path_buf())
                    .or_insert_with(Vec::new)
                    .extend(clicks.iter().copied());
            }
        }

        // Build (query, file) index - add 2 query-specific clicks
        let mut clicks_by_query_and_file = HashMap::new();
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

        // Compute features using the standalone function
        let features = compute_features(
            query,
            &file,
            current_timestamp,
            &cwd,
            &clicks_by_file,
            &clicks_by_parent_dir,
            &clicks_by_query_and_file,
        )
        .expect("Failed to compute features");

        // Format as string for expect-test style comparison
        let actual = format!("{:?}", features);

        // Expected output: [filename_starts_with_query, clicks_last_30_days, modified_today, is_under_cwd, is_hidden, clicks_last_week_parent_dir, clicks_last_hour, clicks_today, clicks_last_7_days, modified_age, clicks_for_this_query, is_dir]
        // filename_starts_with_query=0 (bar.txt doesn't start with "test")
        // clicks_last_30_days=3 (3 clicks on bar.txt itself)
        // modified_today=0 (mtime is old - Nov 2023, test runs in 2025)
        // is_under_cwd=1 (is_from_walker=true, so guaranteed to be under cwd)
        // is_hidden=0 (no dot-prefixed components)
        // clicks_last_week_parent_dir=4 (3 clicks on bar.txt + 1 click on other.txt in /tmp/foo/)
        // clicks_last_hour=0 (clicks are old)
        // clicks_today=0 (clicks are old)
        // clicks_last_7_days=3 (all 3 clicks are within the last 7 days of the test timestamp)
        // modified_age=86400 (1 day in seconds)
        // clicks_for_this_query=2 (2 query-specific clicks for "test" + bar.txt)
        // is_dir=0 (this is a file, not a directory)
        let expected = "[0.0, 3.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 3.0, 86400.0, 2.0, 0.0]";

        assert_eq!(actual, expected, "Feature vector mismatch");
    }
}
