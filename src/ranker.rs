use anyhow::{Context, Result};
use jiff::{Span, Timestamp};
use lightgbm3::Booster;
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Instant;

// Import features module
use crate::feature_defs::{ClickEvent, FeatureInputs, FEATURE_REGISTRY, feature_names};
use crate::{db, features};

#[derive(Debug, Clone)]
pub struct FileScore {
    pub path: String,
    pub score: f64,
    pub features: HashMap<String, f64>,
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

pub struct Ranker {
    model: Booster,
    // Precomputed click data for last 30 days (full_path -> Vec of ClickEvent)
    pub clicks_by_file: HashMap<String, Vec<ClickEvent>>,
    pub db_path: PathBuf,
    pub stats: Option<ModelStats>,
    pub loaded_at: Timestamp,
    pub clicks_loaded_at: Timestamp,
}

impl Ranker {
    pub fn new(model_path: &Path, db_path: PathBuf) -> Result<Self> {
        let model = Booster::from_file(model_path.to_str().unwrap())
            .context("Failed to load LightGBM model")?;

        let now = Timestamp::now();
        let clicks_by_file = Self::load_clicks(&db_path)?;

        // Load model stats from same directory as model
        let stats_path = model_path.parent()
            .map(|p| p.join("model_stats.json"))
            .unwrap_or_else(|| PathBuf::from("model_stats.json"));
        let stats = Self::load_stats(&stats_path);

        Ok(Ranker {
            model,
            clicks_by_file,
            db_path,
            stats,
            loaded_at: now,
            clicks_loaded_at: now,
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
    pub fn load_clicks(db_path: &PathBuf) -> Result<HashMap<String, Vec<ClickEvent>>> {
        let now = Timestamp::now();
        let session_tz = jiff::tz::TimeZone::system();
        let now_zoned = now.to_zoned(session_tz);
        let thirty_days_ago = now_zoned.checked_sub(Span::new().days(30))?.timestamp();
        let thirty_days_ago_ts = thirty_days_ago.as_second();

        let mut clicks_by_file: HashMap<String, Vec<ClickEvent>> = HashMap::new();

        let conn = Connection::open(db_path)
            .context("Failed to open database for preloading clicks")?;

        let mut stmt = conn.prepare(
            "SELECT full_path, timestamp
             FROM events
             WHERE action = 'click'
             AND timestamp >= ?1"
        )?;

        let rows = stmt.query_map([thirty_days_ago_ts], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;

        for row in rows {
            let (path, timestamp) = row?;
            clicks_by_file
                .entry(path)
                .or_default()
                .push(ClickEvent { timestamp });
        }

        log::debug!("Loaded {} files with click history from last 30 days", clicks_by_file.len());

        Ok(clicks_by_file)
    }

    pub fn rank_files(
        &self,
        query: &str,
        files: &[(String, PathBuf, Option<i64>)], // (relative_path, full_path, mtime)
        session_id: &str,
        cwd: &Path,
    ) -> Result<Vec<FileScore>> {
        if files.is_empty() {
            return Ok(Vec::new());
        }

        // Compute features for each file
        let mut scored_files = Vec::new();
        for (relative_path, full_path, mtime) in files {
            let features = self.compute_features(
                query,
                relative_path,
                full_path,
                *mtime,
                session_id,
                cwd,
            )?;

            // Convert features to vector in the same order as training
            let feature_vec = self.features_to_vec(&features);

            // Get prediction score
            let score = self.model
                .predict_with_params(&feature_vec, feature_vec.len() as i32, true, "num_threads=1")
                .context("Failed to predict with model")?[0];

            scored_files.push(FileScore {
                path: relative_path.clone(),
                score,
                features,
            });
        }

        // Sort by score descending (higher scores first)
        scored_files.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored_files)
    }

    fn compute_features(
        &self,
        query: &str,
        file_path: &str,
        full_path: &Path,
        mtime: Option<i64>,
        _session_id: &str,
        cwd: &Path,
    ) -> Result<HashMap<String, f64>> {
        // Create FeatureInputs for inference
        let inputs = FeatureInputs {
            query,
            file_path,
            full_path,
            mtime,
            cwd,
            clicks_by_file: &self.clicks_by_file,
            current_timestamp: Timestamp::now().as_second(),
            session: None, // No session context at inference time
        };

        // Compute all features using the registry
        let mut features = HashMap::new();
        for feature in FEATURE_REGISTRY.iter() {
            let value = feature.compute(&inputs)?;
            features.insert(feature.name().to_string(), value);
        }

        Ok(features)
    }

    fn features_to_vec(&self, features: &HashMap<String, f64>) -> Vec<f64> {
        // Automatically uses registry order - guaranteed to match training!
        feature_names()
            .iter()
            .map(|name| features.get(*name).copied().unwrap_or(0.0))
            .collect()
    }
}

/// Find train.py by searching: binary dir -> parent dir -> grandparent dir
fn find_train_py() -> Result<PathBuf> {
    let exe_path = std::env::current_exe().context("Failed to get current executable path")?;

    // Canonicalize to resolve symlinks
    let exe_path = exe_path.canonicalize()
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
        let db_path = db::Database::get_db_path(&data_dir)?;
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
        log::info!("Features generated at {:?} ({:.2}s)", features_csv, feature_duration.as_secs_f64());

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
        log::info!("Training complete! ({:.2}s)", training_duration.as_secs_f64());
        log::info!("Model saved at {:?}", output_prefix.with_extension("txt"));

        let total_duration = total_start.elapsed();
        log::info!("Total retraining time: {:.2}s", total_duration.as_secs_f64());

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

        let db_path = Database::get_db_path().expect("Failed to get db path");

        let ranker = Ranker::new(&model_path, db_path);
        match &ranker {
            Ok(_) => println!("✓ Ranker loaded successfully"),
            Err(e) => {
                eprintln!("✗ Failed to load ranker: {}", e);
                panic!("Ranker failed to load: {}", e);
            }
        }

        let ranker = ranker.unwrap();

        // Test with simple data
        let test_files = vec![
            ("test.md".to_string(), PathBuf::from("/tmp/test.md"), Some(1234567890)),
        ];

        let result = ranker.rank_files("test", &test_files, "test_session", &PathBuf::from("/tmp"));
        match &result {
            Ok(file_scores) => {
                println!("✓ Ranking succeeded");
                for fs in file_scores {
                    println!("  {} - score: {:.4}, features: {:?}", fs.path, fs.score, fs.features);
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
}
