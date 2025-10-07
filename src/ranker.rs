use anyhow::{Context, Result};
use jiff::{Span, Timestamp};
use lightgbm3::Booster;
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// Import features module
use crate::feature_defs::{ClickEvent, FeatureInputs, FEATURE_REGISTRY, feature_names};

#[derive(Debug, Clone)]
pub struct FileScore {
    pub path: String,
    pub score: f64,
    pub features: HashMap<String, f64>,
}

pub struct Ranker {
    model: Booster,
    // Precomputed click data for last 30 days (full_path -> Vec of ClickEvent)
    clicks_by_file: HashMap<String, Vec<ClickEvent>>,
}

impl Ranker {
    pub fn new(model_path: &Path, db_path: PathBuf) -> Result<Self> {
        let model = Booster::from_file(model_path.to_str().unwrap())
            .context("Failed to load LightGBM model")?;

        // Preload all click events from last 30 days
        let now = Timestamp::now();
        let session_tz = jiff::tz::TimeZone::system();
        let now_zoned = now.to_zoned(session_tz);
        let thirty_days_ago = now_zoned.checked_sub(Span::new().days(30))?.timestamp();
        let thirty_days_ago_ts = thirty_days_ago.as_second();

        let mut clicks_by_file: HashMap<String, Vec<ClickEvent>> = HashMap::new();

        let conn = Connection::open(&db_path)
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

        Ok(Ranker {
            model,
            clicks_by_file,
        })
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
