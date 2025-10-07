use anyhow::{Context, Result};
use jiff::{Span, Timestamp};
use lightgbm3::Booster;
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct Ranker {
    model: Booster,
    db_path: PathBuf,
}

impl Ranker {
    pub fn new(model_path: &Path, db_path: PathBuf) -> Result<Self> {
        let model = Booster::from_file(model_path.to_str().unwrap())
            .context("Failed to load LightGBM model")?;
        Ok(Ranker { model, db_path })
    }

    pub fn rank_files(
        &self,
        query: &str,
        files: Vec<(String, PathBuf, Option<i64>)>, // (relative_path, full_path, mtime)
        session_id: &str,
    ) -> Result<Vec<String>> {
        if files.is_empty() {
            return Ok(Vec::new());
        }

        // Compute features for each file
        let mut scored_files = Vec::new();
        for (relative_path, full_path, mtime) in files {
            let features = self.compute_features(
                query,
                &relative_path,
                &full_path,
                mtime,
                session_id,
            )?;

            // Convert features to vector in the same order as training
            let feature_vec = self.features_to_vec(&features);

            // Get prediction score
            let score = self.model
                .predict_with_params(&feature_vec, feature_vec.len() as i32, true, "num_threads=1")
                .context("Failed to predict with model")?[0];

            scored_files.push((relative_path, score));
        }

        // Sort by score descending (higher scores first)
        scored_files.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored_files.into_iter().map(|(path, _)| path).collect())
    }

    fn compute_features(
        &self,
        query: &str,
        file_path: &str,
        full_path: &Path,
        mtime: Option<i64>,
        session_id: &str,
    ) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();

        // filename_starts_with_query (binary feature)
        let filename = Path::new(file_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let filename_starts_with_query = if !query.is_empty()
            && filename.to_lowercase().starts_with(&query.to_lowercase())
        {
            1.0
        } else {
            0.0
        };
        features.insert("filename_starts_with_query".to_string(), filename_starts_with_query);

        // clicks_last_30_days - query the database
        let clicks_last_30_days = self.count_clicks_last_30_days(full_path, session_id)?;
        features.insert("clicks_last_30_days".to_string(), clicks_last_30_days as f64);

        // modified_today (binary feature)
        let modified_today = if let Some(mtime) = mtime {
            let now = Timestamp::now();
            let mtime_ts = Timestamp::from_second(mtime)?;
            let diff = now.duration_since(mtime_ts);
            if diff.as_secs() < 24 * 3600 {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        features.insert("modified_today".to_string(), modified_today);

        Ok(features)
    }

    fn count_clicks_last_30_days(&self, full_path: &Path, session_id: &str) -> Result<usize> {
        let conn = Connection::open(&self.db_path)
            .context("Failed to open database for click counting")?;

        // Get timezone for the session
        let timezone: String = conn
            .query_row(
                "SELECT timezone FROM sessions WHERE session_id = ?1",
                [session_id],
                |row| row.get(0),
            )
            .unwrap_or_else(|_| "UTC".to_string());

        let now = Timestamp::now();
        let session_tz =
            jiff::tz::TimeZone::get(&timezone).unwrap_or(jiff::tz::TimeZone::system());
        let now_zoned = now.to_zoned(session_tz);
        let thirty_days_ago = now_zoned.checked_sub(Span::new().days(30))?.timestamp();

        let count: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM events
                 WHERE action = 'click'
                 AND full_path = ?1
                 AND timestamp >= ?2
                 AND timestamp <= ?3",
                [
                    full_path.to_string_lossy().to_string(),
                    thirty_days_ago.as_second().to_string(),
                    now.as_second().to_string(),
                ],
                |row| row.get(0),
            )
            .unwrap_or(0);

        Ok(count)
    }

    fn features_to_vec(&self, features: &HashMap<String, f64>) -> Vec<f64> {
        // MUST match the order in training CSV:
        // filename_starts_with_query, clicks_last_30_days, modified_today
        vec![
            features.get("filename_starts_with_query").copied().unwrap_or(0.0),
            features.get("clicks_last_30_days").copied().unwrap_or(0.0),
            features.get("modified_today").copied().unwrap_or(0.0),
        ]
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

        let result = ranker.rank_files("test", test_files, "test_session");
        match &result {
            Ok(files) => println!("✓ Ranking succeeded: {:?}", files),
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
