use anyhow::{Context, Result};
use jiff::{Span, Timestamp};
use lightgbm3::Booster;
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct FileScore {
    pub path: String,
    pub score: f64,
    pub features: HashMap<String, f64>,
}

pub struct Ranker {
    model: Booster,
    click_counts: HashMap<String, usize>, // full_path -> click count in last 30 days
}

impl Ranker {
    pub fn new(model_path: &Path, db_path: PathBuf) -> Result<Self> {
        let model = Booster::from_file(model_path.to_str().unwrap())
            .context("Failed to load LightGBM model")?;

        // Preload all click data from last 30 days
        let now = Timestamp::now();
        let session_tz = jiff::tz::TimeZone::system();
        let now_zoned = now.to_zoned(session_tz);
        let thirty_days_ago = now_zoned.checked_sub(Span::new().days(30))?.timestamp();
        let thirty_days_ago_ts = thirty_days_ago.as_second();

        let mut click_counts = HashMap::new();

        let conn = Connection::open(&db_path)
            .context("Failed to open database for preloading clicks")?;

        let mut stmt = conn.prepare(
            "SELECT full_path, COUNT(*) as count
             FROM events
             WHERE action = 'click'
             AND timestamp >= ?1
             GROUP BY full_path"
        )?;

        let rows = stmt.query_map([thirty_days_ago_ts], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        })?;

        for row in rows {
            let (path, count) = row?;
            click_counts.insert(path, count);
        }

        eprintln!("Loaded {} click counts from last 30 days", click_counts.len());

        Ok(Ranker {
            model,
            click_counts,
        })
    }

    pub fn rank_files(
        &self,
        query: &str,
        files: Vec<(String, PathBuf, Option<i64>)>, // (relative_path, full_path, mtime)
        session_id: &str,
    ) -> Result<Vec<FileScore>> {
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

            scored_files.push(FileScore {
                path: relative_path,
                score,
                features: features.clone(),
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

        // clicks_last_30_days - lookup from preloaded data
        let clicks_last_30_days = self.click_counts
            .get(&full_path.to_string_lossy().to_string())
            .copied()
            .unwrap_or(0);
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
