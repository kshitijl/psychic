use anyhow::{Context, Result};
use csv::Writer;
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use rusqlite::Connection;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// Import the feature definitions module
use crate::feature_defs::{FEATURE_REGISTRY, FeatureInputs, csv_columns};

// Re-export types that other modules need
pub use crate::feature_defs::{ClickEvent, Session};

// Data structures for holding event and session data in memory

#[derive(Debug, Clone)]
struct Event {
    session_id: String,
    subsession_id: u64,
    query: String,
    file_path: String,
    full_path: String,
    timestamp: i64,
    mtime: Option<i64>,
    file_size: Option<i64>,
    action: String,
    episode_queries: Option<String>, // JSON array of queries in this episode
    /// What the row was when the user saw it, as the UI recorded it. `None` for
    /// rows written before the column existed, which fall back to a `stat`.
    is_dir: Option<bool>,
}

// Output format enum
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Csv,
    Json,
}

// Accumulator for fold-based processing
struct Accumulator {
    clicks_by_file: FxHashMap<String, Vec<ClickEvent>>,
    /// Directories the user changed into, from `startup_visit` events. Fed by
    /// the same fold over time-sorted events, so an impression only ever sees
    /// the visits that had happened by then.
    visits_by_dir: FxHashMap<String, Vec<ClickEvent>>,
    /// Engagements per extension, and the running total, the same pair
    /// `Ranker::load_clicks` builds.
    clicks_by_extension: FxHashMap<String, usize>,
    clicks_indexed: usize,
    clicks_by_parent_dir: FxHashMap<std::path::PathBuf, Vec<ClickEvent>>,
    /// query -> path -> events, the same shape `Ranker::load_clicks` builds.
    clicks_by_query_and_file: FxHashMap<String, FxHashMap<String, Vec<ClickEvent>>>,
    engagements_by_episode_query_and_file: FxHashMap<String, FxHashMap<String, Vec<ClickEvent>>>,
    // Key: (session_id, subsession_id, full_path)
    pending_impressions: FxHashMap<(String, u64, String), PendingImpression>,
    output_rows: Vec<HashMap<String, String>>,
    // Track current episode ID - increments with each click/scroll and session change
    current_episode_id: u64,
    // Track last session to detect session boundaries
    last_session_id: Option<String>,
}

#[derive(Debug, Clone)]
struct PendingImpression {
    features: HashMap<String, String>,
    timestamp: i64,
}

impl Accumulator {
    fn new() -> Self {
        Self {
            clicks_by_file: FxHashMap::default(),
            visits_by_dir: FxHashMap::default(),
            clicks_by_extension: FxHashMap::default(),
            clicks_indexed: 0,
            clicks_by_parent_dir: FxHashMap::default(),
            clicks_by_query_and_file: FxHashMap::default(),
            engagements_by_episode_query_and_file: FxHashMap::default(),
            pending_impressions: FxHashMap::default(),
            output_rows: Vec::new(),
            current_episode_id: 0,
            last_session_id: None,
        }
    }

    fn record_click(&mut self, event: &Event) {
        let click = ClickEvent {
            timestamp: event.timestamp,
        };
        self.clicks_by_file
            .entry(event.full_path.clone())
            .or_default()
            .push(click);

        // Also index by extension, and count the total the shares divide by
        *self
            .clicks_by_extension
            .entry(crate::feature_defs::extension_key(Path::new(
                &event.full_path,
            )))
            .or_default() += 1;
        self.clicks_indexed += 1;

        // Also index by parent directory
        if let Some(parent) = Path::new(&event.full_path).parent() {
            self.clicks_by_parent_dir
                .entry(parent.to_path_buf())
                .or_default()
                .push(click);
        }

        // Also index by (query, file_path)
        self.clicks_by_query_and_file
            .entry(event.query.clone())
            .or_default()
            .entry(event.full_path.clone())
            .or_default()
            .push(click);

        // Build episode query index if episode_queries is present
        if let Some(ref episode_json) = event.episode_queries
            && let Ok(episode_queries) = serde_json::from_str::<Vec<String>>(episode_json)
        {
            for episode_query in episode_queries {
                self.engagements_by_episode_query_and_file
                    .entry(episode_query)
                    .or_default()
                    .entry(event.full_path.clone())
                    .or_default()
                    .push(click);
            }
        }
    }

    /// Remember that the user changed into this directory.
    fn record_visit(&mut self, event: &Event) {
        self.visits_by_dir
            .entry(event.full_path.clone())
            .or_default()
            .push(ClickEvent {
                timestamp: event.timestamp,
            });
    }

    fn add_impression(&mut self, event: &Event, mut features: HashMap<String, String>) {
        // Check if this is a new session - if so, increment episode_id
        if let Some(ref last_session) = self.last_session_id
            && last_session != &event.session_id
        {
            self.current_episode_id += 1;
        }
        self.last_session_id = Some(event.session_id.clone());

        // Add episode_id to features
        features.insert(
            "episode_id".to_string(),
            self.current_episode_id.to_string(),
        );

        let key = (
            event.session_id.clone(),
            event.subsession_id,
            event.full_path.clone(),
        );
        self.pending_impressions.insert(
            key,
            PendingImpression {
                features,
                timestamp: event.timestamp,
            },
        );
    }

    fn mark_impressions_as_engaged(&mut self, event: &Event) {
        // Update last_session_id when processing click/scroll
        if let Some(ref last_session) = self.last_session_id
            && last_session != &event.session_id
        {
            self.current_episode_id += 1;
        }
        self.last_session_id = Some(event.session_id.clone());

        // Find impressions in same subsession with same file that happened BEFORE this click/scroll
        let key = (
            event.session_id.clone(),
            event.subsession_id,
            event.full_path.clone(),
        );

        if let Some(pending) = self.pending_impressions.get_mut(&key)
            && pending.timestamp <= event.timestamp
        {
            pending
                .features
                .insert("label".to_string(), "1".to_string());
        }

        // Increment episode_id after each engagement event (click or scroll)
        // This creates a new episode for the next sequence of impressions
        self.current_episode_id += 1;
    }

    fn finalize(mut self) -> Vec<HashMap<String, String>> {
        // Move all pending impressions to output in deterministic order
        // Sort by (session_id, subsession_id, full_path) to ensure consistent ordering
        let mut sorted_keys: Vec<_> = self.pending_impressions.keys().cloned().collect();
        sorted_keys.sort_by(|a, b| {
            a.0.cmp(&b.0) // session_id
                .then(a.1.cmp(&b.1)) // subsession_id
                .then(a.2.cmp(&b.2)) // full_path
        });

        for key in sorted_keys {
            if let Some(impression) = self.pending_impressions.remove(&key) {
                self.output_rows.push(impression.features);
            }
        }
        self.output_rows
    }
}

/// What came out of a feature generation run.
///
/// `positives` is the number that matter: an impression that was clicked or
/// scrolled. A model cannot be trained without at least one, which is the
/// state every psychic install starts in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureSummary {
    pub rows: usize,
    pub positives: usize,
}

// Main function to generate features

pub fn generate_features(
    db_path: &Path,
    output_path: &Path,
    schema_path: &Path,
    format: OutputFormat,
) -> Result<FeatureSummary> {
    // Through `Database` so this gets the same pragmas as everything else,
    // and so the schema exists when a fresh install retrains before the UI has
    // opened anything.
    let db = crate::db::Database::new(db_path)?;
    let conn = db.connection();
    let mut all_events = fetch_all_events(conn)?;
    let all_sessions = fetch_all_sessions(conn)?;

    // Sort events by timestamp (critical for temporal correctness)
    all_events.sort_by_key(|e| e.timestamp);

    let mut acc = Accumulator::new();
    // One matcher for the whole pass. Building one per row is what inference
    // used to do per file per keystroke, and it is not free.
    let matcher = SkimMatcherV2::default();

    for event in &all_events {
        match event.action.as_str() {
            "impression" => {
                let features =
                    compute_features_from_accumulator(event, &acc, &all_sessions, &matcher)?;
                acc.add_impression(event, features);
            }
            "click" => {
                acc.record_click(event);
                acc.mark_impressions_as_engaged(event);
            }
            "scroll" => {
                acc.mark_impressions_as_engaged(event);
            }
            "startup_visit" => {
                // A visit is not an engagement: it primes history and the
                // directory-visit features, and never counts as a click.
                acc.record_visit(event);
            }
            _ => {} // Ignore unknown actions
        }
    }

    let output_rows = acc.finalize();

    match format {
        OutputFormat::Csv => {
            let mut wtr = Writer::from_path(output_path)?;
            write_csv_header(&mut wtr)?;
            for row in &output_rows {
                write_csv_row(&mut wtr, row)?;
            }
            wtr.flush()?;
        }
        OutputFormat::Json => {
            write_features_to_json(&output_rows, output_path)?;
        }
    }

    // Write feature schema
    let schema_json = crate::feature_defs::export_json();
    fs::write(schema_path, schema_json).context("Failed to write feature schema")?;

    Ok(FeatureSummary {
        rows: output_rows.len(),
        positives: output_rows
            .iter()
            .filter(|row| row.get("label").is_some_and(|label| label == "1"))
            .count(),
    })
}

// Database fetching functions

fn fetch_all_events(conn: &Connection) -> Result<Vec<Event>> {
    let mut stmt = conn.prepare(
        "SELECT session_id, subsession_id, query, file_path, full_path, timestamp, mtime, file_size, action, episode_queries, is_dir FROM events ORDER BY timestamp, id",
    )?;
    let event_iter = stmt.query_map([], |row| {
        Ok(Event {
            session_id: row.get(0)?,
            subsession_id: row.get(1)?,
            query: row.get(2)?,
            file_path: row.get(3)?,
            full_path: row.get(4)?,
            timestamp: row.get(5)?,
            mtime: row.get(6)?,
            file_size: row.get(7)?,
            action: row.get(8)?,
            episode_queries: row.get(9)?,
            is_dir: row.get::<_, Option<i64>>(10)?.map(|flag| flag != 0),
        })
    })?;

    let mut events = Vec::new();
    for event in event_iter {
        events.push(event?);
    }
    Ok(events)
}

fn fetch_all_sessions(conn: &Connection) -> Result<HashMap<String, Session>> {
    let mut stmt = conn.prepare("SELECT session_id, cwd FROM sessions")?;
    let session_iter = stmt.query_map([], |row| {
        Ok(Session {
            session_id: row.get(0)?,
            cwd: row.get(1)?,
        })
    })?;

    let mut sessions = HashMap::new();
    for session in session_iter {
        let session = session?;
        sessions.insert(session.session_id.clone(), session);
    }
    Ok(sessions)
}

// Feature computation from accumulator state

fn compute_features_from_accumulator(
    impression: &Event,
    acc: &Accumulator,
    sessions: &HashMap<String, Session>,
    matcher: &SkimMatcherV2,
) -> Result<HashMap<String, String>> {
    let mut features = HashMap::new();

    // Metadata columns
    features.insert("label".to_string(), "0".to_string());
    features.insert(
        "subsession_id".to_string(),
        impression.subsession_id.to_string(),
    );
    features.insert("session_id".to_string(), impression.session_id.clone());
    features.insert("timestamp".to_string(), impression.timestamp.to_string());
    features.insert("query".to_string(), impression.query.clone());
    features.insert("file_path".to_string(), impression.file_path.clone());

    // Create FeatureInputs from Event + Accumulator
    let session = sessions.get(&impression.session_id);
    let cwd = session
        .map(|s| Path::new(&s.cwd))
        .unwrap_or_else(|| Path::new("/"));

    // Check if file is under cwd - files under cwd at impression time would have come from walker
    let full_path = Path::new(&impression.full_path);

    // What the row was when the user saw it. Recorded on the event since
    // 2026-09-10; before that there is nothing to read, and the best available
    // answer is today's filesystem - which is wrong for anything since deleted
    // or replaced, and costs a syscall per row. Rows keep arriving with the
    // column set, so this fallback ages out on its own.
    let is_dir = impression.is_dir.unwrap_or_else(|| full_path.is_dir());

    let inputs = FeatureInputs {
        query: &impression.query,
        file_path: &impression.file_path,
        full_path,
        mtime: impression.mtime,
        file_size: impression.file_size,
        cwd,
        clicks_by_file: &acc.clicks_by_file,
        visits_by_dir: &acc.visits_by_dir,
        clicks_by_extension: &acc.clicks_by_extension,
        clicks_indexed: acc.clicks_indexed,
        clicks_by_parent_dir: &acc.clicks_by_parent_dir,
        clicks_for_query: acc.clicks_by_query_and_file.get(&impression.query),
        engagements_for_query: acc
            .engagements_by_episode_query_and_file
            .get(&impression.query),
        current_timestamp: impression.timestamp,
        is_dir,
        // Inference reuses the score the filter already computed; training has
        // no filter, so it does the same match here - against the same string,
        // with one matcher shared across every row rather than one per row.
        fuzzy_score: if impression.query.is_empty() {
            0
        } else {
            matcher
                .fuzzy_match(&impression.file_path, &impression.query)
                .unwrap_or(0)
        },
    };

    // Compute all features using the registry
    for feature in FEATURE_REGISTRY.iter() {
        let value = feature.compute(&inputs);
        features.insert(feature.name().to_string(), value.to_string());
    }

    Ok(features)
}

// JSON writing

fn write_features_to_json(
    feature_rows: &[HashMap<String, String>],
    output_path: &Path,
) -> Result<()> {
    let json_output = serde_json::to_string_pretty(feature_rows)
        .context("Failed to serialize features to JSON")?;
    fs::write(output_path, json_output).context("Failed to write JSON file")?;
    Ok(())
}

// CSV writing

fn write_csv_header(wtr: &mut Writer<std::fs::File>) -> Result<()> {
    wtr.write_record(csv_columns())?;
    Ok(())
}

fn write_csv_row(
    wtr: &mut Writer<std::fs::File>,
    features: &HashMap<String, String>,
) -> Result<()> {
    let row: Vec<String> = csv_columns()
        .iter()
        .map(|&col| features.get(col).cloned().unwrap_or_default())
        .collect();

    wtr.write_record(&row)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_temporal_correctness() {
        // Test that impressions don't see future clicks in their click counts
        let now = 1000i64;

        let events = vec![
            Event {
                session_id: "s1".to_string(),
                subsession_id: 1,
                query: "test".to_string(),
                file_path: "test.rs".to_string(),
                full_path: "/test.rs".to_string(),
                timestamp: now,
                mtime: Some(now - 100),
                file_size: Some(100),
                action: "impression".to_string(),
                episode_queries: None,
                is_dir: None,
            },
            Event {
                session_id: "s1".to_string(),
                subsession_id: 1,
                query: "test".to_string(),
                file_path: "test.rs".to_string(),
                full_path: "/test.rs".to_string(),
                timestamp: now + 200, // Future click
                mtime: None,
                file_size: Some(100),
                action: "click".to_string(),
                episode_queries: None,
                is_dir: None,
            },
            Event {
                session_id: "s1".to_string(),
                subsession_id: 2,
                query: "test".to_string(),
                file_path: "test.rs".to_string(),
                full_path: "/test.rs".to_string(),
                timestamp: now + 400, // Another impression after the click
                mtime: Some(now - 100),
                file_size: Some(100),
                action: "impression".to_string(),
                episode_queries: None,
                is_dir: None,
            },
        ];

        let mut sessions = HashMap::new();
        sessions.insert(
            "s1".to_string(),
            Session {
                session_id: "s1".to_string(),
                cwd: "/".to_string(),
            },
        );

        let mut acc = Accumulator::new();

        for event in &events {
            match event.action.as_str() {
                "impression" => {
                    let features = compute_features_from_accumulator(
                        event,
                        &acc,
                        &sessions,
                        &SkimMatcherV2::default(),
                    )
                    .expect("Failed to compute features");
                    acc.add_impression(event, features);
                }
                "click" => {
                    acc.record_click(event);
                    acc.mark_impressions_as_engaged(event);
                }
                "scroll" => {
                    acc.mark_impressions_as_engaged(event);
                }
                "startup_visit" => {
                    // Ignore for training labels
                }
                _ => {}
            }
        }

        let output_rows = acc.finalize();

        // Find rows by subsession_id (order is not guaranteed from HashMap)
        let first_impression = output_rows
            .iter()
            .find(|row| row.get("subsession_id") == Some(&"1".to_string()))
            .expect("Should have impression from subsession 1");
        let second_impression = output_rows
            .iter()
            .find(|row| row.get("subsession_id") == Some(&"2".to_string()))
            .expect("Should have impression from subsession 2");

        // First impression at T=1000 should NOT see click at T=1200
        assert_eq!(
            first_impression.get("clicks_last_30_days"),
            Some(&"0".to_string()),
            "First impression should not see future click"
        );
        // But it SHOULD get label=1 because click happened in same subsession
        assert_eq!(
            first_impression.get("label"),
            Some(&"1".to_string()),
            "First impression should have label=1 from future click in same subsession"
        );
        // Episode ID should be 0 (first episode)
        assert_eq!(
            first_impression.get("episode_id"),
            Some(&"0".to_string()),
            "First impression should be in episode 0"
        );

        // Second impression at T=1400 SHOULD see click at T=1200
        assert_eq!(
            second_impression.get("clicks_last_30_days"),
            Some(&"1".to_string()),
            "Second impression should see past click"
        );
        // But label should be 0 (different subsession)
        assert_eq!(
            second_impression.get("label"),
            Some(&"0".to_string()),
            "Second impression in different subsession should have label=0"
        );
        // Episode ID should be 1 (second episode, after the click)
        assert_eq!(
            second_impression.get("episode_id"),
            Some(&"1".to_string()),
            "Second impression should be in episode 1 (after click)"
        );
    }

    #[test]
    fn test_generated_features_match_the_schema_the_trainer_reads() {
        // This used to read a `test/events.db` that is not in the repository,
        // return early when it was missing - which was always - and assert a
        // header six columns long that the code stopped producing years ago. It
        // builds its own database now, so it runs.
        use crate::db::{Database, EventData, FileMetadata, UserInteraction};

        let dir = std::env::temp_dir().join(format!("psychic-feat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("events.db");

        {
            let db = Database::new(&db_path).unwrap();
            let shown: Vec<FileMetadata> = ["alpha.rs", "beta.rs"]
                .iter()
                .map(|name| FileMetadata {
                    relative_path: name.to_string(),
                    full_path: format!("/test/{}", name),
                    mtime: Some(1_700_000_000),
                    atime: None,
                    size: Some(100),
                    is_dir: false,
                })
                .collect();
            db.log_impressions("al", &shown, 1, "session-1").unwrap();
            db.log_event(EventData {
                query: "al",
                file_path: "alpha.rs",
                full_path: "/test/alpha.rs",
                mtime: Some(1_700_000_000),
                atime: None,
                file_size: Some(100),
                subsession_id: 1,
                action: UserInteraction::Click,
                session_id: "session-1",
                episode_queries: None,
                rank: None,
                is_dir: Some(false),
            })
            .unwrap();
        }

        let csv_path = dir.join("features.csv");
        let schema_path = dir.join("feature_schema.json");
        let summary =
            generate_features(&db_path, &csv_path, &schema_path, OutputFormat::Csv).unwrap();

        let csv = std::fs::read_to_string(&csv_path).unwrap();
        let mut lines = csv.lines();

        assert_eq!(
            lines.next().unwrap(),
            csv_columns().join(","),
            "the header is the column list the trainer reads, not a copy of it"
        );
        assert_eq!(summary.rows, 2, "one row per impression");
        assert_eq!(summary.positives, 1, "the one that was clicked");

        let labels: Vec<&str> = lines.map(|line| line.split(',').next().unwrap()).collect();
        assert_eq!(labels, ["1", "0"], "the clicked row is the positive one");

        std::fs::remove_dir_all(&dir).ok();
    }
}
