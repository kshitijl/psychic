use anyhow::{Context, Result};
use csv::Writer;
use rusqlite::Connection;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// Import the feature definitions module
use crate::feature_defs::{FeatureInputs, FEATURE_REGISTRY, csv_columns};

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
    action: String,
}

// Output format enum
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Csv,
    Json,
}

// Accumulator for fold-based processing
struct Accumulator {
    clicks_by_file: HashMap<String, Vec<ClickEvent>>,
    // Key: (session_id, subsession_id, full_path)
    pending_impressions: HashMap<(String, u64, String), PendingImpression>,
    output_rows: Vec<HashMap<String, String>>,
    // Track current group ID - increments with each click/scroll and session change
    current_group_id: u64,
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
            clicks_by_file: HashMap::new(),
            pending_impressions: HashMap::new(),
            output_rows: Vec::new(),
            current_group_id: 0,
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
    }

    fn add_impression(&mut self, event: &Event, mut features: HashMap<String, String>) {
        // Check if this is a new session - if so, increment group_id
        if let Some(ref last_session) = self.last_session_id
            && last_session != &event.session_id {
            self.current_group_id += 1;
        }
        self.last_session_id = Some(event.session_id.clone());

        // Add group_id to features
        features.insert("group_id".to_string(), self.current_group_id.to_string());

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
            && last_session != &event.session_id {
            self.current_group_id += 1;
        }
        self.last_session_id = Some(event.session_id.clone());

        // Find impressions in same subsession with same file that happened BEFORE this click/scroll
        let key = (
            event.session_id.clone(),
            event.subsession_id,
            event.full_path.clone(),
        );

        if let Some(pending) = self.pending_impressions.get_mut(&key)
            && pending.timestamp <= event.timestamp {
            pending.features.insert("label".to_string(), "1".to_string());
        }

        // Increment group_id after each engagement event (click or scroll)
        // This creates a new group for the next sequence of impressions
        self.current_group_id += 1;
    }

    fn finalize(mut self) -> Vec<HashMap<String, String>> {
        // Move all pending impressions to output
        for (_, impression) in self.pending_impressions {
            self.output_rows.push(impression.features);
        }
        self.output_rows
    }
}

// Main function to generate features

pub fn generate_features(db_path: &Path, output_path: &Path, format: OutputFormat) -> Result<()> {
    let conn = Connection::open(db_path).context("Failed to open database")?;
    let mut all_events = fetch_all_events(&conn)?;
    let all_sessions = fetch_all_sessions(&conn)?;

    // Sort events by timestamp (critical for temporal correctness)
    all_events.sort_by_key(|e| e.timestamp);

    let mut acc = Accumulator::new();

    for event in &all_events {
        match event.action.as_str() {
            "impression" => {
                let features = compute_features_from_accumulator(event, &acc, &all_sessions)?;
                acc.add_impression(event, features);
            }
            "click" => {
                acc.record_click(event);
                acc.mark_impressions_as_engaged(event);
            }
            "scroll" => {
                acc.mark_impressions_as_engaged(event);
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

    Ok(())
}

// Database fetching functions

fn fetch_all_events(conn: &Connection) -> Result<Vec<Event>> {
    let mut stmt = conn.prepare(
        "SELECT session_id, subsession_id, query, file_path, full_path, timestamp, mtime, action FROM events ORDER BY timestamp, id",
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
            action: row.get(7)?,
        })
    })?;

    let mut events = Vec::new();
    for event in event_iter {
        events.push(event?);
    }
    Ok(events)
}

fn fetch_all_sessions(conn: &Connection) -> Result<HashMap<String, Session>> {
    let mut stmt = conn.prepare("SELECT session_id, timezone, cwd FROM sessions")?;
    let session_iter = stmt.query_map([], |row| {
        Ok(Session {
            session_id: row.get(0)?,
            timezone: row.get(1)?,
            cwd: row.get(2)?,
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
) -> Result<HashMap<String, String>> {
    let mut features = HashMap::new();

    // Metadata columns
    features.insert("label".to_string(), "0".to_string());
    features.insert("subsession_id".to_string(), impression.subsession_id.to_string());
    features.insert("session_id".to_string(), impression.session_id.clone());
    features.insert("query".to_string(), impression.query.clone());
    features.insert("file_path".to_string(), impression.file_path.clone());

    // Create FeatureInputs from Event + Accumulator
    let session = sessions.get(&impression.session_id);
    let cwd = session
        .map(|s| Path::new(&s.cwd))
        .unwrap_or_else(|| Path::new("/"));

    let inputs = FeatureInputs {
        query: &impression.query,
        file_path: &impression.file_path,
        full_path: Path::new(&impression.full_path),
        mtime: impression.mtime,
        cwd,
        clicks_by_file: &acc.clicks_by_file,
        current_timestamp: impression.timestamp,
        session,
    };

    // Compute all features using the registry
    for feature in FEATURE_REGISTRY.iter() {
        let value = feature.compute(&inputs)?;
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
                action: "impression".to_string(),
            },
            Event {
                session_id: "s1".to_string(),
                subsession_id: 1,
                query: "test".to_string(),
                file_path: "test.rs".to_string(),
                full_path: "/test.rs".to_string(),
                timestamp: now + 200, // Future click
                mtime: None,
                action: "click".to_string(),
            },
            Event {
                session_id: "s1".to_string(),
                subsession_id: 2,
                query: "test".to_string(),
                file_path: "test.rs".to_string(),
                full_path: "/test.rs".to_string(),
                timestamp: now + 400, // Another impression after the click
                mtime: Some(now - 100),
                action: "impression".to_string(),
            },
        ];

        let mut sessions = HashMap::new();
        sessions.insert(
            "s1".to_string(),
            Session {
                session_id: "s1".to_string(),
                timezone: "UTC".to_string(),
                cwd: "/".to_string(),
            },
        );

        let mut acc = Accumulator::new();

        for event in &events {
            match event.action.as_str() {
                "impression" => {
                    let features = compute_features_from_accumulator(event, &acc, &sessions)
                        .expect("Failed to compute features");
                    acc.add_impression(event, features);
                }
                "click" => {
                    acc.record_click(event);
                    acc.mark_impressions_as_engaged(event);
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
        // Group ID should be 0 (first group)
        assert_eq!(
            first_impression.get("group_id"),
            Some(&"0".to_string()),
            "First impression should be in group 0"
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
        // Group ID should be 1 (second group, after the click)
        assert_eq!(
            second_impression.get("group_id"),
            Some(&"1".to_string()),
            "Second impression should be in group 1 (after click)"
        );
    }

    #[test]
    fn test_basic_feature_generation() {
        use std::path::PathBuf;

        let db_path = PathBuf::from("test/events.db");
        if !db_path.exists() {
            eprintln!("Skipping test - test/events.db not found");
            return;
        }

        let output_path = PathBuf::from("test/features_test.csv");
        generate_features(&db_path, &output_path, OutputFormat::Csv)
            .expect("Failed to generate features");

        // Verify CSV was created
        let csv_content = std::fs::read_to_string(&output_path).expect("Failed to read CSV");
        let lines: Vec<&str> = csv_content.lines().collect();

        assert!(!lines.is_empty(), "CSV should not be empty");
        assert_eq!(lines[0], "label,query,file_path,filename_starts_with_query,clicks_last_30_days,modified_today");
    }
}
