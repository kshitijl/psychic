use anyhow::{Context, Result};
use csv::Writer;
use jiff::{Timestamp, Span};
use rusqlite::Connection;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

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

#[derive(Debug, Clone)]
struct Session {
    session_id: String,
    timezone: String,
}

// Output format enum
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Csv,
    Json,
}

// Helper structs for accumulator
#[derive(Debug, Clone)]
struct ClickEvent {
    timestamp: i64,
}

// Accumulator for fold-based processing
struct Accumulator {
    clicks_by_file: HashMap<String, Vec<ClickEvent>>,
    // Key: (session_id, subsession_id, full_path)
    pending_impressions: HashMap<(String, u64, String), PendingImpression>,
    output_rows: Vec<HashMap<String, String>>,
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

    fn add_impression(&mut self, event: &Event, features: HashMap<String, String>) {
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
        // Find impressions in same subsession with same file that happened BEFORE this click/scroll
        let key = (
            event.session_id.clone(),
            event.subsession_id,
            event.full_path.clone(),
        );

        if let Some(pending) = self.pending_impressions.get_mut(&key) {
            if pending.timestamp <= event.timestamp {
                pending.features.insert("label".to_string(), "1".to_string());
            }
        }
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
    let mut stmt = conn.prepare("SELECT session_id, timezone FROM sessions")?;
    let session_iter = stmt.query_map([], |row| {
        Ok(Session {
            session_id: row.get(0)?,
            timezone: row.get(1)?,
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

    // Label starts as 0, will be updated to 1 if click/scroll happens later
    features.insert("label".to_string(), "0".to_string());

    // Subsession ID (for grouping in LambdaRank)
    features.insert("subsession_id".to_string(), impression.subsession_id.to_string());

    // Session ID (for potential session-level features)
    features.insert("session_id".to_string(), impression.session_id.clone());

    // Query
    features.insert("query".to_string(), impression.query.clone());

    // File path
    features.insert("file_path".to_string(), impression.file_path.clone());

    // filename_starts_with_query
    let filename = Path::new(&impression.file_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    features.insert(
        "filename_starts_with_query".to_string(),
        if !impression.query.is_empty()
            && filename
                .to_lowercase()
                .starts_with(&impression.query.to_lowercase())
        {
            "1"
        } else {
            "0"
        }
        .to_string(),
    );

    // clicks_last_30_days - only count clicks BEFORE this impression
    if let Some(session) = sessions.get(&impression.session_id) {
        let now_ts = Timestamp::from_second(impression.timestamp)?;
        let session_tz =
            jiff::tz::TimeZone::get(&session.timezone).unwrap_or(jiff::tz::TimeZone::system());
        let now_zoned = now_ts.to_zoned(session_tz);
        let thirty_days_ago = now_zoned.checked_sub(Span::new().days(30))?.timestamp();

        let clicks_last_30_days = acc
            .clicks_by_file
            .get(&impression.full_path)
            .map(|clicks| {
                clicks
                    .iter()
                    .filter(|c| {
                        c.timestamp >= thirty_days_ago.as_second()
                            && c.timestamp <= impression.timestamp // Only past clicks!
                    })
                    .count()
            })
            .unwrap_or(0);

        features.insert(
            "clicks_last_30_days".to_string(),
            clicks_last_30_days.to_string(),
        );
    } else {
        features.insert("clicks_last_30_days".to_string(), "0".to_string());
    }

    // modified_today
    if let Some(mtime) = impression.mtime {
        let seconds_since_mod = impression.timestamp - mtime;
        let hours = seconds_since_mod / 3600;
        features.insert(
            "modified_today".to_string(),
            if hours < 24 { "1" } else { "0" }.to_string(),
        );
    } else {
        features.insert("modified_today".to_string(), "0".to_string());
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
    wtr.write_record([
        "label",
        "subsession_id",
        "session_id",
        "query",
        "file_path",
        "filename_starts_with_query",
        "clicks_last_30_days",
        "modified_today",
    ])?;
    Ok(())
}

fn write_csv_row(
    wtr: &mut Writer<std::fs::File>,
    features: &HashMap<String, String>,
) -> Result<()> {
    let columns = [
        "label",
        "subsession_id",
        "session_id",
        "query",
        "file_path",
        "filename_starts_with_query",
        "clicks_last_30_days",
        "modified_today",
    ];

    let row: Vec<String> = columns
        .iter()
        .map(|&col| features.get(col).cloned().unwrap_or_else(|| "".to_string()))
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

        // First impression at T=1000 should NOT see click at T=1200
        let first_row = &output_rows[0];
        assert_eq!(
            first_row.get("clicks_last_30_days"),
            Some(&"0".to_string()),
            "First impression should not see future click"
        );
        // But it SHOULD get label=1 because click happened in same subsession
        assert_eq!(
            first_row.get("label"),
            Some(&"1".to_string()),
            "First impression should have label=1 from future click in same subsession"
        );

        // Second impression at T=1400 SHOULD see click at T=1200
        let second_row = &output_rows[1];
        assert_eq!(
            second_row.get("clicks_last_30_days"),
            Some(&"1".to_string()),
            "Second impression should see past click"
        );
        // But label should be 0 (different subsession)
        assert_eq!(
            second_row.get("label"),
            Some(&"0".to_string()),
            "Second impression in different subsession should have label=0"
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
