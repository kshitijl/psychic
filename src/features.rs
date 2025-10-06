use anyhow::{Context, Result};
use csv::Writer;
use regex::Regex;
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
struct ImpressionEvent {
    session_id: String,
    subsession_id: u64,
    query: String,
    file_path: String,
    full_path: String,
    timestamp: i64,
    mtime: Option<i64>,
    atime: Option<i64>,
    file_size: Option<i64>,
    rank: usize,
}

#[derive(Debug)]
struct SessionContext {
    cwd: String,
    gateway: String,
    subnet: String,
    dns: String,
    shell_history: String,
    running_processes: String,
    timezone: String,
    created_at: i64,
}

pub fn generate_features(db_path: &Path, output_path: &Path) -> Result<()> {
    let conn = Connection::open(db_path).context("Failed to open database")?;

    // Get all impressions with rank
    let impressions = fetch_impressions(&conn)?;

    // Get all clicks/scrolls (label=1)
    let labels = fetch_labels(&conn)?;

    // Get session contexts
    let sessions = fetch_sessions(&conn)?;

    // Build feature rows
    let mut wtr = Writer::from_path(output_path)?;

    // Write CSV header
    write_csv_header(&mut wtr)?;

    // Generate features for each impression
    for impression in impressions {
        let features = compute_features(&impression, &labels, &sessions, &conn)?;
        write_csv_row(&mut wtr, &features)?;
    }

    wtr.flush()?;
    Ok(())
}

fn fetch_impressions(conn: &Connection) -> Result<Vec<ImpressionEvent>> {
    let mut stmt = conn.prepare(
        "SELECT session_id, subsession_id, query, file_path, full_path, timestamp, mtime, atime, file_size
         FROM events
         WHERE action = 'impression'
         ORDER BY session_id, subsession_id, timestamp",
    )?;

    let mut impressions = Vec::new();
    let mut current_subsession = (String::new(), 0u64);
    let mut rank = 1;

    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, u64>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, i64>(5)?,
            row.get::<_, Option<i64>>(6)?,
            row.get::<_, Option<i64>>(7)?,
            row.get::<_, Option<i64>>(8)?,
        ))
    })?;

    for row in rows {
        let (session_id, subsession_id, query, file_path, full_path, timestamp, mtime, atime, file_size) = row?;

        // Reset rank for new subsession
        if current_subsession != (session_id.clone(), subsession_id) {
            current_subsession = (session_id.clone(), subsession_id);
            rank = 1;
        }

        impressions.push(ImpressionEvent {
            session_id,
            subsession_id,
            query,
            file_path,
            full_path,
            timestamp,
            mtime,
            atime,
            file_size,
            rank,
        });

        rank += 1;
    }

    Ok(impressions)
}

fn fetch_labels(conn: &Connection) -> Result<HashMap<(String, u64, String), bool>> {
    let mut stmt = conn.prepare(
        "SELECT session_id, subsession_id, full_path
         FROM events
         WHERE action IN ('click', 'scroll')",
    )?;

    let mut labels = HashMap::new();

    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, u64>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;

    for row in rows {
        let (session_id, subsession_id, full_path) = row?;
        labels.insert((session_id, subsession_id, full_path), true);
    }

    Ok(labels)
}

fn fetch_sessions(conn: &Connection) -> Result<HashMap<String, SessionContext>> {
    let mut stmt = conn.prepare(
        "SELECT session_id, cwd, gateway, subnet, dns, shell_history, running_processes, timezone, created_at
         FROM sessions",
    )?;

    let mut sessions = HashMap::new();

    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            SessionContext {
                cwd: row.get(1)?,
                gateway: row.get(2)?,
                subnet: row.get(3)?,
                dns: row.get(4)?,
                shell_history: row.get(5)?,
                running_processes: row.get(6)?,
                timezone: row.get(7)?,
                created_at: row.get(8)?,
            },
        ))
    })?;

    for row in rows {
        let (session_id, context) = row?;
        sessions.insert(session_id, context);
    }

    Ok(sessions)
}

fn compute_features(
    impression: &ImpressionEvent,
    labels: &HashMap<(String, u64, String), bool>,
    sessions: &HashMap<String, SessionContext>,
    conn: &Connection,
) -> Result<HashMap<String, String>> {
    let mut features = HashMap::new();

    // Label
    let label = if labels.contains_key(&(
        impression.session_id.clone(),
        impression.subsession_id,
        impression.full_path.clone(),
    )) {
        "1"
    } else {
        "0"
    };
    features.insert("label".to_string(), label.to_string());

    // Session and query info
    features.insert("session_id".to_string(), impression.session_id.clone());
    features.insert("subsession_id".to_string(), impression.subsession_id.to_string());
    features.insert("query".to_string(), impression.query.clone());
    features.insert("file_path".to_string(), impression.file_path.clone());
    features.insert("full_path".to_string(), impression.full_path.clone());

    // Ranking features
    features.insert("rank".to_string(), impression.rank.to_string());

    // Static file features
    compute_static_features(&mut features, impression);

    // Temporal features
    compute_temporal_features(&mut features, impression, sessions);

    // Query-based features
    compute_query_features(&mut features, impression);

    // Historical features
    compute_historical_features(&mut features, impression, conn)?;

    // Session context features
    if let Some(session_ctx) = sessions.get(&impression.session_id) {
        features.insert("timezone".to_string(), session_ctx.timezone.clone());
        features.insert("subnet".to_string(), session_ctx.subnet.clone());
    } else {
        features.insert("timezone".to_string(), "unknown".to_string());
        features.insert("subnet".to_string(), "unknown".to_string());
    }

    Ok(features)
}

fn compute_static_features(features: &mut HashMap<String, String>, impression: &ImpressionEvent) {
    let path = Path::new(&impression.file_path);

    // Extension
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("none")
        .to_string();
    features.insert("extension".to_string(), extension);

    // Filename
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    features.insert("filename".to_string(), filename.clone());

    // Path depth
    let depth = impression.file_path.matches('/').count();
    features.insert("path_depth".to_string(), depth.to_string());

    // Filename length
    features.insert("filename_len".to_string(), filename.len().to_string());

    // Path length
    features.insert("path_len".to_string(), impression.file_path.len().to_string());

    // Is hidden
    let is_hidden = if filename.starts_with('.') { "1" } else { "0" };
    features.insert("is_hidden".to_string(), is_hidden.to_string());

    // File size
    let file_size = impression.file_size.unwrap_or(0);
    features.insert("file_size".to_string(), file_size.to_string());

    // Special directory patterns
    let path_str = &impression.file_path;
    features.insert("in_src".to_string(), if path_str.contains("/src/") { "1" } else { "0" }.to_string());
    features.insert("in_test".to_string(), if path_str.contains("/test") || path_str.contains("_test") { "1" } else { "0" }.to_string());
    features.insert("in_lib".to_string(), if path_str.contains("/lib/") { "1" } else { "0" }.to_string());
    features.insert("in_bin".to_string(), if path_str.contains("/bin/") { "1" } else { "0" }.to_string());
    features.insert("in_config".to_string(), if path_str.contains("config") || path_str.ends_with(".toml") || path_str.ends_with(".yaml") { "1" } else { "0" }.to_string());

    // Has version in path (e.g., v1, v2)
    let version_regex = Regex::new(r"v\d+").unwrap();
    features.insert("has_version".to_string(), if version_regex.is_match(path_str) { "1" } else { "0" }.to_string());

    // Has UUID pattern
    let uuid_regex = Regex::new(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}").unwrap();
    features.insert("has_uuid".to_string(), if uuid_regex.is_match(path_str) { "1" } else { "0" }.to_string());

    // Has hash pattern (8+ hex chars)
    let hash_regex = Regex::new(r"[0-9a-f]{8,}").unwrap();
    features.insert("has_hash".to_string(), if hash_regex.is_match(path_str) { "1" } else { "0" }.to_string());

    // Number of dots in filename
    let dot_count = filename.matches('.').count();
    features.insert("filename_dots".to_string(), dot_count.to_string());

    // Shannon entropy of filename (complexity measure)
    let entropy = calculate_shannon_entropy(&filename);
    features.insert("filename_entropy".to_string(), format!("{:.4}", entropy));

    // Is readme
    let is_readme = if filename.to_lowercase().contains("readme") { "1" } else { "0" };
    features.insert("is_readme".to_string(), is_readme.to_string());

    // Is main file
    let is_main = if filename.contains("main") || filename.contains("index") { "1" } else { "0" };
    features.insert("is_main".to_string(), is_main.to_string());
}

fn compute_temporal_features(
    features: &mut HashMap<String, String>,
    impression: &ImpressionEvent,
    sessions: &HashMap<String, SessionContext>,
) {
    let now = impression.timestamp;

    // Time since modification
    if let Some(mtime) = impression.mtime {
        let seconds_since_mod = now - mtime;
        features.insert("seconds_since_mod".to_string(), seconds_since_mod.to_string());

        // Buckets
        let hours = seconds_since_mod / 3600;
        features.insert("modified_today".to_string(), if hours < 24 { "1" } else { "0" }.to_string());
        features.insert("modified_this_week".to_string(), if hours < 168 { "1" } else { "0" }.to_string());
        features.insert("modified_this_month".to_string(), if hours < 720 { "1" } else { "0" }.to_string());
    } else {
        features.insert("seconds_since_mod".to_string(), "-1".to_string());
        features.insert("modified_today".to_string(), "0".to_string());
        features.insert("modified_this_week".to_string(), "0".to_string());
        features.insert("modified_this_month".to_string(), "0".to_string());
    }

    // Time since access
    if let Some(atime) = impression.atime {
        let seconds_since_access = now - atime;
        features.insert("seconds_since_access".to_string(), seconds_since_access.to_string());

        let hours = seconds_since_access / 3600;
        features.insert("accessed_today".to_string(), if hours < 24 { "1" } else { "0" }.to_string());
    } else {
        features.insert("seconds_since_access".to_string(), "-1".to_string());
        features.insert("accessed_today".to_string(), "0".to_string());
    }

    // Session time
    if let Some(session_ctx) = sessions.get(&impression.session_id) {
        let session_duration = now - session_ctx.created_at;
        features.insert("session_duration_seconds".to_string(), session_duration.to_string());
    } else {
        features.insert("session_duration_seconds".to_string(), "0".to_string());
    }
}

fn compute_query_features(features: &mut HashMap<String, String>, impression: &ImpressionEvent) {
    let query = &impression.query;
    let filename = Path::new(&impression.file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let path = &impression.file_path;

    // Query length
    features.insert("query_len".to_string(), query.len().to_string());

    // Is empty query
    features.insert("query_empty".to_string(), if query.is_empty() { "1" } else { "0" }.to_string());

    // Exact match
    features.insert("query_exact_match".to_string(), if filename == query { "1" } else { "0" }.to_string());

    // Starts with query
    features.insert("filename_starts_with_query".to_string(),
        if !query.is_empty() && filename.to_lowercase().starts_with(&query.to_lowercase()) { "1" } else { "0" }.to_string());

    // Contains query
    features.insert("filename_contains_query".to_string(),
        if !query.is_empty() && filename.to_lowercase().contains(&query.to_lowercase()) { "1" } else { "0" }.to_string());

    // Path contains query
    features.insert("path_contains_query".to_string(),
        if !query.is_empty() && path.to_lowercase().contains(&query.to_lowercase()) { "1" } else { "0" }.to_string());

    // Query matches extension
    features.insert("query_matches_extension".to_string(),
        if query == Path::new(&impression.file_path).extension().and_then(|e| e.to_str()).unwrap_or("") { "1" } else { "0" }.to_string());

    // Number of query tokens
    let query_tokens: Vec<&str> = query.split_whitespace().collect();
    features.insert("query_token_count".to_string(), query_tokens.len().to_string());

    // Filename word match count
    let filename_lower = filename.to_lowercase();
    let matching_tokens = query_tokens.iter().filter(|&&token| filename_lower.contains(&token.to_lowercase())).count();
    features.insert("filename_matching_tokens".to_string(), matching_tokens.to_string());
}

fn compute_historical_features(
    features: &mut HashMap<String, String>,
    impression: &ImpressionEvent,
    conn: &Connection,
) -> Result<()> {
    // Count how many times this file was clicked in previous sessions
    let mut stmt = conn.prepare(
        "SELECT COUNT(*) FROM events WHERE full_path = ?1 AND action = 'click' AND session_id != ?2",
    )?;
    let prev_clicks: i64 = stmt.query_row([&impression.full_path, &impression.session_id], |row| row.get(0))?;
    features.insert("prev_session_clicks".to_string(), prev_clicks.to_string());

    // Count scrolls
    let mut stmt = conn.prepare(
        "SELECT COUNT(*) FROM events WHERE full_path = ?1 AND action = 'scroll' AND session_id != ?2",
    )?;
    let prev_scrolls: i64 = stmt.query_row([&impression.full_path, &impression.session_id], |row| row.get(0))?;
    features.insert("prev_session_scrolls".to_string(), prev_scrolls.to_string());

    // Total engagement (clicks + scrolls)
    features.insert("prev_session_engagement".to_string(), (prev_clicks + prev_scrolls).to_string());

    // Count clicks for same query
    let mut stmt = conn.prepare(
        "SELECT COUNT(*) FROM events WHERE full_path = ?1 AND query = ?2 AND action = 'click' AND session_id != ?3",
    )?;
    let query_clicks: i64 = stmt.query_row([&impression.full_path, &impression.query, &impression.session_id], |row| row.get(0))?;
    features.insert("prev_query_clicks".to_string(), query_clicks.to_string());

    // Has been clicked ever (global)
    features.insert("ever_clicked".to_string(), if prev_clicks > 0 { "1" } else { "0" }.to_string());

    // Clicks in current session (before this impression)
    let mut stmt = conn.prepare(
        "SELECT COUNT(*) FROM events WHERE full_path = ?1 AND action = 'click' AND session_id = ?2 AND timestamp < ?3",
    )?;
    let current_session_clicks: i64 = stmt.query_row([&impression.full_path, &impression.session_id, &impression.timestamp.to_string()], |row| row.get(0))?;
    features.insert("current_session_clicks".to_string(), current_session_clicks.to_string());

    Ok(())
}

fn calculate_shannon_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }

    let mut freq: HashMap<char, usize> = HashMap::new();
    for c in s.chars() {
        *freq.entry(c).or_insert(0) += 1;
    }

    let len = s.len() as f64;
    freq.values()
        .map(|&count| {
            let p = count as f64 / len;
            -p * p.log2()
        })
        .sum()
}

fn write_csv_header(wtr: &mut Writer<std::fs::File>) -> Result<()> {
    wtr.write_record(&[
        "label",
        "session_id",
        "subsession_id",
        "query",
        "file_path",
        "full_path",
        "rank",
        "extension",
        "filename",
        "path_depth",
        "filename_len",
        "path_len",
        "is_hidden",
        "file_size",
        "in_src",
        "in_test",
        "in_lib",
        "in_bin",
        "in_config",
        "has_version",
        "has_uuid",
        "has_hash",
        "filename_dots",
        "filename_entropy",
        "is_readme",
        "is_main",
        "seconds_since_mod",
        "modified_today",
        "modified_this_week",
        "modified_this_month",
        "seconds_since_access",
        "accessed_today",
        "session_duration_seconds",
        "query_len",
        "query_empty",
        "query_exact_match",
        "filename_starts_with_query",
        "filename_contains_query",
        "path_contains_query",
        "query_matches_extension",
        "query_token_count",
        "filename_matching_tokens",
        "prev_session_clicks",
        "prev_session_scrolls",
        "prev_session_engagement",
        "prev_query_clicks",
        "ever_clicked",
        "current_session_clicks",
        "timezone",
        "subnet",
    ])?;
    Ok(())
}

fn write_csv_row(wtr: &mut Writer<std::fs::File>, features: &HashMap<String, String>) -> Result<()> {
    let columns = [
        "label",
        "session_id",
        "subsession_id",
        "query",
        "file_path",
        "full_path",
        "rank",
        "extension",
        "filename",
        "path_depth",
        "filename_len",
        "path_len",
        "is_hidden",
        "file_size",
        "in_src",
        "in_test",
        "in_lib",
        "in_bin",
        "in_config",
        "has_version",
        "has_uuid",
        "has_hash",
        "filename_dots",
        "filename_entropy",
        "is_readme",
        "is_main",
        "seconds_since_mod",
        "modified_today",
        "modified_this_week",
        "modified_this_month",
        "seconds_since_access",
        "accessed_today",
        "session_duration_seconds",
        "query_len",
        "query_empty",
        "query_exact_match",
        "filename_starts_with_query",
        "filename_contains_query",
        "path_contains_query",
        "query_matches_extension",
        "query_token_count",
        "filename_matching_tokens",
        "prev_session_clicks",
        "prev_session_scrolls",
        "prev_session_engagement",
        "prev_query_clicks",
        "ever_clicked",
        "current_session_clicks",
        "timezone",
        "subnet",
    ];

    let row: Vec<String> = columns
        .iter()
        .map(|&col| features.get(col).cloned().unwrap_or_else(|| "".to_string()))
        .collect();

    wtr.write_record(&row)?;
    Ok(())
}
