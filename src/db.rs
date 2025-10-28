use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::path::{Path, PathBuf};

pub struct FileMetadata {
    pub relative_path: String,
    pub full_path: String,
    pub mtime: Option<i64>,
    pub atime: Option<i64>,
    pub size: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct ContextData {
    pub cwd: String,
    pub gateway: String,
    pub subnet: String,
    pub dns: String,
    pub shell_history: String,
    pub running_processes: String,
    pub timezone: String,
}

#[derive(Debug, Clone, Copy)]
pub enum UserInteraction {
    Click,
    Scroll,
    Impression,
    StartupVisit,
}

pub struct EventData<'a> {
    pub query: &'a str,
    pub file_path: &'a str,
    pub full_path: &'a str,
    pub mtime: Option<i64>,
    pub atime: Option<i64>,
    pub file_size: Option<i64>,
    pub subsession_id: u64,
    pub action: UserInteraction,
    pub session_id: &'a str,
    pub episode_queries: Option<&'a str>, // JSON array of queries in this episode
}

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path).context("Failed to open database")?;

        // Configure SQLite for better concurrency
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;",
        )?;

        // Create tables if they don't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                query TEXT NOT NULL,
                file_path TEXT NOT NULL,
                full_path TEXT NOT NULL,
                mtime INTEGER,
                atime INTEGER,
                file_size INTEGER,
                subsession_id INTEGER,
                action TEXT NOT NULL,
                session_id TEXT NOT NULL,
                episode_queries TEXT
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                gateway TEXT NOT NULL,
                subnet TEXT NOT NULL,
                dns TEXT NOT NULL,
                shell_history TEXT NOT NULL,
                running_processes TEXT NOT NULL,
                timezone TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )",
            [],
        )?;

        // This index is for click count queries. We might want to do those in Rust
        // code at some point, but this is convenient for now.
        // The clause: WHERE action = 'click' AND timestamp >= ? GROUP BY full_path
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_click_lookup
             ON events(action, timestamp, full_path)",
            [],
        )?;

        Ok(Database { conn })
    }

    pub fn get_db_path(data_dir: &Path) -> PathBuf {
        data_dir.join("events.db")
    }

    pub fn log_session(&self, session_id: &str, context: &ContextData) -> Result<()> {
        let timestamp = jiff::Timestamp::now().as_second();

        self.conn.execute(
            "INSERT INTO sessions (session_id, cwd, gateway, subnet, dns, shell_history, running_processes, timezone, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                session_id,
                &context.cwd,
                &context.gateway,
                &context.subnet,
                &context.dns,
                &context.shell_history,
                &context.running_processes,
                &context.timezone,
                timestamp
            ],
        )?;

        Ok(())
    }

    pub fn log_event(&self, event: EventData) -> Result<()> {
        let timestamp = jiff::Timestamp::now().as_second();

        let action = match event.action {
            UserInteraction::Click => "click",
            UserInteraction::Scroll => "scroll",
            UserInteraction::Impression => "impression",
            UserInteraction::StartupVisit => "startup_visit",
        };

        self.conn.execute(
            "INSERT INTO events (timestamp, query, file_path, full_path, mtime, atime, file_size, subsession_id, action, session_id, episode_queries)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                timestamp,
                event.query,
                event.file_path,
                event.full_path,
                event.mtime,
                event.atime,
                event.file_size,
                event.subsession_id,
                action,
                event.session_id,
                event.episode_queries
            ],
        )?;

        Ok(())
    }

    pub fn log_impressions(
        &self,
        query: &str,
        file_paths: &[FileMetadata],
        subsession_id: u64,
        session_id: &str,
    ) -> Result<()> {
        for FileMetadata {
            relative_path,
            full_path,
            mtime,
            atime,
            size,
        } in file_paths
        {
            self.log_event(EventData {
                query,
                file_path: relative_path,
                full_path,
                mtime: *mtime,
                atime: *atime,
                file_size: *size,
                subsession_id,
                action: UserInteraction::Impression,
                session_id,
                episode_queries: None,
            })?;
        }

        Ok(())
    }

    pub fn get_previously_interacted_files(&self) -> Result<Vec<String>> {
        // Get all unique full_paths that have been clicked, scrolled, or auto-visited at startup
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT full_path
             FROM events
             WHERE action IN ('click', 'scroll', 'startup_visit')
             ORDER BY timestamp DESC",
        )?;

        let paths = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<String>, _>>()?;

        Ok(paths)
    }

    pub fn summarize_events(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT action, COUNT(*) as count
             FROM events
             GROUP BY action
             ORDER BY count DESC",
        )?;

        let summary = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?
            .collect::<Result<Vec<(String, i64)>, _>>()?;

        Ok(summary)
    }
}
