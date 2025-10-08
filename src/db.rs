use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::path::{Path, PathBuf};

// Type alias for file metadata tuples
pub type FileMetadata = (String, String, Option<i64>, Option<i64>, Option<i64>); // (relative, full, mtime, atime, file_size)

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

pub struct EventData<'a> {
    pub query: &'a str,
    pub file_path: &'a str,
    pub full_path: &'a str,
    pub mtime: Option<i64>,
    pub atime: Option<i64>,
    pub file_size: Option<i64>,
    pub subsession_id: u64,
    pub action: &'a str,
    pub session_id: &'a str,
}

pub struct Database {
    conn: Connection,
    db_path: PathBuf,
}

impl Database {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let db_path = Self::get_db_path(data_dir)?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&db_path).context("Failed to open database")?;

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
                session_id TEXT NOT NULL
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

        // Create index for efficient click count queries
        // Covers: WHERE action = 'click' AND timestamp >= ? GROUP BY full_path
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_click_lookup
             ON events(action, timestamp, full_path)",
            [],
        )?;

        Ok(Database { conn, db_path })
    }

    pub fn get_db_path(data_dir: &Path) -> Result<PathBuf> {
        Ok(data_dir.join("events.db"))
    }

    pub fn db_path(&self) -> PathBuf {
        self.db_path.clone()
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

        self.conn.execute(
            "INSERT INTO events (timestamp, query, file_path, full_path, mtime, atime, file_size, subsession_id, action, session_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                timestamp,
                event.query,
                event.file_path,
                event.full_path,
                event.mtime,
                event.atime,
                event.file_size,
                event.subsession_id,
                event.action,
                event.session_id
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
        for (file_path, full_path, mtime, atime, file_size) in file_paths {
            self.log_event(EventData {
                query,
                file_path,
                full_path,
                mtime: *mtime,
                atime: *atime,
                file_size: *file_size,
                subsession_id,
                action: "impression",
                session_id,
            })?;
        }

        Ok(())
    }

    pub fn log_click(&self, event: EventData) -> Result<()> {
        self.log_event(event)
    }

    pub fn log_scroll(&self, event: EventData) -> Result<()> {
        self.log_event(event)
    }

    pub fn get_previously_interacted_files(&self) -> Result<Vec<String>> {
        // Get all unique full_paths that have been clicked or scrolled
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT full_path
             FROM events
             WHERE action IN ('click', 'scroll')
             ORDER BY timestamp DESC",
        )?;

        let paths = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<String>, _>>()?;

        Ok(paths)
    }
}
