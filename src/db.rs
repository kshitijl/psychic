use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ContextData {
    pub cwd: String,
    pub gateway: String,
    pub subnet: String,
    pub dns: String,
    pub shell_history: String,
    pub running_processes: String,
}

pub struct EventData<'a> {
    pub query: &'a str,
    pub file_path: &'a str,
    pub full_path: &'a str,
    pub mtime: Option<i64>,
    pub atime: Option<i64>,
    pub action: &'a str,
    pub session_id: &'a str,
}

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new() -> Result<Self> {
        let db_path = Self::get_db_path()?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open database")?;

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
                created_at INTEGER NOT NULL
            )",
            [],
        )?;

        Ok(Database { conn })
    }

    pub fn get_db_path() -> Result<PathBuf> {
        let home = std::env::var("HOME")
            .context("HOME environment variable not set")?;
        Ok(PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("sg")
            .join("events.db"))
    }

    pub fn log_session(&self, session_id: &str, context: &ContextData) -> Result<()> {
        let timestamp = jiff::Timestamp::now().as_second();

        self.conn.execute(
            "INSERT INTO sessions (session_id, cwd, gateway, subnet, dns, shell_history, running_processes, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                session_id,
                &context.cwd,
                &context.gateway,
                &context.subnet,
                &context.dns,
                &context.shell_history,
                &context.running_processes,
                timestamp
            ],
        )?;

        Ok(())
    }

    pub fn log_event(&self, event: EventData) -> Result<()> {
        let timestamp = jiff::Timestamp::now().as_second();

        self.conn.execute(
            "INSERT INTO events (timestamp, query, file_path, full_path, mtime, atime, action, session_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                timestamp,
                event.query,
                event.file_path,
                event.full_path,
                event.mtime,
                event.atime,
                event.action,
                event.session_id
            ],
        )?;

        Ok(())
    }

    pub fn log_impressions(
        &self,
        query: &str,
        file_paths: &[(String, String, Option<i64>, Option<i64>)], // (relative, full, mtime, atime)
        session_id: &str,
    ) -> Result<()> {
        for (file_path, full_path, mtime, atime) in file_paths {
            self.log_event(EventData {
                query,
                file_path,
                full_path,
                mtime: *mtime,
                atime: *atime,
                action: "impression",
                session_id,
            })?;
        }

        Ok(())
    }

    pub fn log_click(
        &self,
        query: &str,
        file_path: &str,
        full_path: &str,
        mtime: Option<i64>,
        atime: Option<i64>,
        session_id: &str,
    ) -> Result<()> {
        self.log_event(EventData {
            query,
            file_path,
            full_path,
            mtime,
            atime,
            action: "click",
            session_id,
        })
    }

    pub fn log_scroll(
        &self,
        query: &str,
        file_path: &str,
        full_path: &str,
        mtime: Option<i64>,
        atime: Option<i64>,
        session_id: &str,
    ) -> Result<()> {
        self.log_event(EventData {
            query,
            file_path,
            full_path,
            mtime,
            atime,
            action: "scroll",
            session_id,
        })
    }
}
