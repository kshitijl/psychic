use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::path::{Path, PathBuf};

const SECONDS_PER_DAY: i64 = 24 * 60 * 60;

/// How far back `get_previously_interacted_files` looks.
///
/// A year: long enough that "the file I was working on last spring" is still
/// findable, short enough that the query stays proportional to a year of use
/// rather than to everything in the database. Events are never deleted, so
/// without a cutoff this grows for the life of the installation.
const HISTORY_LOOKBACK_DAYS: i64 = 365;

/// Most paths `get_previously_interacted_files` will return.
///
/// Each one becomes a file registry entry that every subsequent search filters
/// over, so this caps steady-state cost as well as startup cost. Results are
/// ordered most-recent-first, so hitting the cap drops the stalest paths.
const HISTORY_MAX_PATHS: usize = 2_000;

/// One click or scroll, as the ranker wants it.
#[derive(Debug, Clone)]
pub struct Engagement {
    pub full_path: String,
    pub timestamp: i64,
    pub query: String,
    pub episode_queries: Option<String>,
}

pub struct FileMetadata {
    pub relative_path: String,
    pub full_path: String,
    pub mtime: Option<i64>,
    pub atime: Option<i64>,
    pub size: Option<i64>,
}

/// A snapshot of what the database holds, for the debug pane.
///
/// Counting rows means scanning the whole action index, so this is gathered off
/// the UI thread and only when the debug pane is actually opened - see
/// `App::request_db_stats`.
#[derive(Debug, Clone, Default)]
pub struct DbStats {
    pub total_events: i64,
    pub impressions: i64,
    pub clicks: i64,
    pub scrolls: i64,
    pub startup_visits: i64,
    pub sessions: i64,
    pub file_size_bytes: u64,
    /// Age of the oldest event, in days: how much history this represents.
    pub history_days: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct ContextData {
    pub cwd: String,
    pub gateway: String,
    pub subnet: String,
    pub dns: String,
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

/// Databases whose schema this process has already set up.
///
/// A `Connection` is `Send` but not `Sync`, so every thread that touches the
/// database needs its own - that part is not waste. Re-running `CREATE TABLE IF
/// NOT EXISTS` and the migration down each of them is: it cost about 700us an
/// open, for work that can only do something the first time.
static PREPARED: std::sync::Mutex<Option<std::collections::HashSet<PathBuf>>> =
    std::sync::Mutex::new(None);

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new(db_path: &Path) -> Result<Self> {
        let opened_at = std::time::Instant::now();
        let conn = Connection::open(db_path).context("Failed to open database")?;

        // Schema and migration are per *file*, not per connection - except that
        // `:memory:` is not a file. Every in-memory connection is its own
        // database that happens to share the name, so remembering it would
        // leave the second one empty. Tests live on that.
        let shared_by_path = db_path != Path::new(":memory:") && db_path != Path::new("");

        // The first open of a file is done holding the lock, so that a second
        // thread arriving at the same moment waits for it rather than racing.
        // Both halves of that first open take a write lock on the database:
        // switching a fresh one to WAL is a write, and so is creating the
        // tables. On a first launch four threads reach here at once, and the
        // losers used to fail outright - a brand-new install logged a failed
        // retrain on its very first run.
        let first_time = {
            let mut prepared = PREPARED.lock().expect("schema registry is not poisoned");
            let known = prepared.get_or_insert_with(Default::default);
            let first = !shared_by_path || known.insert(db_path.to_path_buf());
            if first {
                Self::configure(&conn)?;
                Self::prepare_schema(&conn)?;
            }
            first
        };
        if !first_time {
            Self::configure(&conn)?;
        }

        log::info!(
            "TIMING {{\"op\":\"db_open\",\"ms\":{},\"schema\":{}}}",
            opened_at.elapsed().as_secs_f64() * 1000.0,
            first_time
        );

        Ok(Database { conn })
    }

    /// Settings every connection needs. Per connection, not per database.
    ///
    /// `busy_timeout` comes first because the statement after it takes a write
    /// lock on a database that is not yet in WAL, and without a timeout already
    /// in force that fails immediately rather than waiting.
    fn configure(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "PRAGMA busy_timeout = 5000;
             PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;",
        )?;
        Ok(())
    }

    /// Create what is missing and bring what is there up to date.
    ///
    /// Runs once per database file per process; see [`PREPARED`].
    fn prepare_schema(conn: &Connection) -> Result<()> {
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
                timezone TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )",
            [],
        )?;

        // Directories the user never wants to see in results again.
        //
        // Its own table, not an `events` row: this is mutable, undoable state,
        // and hiding is deliberately not a training signal. It suppresses rows
        // at display time and nothing else - the events under a hidden
        // directory stay exactly as they are, and the model never learns from
        // the fact that it was hidden.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hidden_prefixes (
                path TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL
            )",
            [],
        )?;

        Self::migrate(conn)
    }

    /// Bring an existing database up to what the code above expects.
    ///
    /// Runs on every open, so every step is a no-op once it has been done. It
    /// lives here rather than at start-up because every path that opens the
    /// database - the TUI, `retrain`, `track-visit`, `hidden` - writes through
    /// the same statements, and one of them inserts a session row that would
    /// fail against the old, wider table.
    fn migrate(conn: &Connection) -> Result<()> {
        let existing: Vec<String> = conn
            .prepare("SELECT name FROM pragma_table_info('sessions')")?
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<_>>()?;

        // `running_processes` was the output of `ps` and `shell_history` the
        // last ten commands typed, both collected every launch and read by
        // nothing. Together they were 40MB of a 67MB database, and the second
        // is a plain-text copy of what the user has been doing.
        let mut dropped = false;
        for column in ["running_processes", "shell_history"] {
            if existing.iter().any(|name| name == column) {
                log::info!("Dropping unused sessions.{} column", column);
                conn.execute(&format!("ALTER TABLE sessions DROP COLUMN {}", column), [])?;
                dropped = true;
            }
        }

        // The old index covered every row, and 96% of them are impressions
        // that nothing looks up by action. Its entries carry `full_path`, so
        // covering them all cost 10MB.
        conn.execute("DROP INDEX IF EXISTS idx_events_click_lookup", [])?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_engagement
             ON events(action, timestamp, full_path)
             WHERE action IN ('click', 'scroll', 'startup_visit')",
            [],
        )?;
        // The debug pane counts events by action, which the partial index
        // cannot answer because it does not hold the rows being counted. This
        // one has no `full_path` in it, so it is a fraction of the size.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_action ON events(action)",
            [],
        )?;

        if dropped {
            // Dropping a column rewrites the rows but does not give the pages
            // back. Once, and only when there was something to reclaim.
            log::info!("Compacting the database after dropping unused columns");
            let start = std::time::Instant::now();
            conn.execute_batch("VACUUM")?;
            log::info!(
                "TIMING {{\"op\":\"vacuum\",\"ms\":{}}}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(())
    }

    /// The connection underneath, for the bulk read that feature generation
    /// does.
    ///
    /// Narrow on purpose. Training reads the whole table into its own types,
    /// which do not belong in this module; everything else goes through the
    /// methods here, which is what keeps the pragmas and the schema in one
    /// place. Before this, feature generation opened its own connection with
    /// no pragmas at all, so it had no `busy_timeout` either.
    pub(crate) fn connection(&self) -> &Connection {
        &self.conn
    }

    pub fn get_db_path(data_dir: &Path) -> PathBuf {
        data_dir.join("events.db")
    }

    pub fn log_session(&self, session_id: &str, context: &ContextData) -> Result<()> {
        let timestamp = jiff::Timestamp::now().as_second();

        self.conn.execute(
            "INSERT INTO sessions (session_id, cwd, gateway, subnet, dns, timezone, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id,
                &context.cwd,
                &context.gateway,
                &context.subnet,
                &context.dns,
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

        self.conn.prepare_cached(
            "INSERT INTO events (timestamp, query, file_path, full_path, mtime, atime, file_size, subsession_id, action, session_id, episode_queries)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        )?.execute(
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

    /// Log a screenful of impressions as one commit.
    ///
    /// These arrive two dozen at a time, on the UI thread, after every query
    /// the user pauses on. One statement each meant a transaction and an fsync
    /// each.
    pub fn log_impressions(
        &self,
        query: &str,
        file_paths: &[FileMetadata],
        subsession_id: u64,
        session_id: &str,
    ) -> Result<()> {
        let transaction = self.conn.unchecked_transaction()?;

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

        transaction.commit()?;

        Ok(())
    }

    /// Stop showing `path` and everything under it in search results.
    ///
    /// Idempotent: hiding an already-hidden directory keeps the original
    /// `created_at`, so the record says when the user first decided this.
    pub fn hide_prefix(&self, path: &Path) -> Result<()> {
        let timestamp = jiff::Timestamp::now().as_second();
        self.conn.execute(
            "INSERT OR IGNORE INTO hidden_prefixes (path, created_at) VALUES (?1, ?2)",
            params![path.to_string_lossy(), timestamp],
        )?;
        Ok(())
    }

    /// Undo `hide_prefix`. Returns whether anything was actually hidden.
    pub fn unhide_prefix(&self, path: &Path) -> Result<bool> {
        let removed = self.conn.execute(
            "DELETE FROM hidden_prefixes WHERE path = ?1",
            params![path.to_string_lossy()],
        )?;
        Ok(removed > 0)
    }

    /// Every hidden directory, most recently hidden first.
    ///
    /// Unbounded on purpose, unlike `get_previously_interacted_files`: this
    /// list only grows when the user explicitly adds to it, so it is small by
    /// construction and does not need a cap to stay off the startup budget.
    pub fn get_hidden_prefixes(&self) -> Result<Vec<PathBuf>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path FROM hidden_prefixes ORDER BY created_at DESC")?;

        let paths = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .map(PathBuf::from)
            .collect();

        Ok(paths)
    }

    /// Paths the user has clicked, scrolled, or visited, most recently
    /// interacted with first.
    ///
    /// This runs on the startup critical path, so it carries two bounds:
    ///
    /// * `HISTORY_LOOKBACK_DAYS` bounds the **work**. `action` and `timestamp` are
    ///   the first two columns of `idx_events_click_lookup`, so the cutoff turns
    ///   this into a range seek over one year of history rather than a scan of
    ///   everything ever recorded. Without it this was the only query on the
    ///   startup path that grew forever.
    /// * `HISTORY_MAX_PATHS` bounds the **result**. Every path returned becomes a
    ///   file registry entry that each later query filters over, so the cap
    ///   protects steady-state search cost, not just startup. The ordering means
    ///   the cap keeps the most recently used paths and drops the stalest.
    ///
    /// Both bounds are generous enough that a normal history never reaches them;
    /// they exist so that an unusual one degrades instead of getting slower.
    ///
    /// Why `GROUP BY` rather than `SELECT DISTINCT ... ORDER BY timestamp`: the
    /// caller registers these in order and file registry order breaks ties between
    /// equally scored results, so the order matters. `DISTINCT` does not define
    /// one, because the sort key is not in the result, so which of a path's many
    /// timestamps wins is up to SQLite. It comes out close to recency order, but
    /// only close. `MAX(timestamp)` says what we mean, with the same query plan.
    pub fn get_previously_interacted_files(&self) -> Result<Vec<String>> {
        let cutoff = jiff::Timestamp::now().as_second() - HISTORY_LOOKBACK_DAYS * SECONDS_PER_DAY;

        let mut stmt = self.conn.prepare(
            "SELECT full_path
             FROM events
             WHERE action IN ('click', 'scroll', 'startup_visit')
               AND timestamp >= ?1
             GROUP BY full_path
             ORDER BY MAX(timestamp) DESC
             LIMIT ?2",
        )?;

        let paths = stmt
            .query_map(params![cutoff, HISTORY_MAX_PATHS], |row| {
                row.get::<_, String>(0)
            })?
            .collect::<Result<Vec<String>, _>>()?;

        Ok(paths)
    }

    /// Every click and scroll since `cutoff`, for the ranker's indexes.
    ///
    /// The SQL lives here, beside the index it depends on and the plan test
    /// that checks it is used, rather than in `ranker.rs` with its own
    /// connection and its own copy of the pragmas.
    ///
    /// **The first `action` clause is redundant and load-bearing.** SQLite uses
    /// a partial index only when the query's WHERE provably implies the
    /// index's, and it does not work out that a two-item `IN` list implies the
    /// three-item one `idx_events_engagement` was built with. Without that line
    /// this is a full table scan. See `plan_tests`.
    pub fn engagements_since(&self, cutoff: i64) -> Result<Vec<Engagement>> {
        let mut stmt = self.conn.prepare(
            "SELECT full_path, timestamp, query, episode_queries
             FROM events
             WHERE action IN ('click', 'scroll', 'startup_visit')
               AND action IN ('click', 'scroll')
               AND timestamp >= ?1",
        )?;

        let rows = stmt
            .query_map([cutoff], |row| {
                Ok(Engagement {
                    full_path: row.get(0)?,
                    timestamp: row.get(1)?,
                    query: row.get(2)?,
                    episode_queries: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        Ok(rows)
    }

    /// Count what is in the database, for the debug pane.
    ///
    /// The action histogram is a covering-index scan, so it is proportional to
    /// total events - the only place psychic reads the whole table outside of
    /// training. Fine for an on-demand debug view, not for the startup path.
    pub fn stats(&self, db_path: &Path) -> Result<DbStats> {
        let mut stats = DbStats {
            file_size_bytes: std::fs::metadata(db_path).map(|m| m.len()).unwrap_or(0),
            ..Default::default()
        };

        for (action, count) in self.summarize_events()? {
            stats.total_events += count;
            match action.as_str() {
                "impression" => stats.impressions = count,
                "click" => stats.clicks = count,
                "scroll" => stats.scrolls = count,
                "startup_visit" => stats.startup_visits = count,
                // An action added later still counts toward the total.
                _ => {}
            }
        }

        stats.sessions = self
            .conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
            .unwrap_or(0);

        let oldest: Option<i64> = self
            .conn
            .query_row("SELECT MIN(timestamp) FROM events", [], |row| row.get(0))
            .unwrap_or(None);
        stats.history_days =
            oldest.map(|oldest| (jiff::Timestamp::now().as_second() - oldest) / SECONDS_PER_DAY);

        Ok(stats)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An in-memory database with the real schema.
    fn test_db() -> Database {
        Database::new(Path::new(":memory:")).expect("Failed to open in-memory database")
    }

    /// Record an interaction with `path` a given number of days ago.
    fn record(db: &Database, path: &str, days_ago: i64) {
        record_seconds_ago(db, path, days_ago * SECONDS_PER_DAY);
    }

    /// Record an interaction at second granularity, for cases needing more
    /// distinct timestamps than the lookback window has days.
    fn record_seconds_ago(db: &Database, path: &str, seconds_ago: i64) {
        let timestamp = jiff::Timestamp::now().as_second() - seconds_ago;
        db.conn
            .execute(
                "INSERT INTO events (timestamp, query, file_path, full_path, action, session_id)
                 VALUES (?1, '', ?2, ?2, 'click', 's')",
                params![timestamp, path],
            )
            .expect("Failed to insert test event");
    }

    #[test]
    fn test_history_is_ordered_by_most_recent_interaction() {
        let db = test_db();
        record(&db, "/old", 10);
        record(&db, "/middle", 5);
        record(&db, "/new", 1);
        // An older touch of /old must not move it up; only its latest one counts.
        record(&db, "/old", 9);

        assert_eq!(
            db.get_previously_interacted_files().unwrap(),
            vec![
                "/new".to_string(),
                "/middle".to_string(),
                "/old".to_string()
            ],
            "Most recently interacted with comes first"
        );
    }

    #[test]
    fn test_history_appears_once_however_often_it_was_used() {
        let db = test_db();
        for days_ago in 1..=20 {
            record(&db, "/used/a/lot", days_ago);
        }
        record(&db, "/used/once", 2);

        assert_eq!(
            db.get_previously_interacted_files().unwrap(),
            vec!["/used/a/lot".to_string(), "/used/once".to_string()],
            "20 interactions still yield one entry, ranked by its latest"
        );
    }

    #[test]
    fn test_history_older_than_the_lookback_is_dropped() {
        let db = test_db();
        record(&db, "/just/inside", HISTORY_LOOKBACK_DAYS - 1);
        record(&db, "/just/outside", HISTORY_LOOKBACK_DAYS + 1);
        record(&db, "/ancient", 5 * HISTORY_LOOKBACK_DAYS);

        assert_eq!(
            db.get_previously_interacted_files().unwrap(),
            vec!["/just/inside".to_string()],
            "The lookback window is what keeps this query from growing forever"
        );
    }

    #[test]
    fn test_history_is_capped_and_keeps_the_most_recent() {
        let db = test_db();
        // One more path than the cap allows, all inside the window, oldest first.
        for i in 0..=HISTORY_MAX_PATHS {
            let seconds_ago = (HISTORY_MAX_PATHS - i) as i64;
            record_seconds_ago(&db, &format!("/path/{:05}", i), seconds_ago);
        }

        let paths = db.get_previously_interacted_files().unwrap();
        assert_eq!(
            paths.len(),
            HISTORY_MAX_PATHS,
            "Result is capped no matter how much history exists"
        );
        assert!(
            !paths.contains(&"/path/00000".to_string()),
            "The stalest path is the one dropped"
        );
    }

    #[test]
    fn test_history_ignores_impressions() {
        let db = test_db();
        record(&db, "/clicked", 1);
        db.conn
            .execute(
                "INSERT INTO events (timestamp, query, file_path, full_path, action, session_id)
                 VALUES (?1, '', '/seen', '/seen', 'impression', 's')",
                params![jiff::Timestamp::now().as_second()],
            )
            .unwrap();

        assert_eq!(
            db.get_previously_interacted_files().unwrap(),
            vec!["/clicked".to_string()],
            "Impressions are 96% of the table and none of them are history"
        );
    }
}

#[cfg(test)]
mod plan_tests {
    //! Every query psychic runs against `events`, and the index it must use.
    //!
    //! A partial index is only used when SQLite can prove the query's WHERE
    //! implies the index's. That proof is easy to break by editing a WHERE
    //! clause - narrowing an `IN` list is enough - and the failure is silent:
    //! the query keeps working and quietly scans the table instead.
    use super::*;

    fn plan(db: &Database, query: &str) -> String {
        db.conn
            .prepare(&format!("EXPLAIN QUERY PLAN {}", query))
            .expect("query should parse")
            .query_map([], |row| row.get::<_, String>(3))
            .expect("plan should run")
            .collect::<rusqlite::Result<Vec<_>>>()
            .expect("plan rows")
            .join(" | ")
    }

    #[test]
    fn test_the_history_query_uses_the_engagement_index() {
        let db = Database::new(Path::new(":memory:")).unwrap();

        let plan = plan(
            &db,
            "SELECT full_path FROM events
             WHERE action IN ('click', 'scroll', 'startup_visit') AND timestamp >= 1
             GROUP BY full_path ORDER BY MAX(timestamp) DESC LIMIT 10",
        );

        assert!(
            plan.contains("idx_events_engagement"),
            "get_previously_interacted_files must not scan the table: {}",
            plan
        );
    }

    #[test]
    fn test_the_click_loading_query_uses_the_engagement_index() {
        let db = Database::new(Path::new(":memory:")).unwrap();

        // Exactly what `Ranker::load_clicks` runs, redundant clause and all.
        let plan = plan(
            &db,
            "SELECT full_path, timestamp, query, episode_queries FROM events
             WHERE action IN ('click', 'scroll', 'startup_visit')
               AND action IN ('click', 'scroll')
               AND timestamp >= 1",
        );

        assert!(
            plan.contains("idx_events_engagement"),
            "load_clicks must not scan the table: {}",
            plan
        );
    }

    #[test]
    fn test_narrowing_the_list_without_the_redundant_clause_loses_the_index() {
        // Why that redundant clause is there, written down as a test so that
        // deleting it fails rather than silently costing a scan.
        let db = Database::new(Path::new(":memory:")).unwrap();

        let plan = plan(
            &db,
            "SELECT full_path FROM events
             WHERE action IN ('click', 'scroll') AND timestamp >= 1",
        );

        assert!(
            !plan.contains("idx_events_engagement"),
            "if SQLite has learned to prove this, the redundant clause in \
             load_clicks can go: {}",
            plan
        );
    }

    #[test]
    fn test_the_event_histogram_uses_an_index() {
        let db = Database::new(Path::new(":memory:")).unwrap();

        let plan = plan(&db, "SELECT action, COUNT(*) FROM events GROUP BY action");

        assert!(
            plan.contains("idx_events_action"),
            "the partial index cannot count rows it excludes, which is what \
             the narrow index on action alone is for: {}",
            plan
        );
    }
}
