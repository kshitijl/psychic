//! Analytics module - subsession tracking, impression logging, scroll tracking
//!
//! This module provides a clean interface for tracking user interactions:
//! - Subsession management (query changes)
//! - Impression debouncing (200ms logic)
//! - Scroll deduplication (HashSet tracking)
//! - Event data formatting
//! - Database writing
//!
//! Deep implementation hiding complexity of timing/deduplication behind a simple interface
//! (just call log_* methods).

use anyhow::Result;
use std::collections::HashSet;
use std::time::{Duration, Instant};

use crate::db::{Database, EventData, FileMetadata};

/// How long a query has to have been on screen before its results count as
/// impressions. Faster than this and the user never saw them - they were typing
/// through it on the way to something else.
const IMPRESSION_AGE: Duration = Duration::from_millis(200);

/// Every time the query changes, as the user types, corresponds to a new
/// subsession. Subsession id is logged to the db.
pub struct Subsession {
    pub id: u64,
    pub query: String,
    /// When this query was typed. An `Instant`, not a wall clock: it is only
    /// ever used for the 200ms "has this been on screen long enough to count as
    /// an impression" test, and a wall clock can go backwards under it.
    pub created_at: Instant,
    pub events_have_been_logged: bool,
}

/// Tracks analytics state for the application
pub struct Analytics {
    /// Current subsession (changes with each query)
    pub current_subsession: Option<Subsession>,
    /// Next subsession ID to assign
    pub next_subsession_id: u64,
    /// Tracks which files we've logged scroll events for (to avoid duplicates)
    scrolled_files: HashSet<(String, String)>, // (query, full_path)
    /// Current episode (tracks all queries until engagement event)
    /// Every distinct query seen since the last engagement.
    ///
    /// The user types "tc", then "todo", then "todo-current", then clicks: all
    /// three queries get credit for that file, which is what
    /// `engagements_in_episode_with_query` is computed from. Cleared when the
    /// engagement is logged.
    episode_queries: Vec<String>,
    /// Session ID for this app instance
    session_id: String,
    /// Database handle
    db: Database,
    /// Whether logging is disabled
    no_logging: bool,
}

impl Analytics {
    pub fn new(session_id: String, db: Database, no_logging: bool) -> Self {
        Self {
            current_subsession: None,
            next_subsession_id: 1, // Start with 1, 0 is for initial query
            scrolled_files: HashSet::new(),
            episode_queries: Vec::new(),
            session_id,
            db,
            no_logging,
        }
    }

    /// Get the current subsession ID (or 0 if none)
    pub fn current_subsession_id(&self) -> u64 {
        self.current_subsession.as_ref().map(|s| s.id).unwrap_or(0)
    }

    /// Get the session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Record the current query as part of this episode, and say whether the
    /// impressions of that query are worth collecting.
    ///
    /// Split from the logging so the caller does not have to build the row list
    /// first. Most calls answer `false` - the query has already been logged, or
    /// has not been on screen long enough - and the list is one clone of a path
    /// and a display name per visible row, built on every event.
    ///
    /// The episode bookkeeping happens either way: a query the user typed on the
    /// way to a click counts towards that click whether or not its own
    /// impressions were logged.
    pub fn wants_impressions(&mut self, force: bool) -> bool {
        if self.no_logging {
            return false;
        }

        let Some(subsession) = &self.current_subsession else {
            return false;
        };
        let (created_at, already_logged) =
            (subsession.created_at, subsession.events_have_been_logged);
        let query = subsession.query.clone();

        if !self.episode_queries.contains(&query) {
            self.episode_queries.push(query);
        }

        !already_logged && (force || created_at.elapsed() >= IMPRESSION_AGE)
    }

    /// Log the rows that were on screen. Call `wants_impressions` first.
    pub fn log_impressions(&mut self, top_n_files: Vec<FileMetadata>) -> Result<()> {
        let (subsession_id, subsession_query) = match &self.current_subsession {
            Some(s) => (s.id, s.query.clone()),
            None => return Ok(()),
        };

        // Log impressions
        if !top_n_files.is_empty() {
            self.db.log_impressions(
                &subsession_query,
                &top_n_files,
                subsession_id,
                &self.session_id,
            )?;

            // Mark as logged
            if let Some(ref mut s) = self.current_subsession {
                s.events_have_been_logged = true;
            }
        }

        Ok(())
    }

    /// Helper: Log an engagement event (click or scroll) with episode queries
    /// This serializes the current episode queries, logs the event, and clears the episode
    fn log_engagement<'a>(
        &mut self,
        mut event_data: EventData<'a>,
        episode_json: &'a str,
    ) -> Result<()> {
        event_data.episode_queries = Some(episode_json);
        self.db.log_event(event_data)?;
        self.episode_queries.clear();
        Ok(())
    }

    pub fn log_scroll(&mut self, query: &str, event_data: EventData) -> Result<()> {
        if self.no_logging {
            return Ok(());
        }

        let key = (query.to_string(), event_data.full_path.to_string());

        if !self.scrolled_files.contains(&key) {
            let episode_json = serde_json::to_string(&self.episode_queries)?;
            self.log_engagement(event_data, &episode_json)?;
            self.scrolled_files.insert(key);
        }

        Ok(())
    }

    /// Log a click event
    pub fn log_click(&mut self, event_data: EventData) -> Result<()> {
        if self.no_logging {
            return Ok(());
        }

        let episode_json = serde_json::to_string(&self.episode_queries)?;
        self.log_engagement(event_data, &episode_json)
    }

    /// Create a new subsession (when query changes)
    pub fn new_subsession(&mut self, query_id: u64, query: String) {
        self.current_subsession = Some(Subsession {
            id: query_id,
            query,
            created_at: Instant::now(),
            events_have_been_logged: false,
        });
    }

    /// Get the next subsession ID and increment
    pub fn next_subsession_id(&mut self) -> u64 {
        let id = self.next_subsession_id;
        self.next_subsession_id += 1;
        id
    }
}

#[cfg(test)]
mod episode_tests {
    //! An episode is every distinct query the user typed on the way to one
    //! engagement. The user types "tc", then "todo", then "todo-current", then
    //! clicks: all three get credit for that file, which is what the
    //! `engagements_in_episode_with_query` feature is computed from.
    //!
    //! This used to be an `Episode` struct in its own module - a `Vec<String>`,
    //! a `contains` check and a `to_json`. These tests are what it took with it.

    use super::*;
    use crate::db::UserInteraction;
    use std::path::Path;

    fn analytics() -> Analytics {
        let db = Database::new(Path::new(":memory:")).expect("in-memory database");
        Analytics::new("test-session".to_string(), db, false)
    }

    fn shown(name: &str) -> Vec<FileMetadata> {
        vec![FileMetadata {
            relative_path: name.to_string(),
            full_path: format!("/tmp/{}", name),
            mtime: Some(1_700_000_000),
            size: Some(100),
            is_dir: false,
        }]
    }

    /// Type a query and let its results count as seen.
    fn typed(analytics: &mut Analytics, query: &str) {
        let id = analytics.next_subsession_id;
        analytics.new_subsession(id, query.to_string());
        if analytics.wants_impressions(true) {
            analytics
                .log_impressions(shown("a.rs"))
                .expect("impressions");
        }
    }

    fn clicked(analytics: &mut Analytics) {
        analytics
            .log_click(EventData {
                query: "todo",
                file_path: "a.rs",
                full_path: "/tmp/a.rs",
                mtime: Some(1_700_000_000),
                file_size: Some(100),
                subsession_id: 1,
                action: UserInteraction::Click,
                session_id: "test-session",
                episode_queries: None,
                rank: None, // not an impression
                is_dir: Some(false),
            })
            .expect("click");
    }

    #[test]
    fn test_every_query_on_the_way_to_a_click_is_remembered_in_order() {
        let mut analytics = analytics();

        typed(&mut analytics, "tc");
        typed(&mut analytics, "todo");
        typed(&mut analytics, "todo-current");

        assert_eq!(analytics.episode_queries, ["tc", "todo", "todo-current"]);
    }

    #[test]
    fn test_a_query_typed_twice_is_only_counted_once() {
        let mut analytics = analytics();

        typed(&mut analytics, "todo");
        typed(&mut analytics, "notes");
        typed(&mut analytics, "todo");

        assert_eq!(
            analytics.episode_queries,
            ["todo", "notes"],
            "returning to an earlier query does not repeat it"
        );
    }

    #[test]
    fn test_the_episode_starts_over_after_an_engagement() {
        let mut analytics = analytics();

        typed(&mut analytics, "tc");
        typed(&mut analytics, "todo");
        clicked(&mut analytics);

        assert!(
            analytics.episode_queries.is_empty(),
            "the click ended the episode"
        );

        typed(&mut analytics, "notes");
        assert_eq!(
            analytics.episode_queries,
            ["notes"],
            "and the next one starts from nothing"
        );
    }

    #[test]
    fn test_the_episode_reaches_the_database_as_json() {
        let mut analytics = analytics();
        typed(&mut analytics, "tc");
        typed(&mut analytics, "todo");

        assert_eq!(
            serde_json::to_string(&analytics.episode_queries).unwrap(),
            r#"["tc","todo"]"#
        );
    }

    #[test]
    fn test_nothing_is_recorded_when_logging_is_off() {
        let db = Database::new(Path::new(":memory:")).expect("in-memory database");
        let mut analytics = Analytics::new("test-session".to_string(), db, true);

        typed(&mut analytics, "todo");

        assert!(analytics.episode_queries.is_empty());
    }
}
