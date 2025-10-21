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

use crate::db::{Database, EventData, FileMetadata};

/// Every time the query changes, as the user types, corresponds to a new
/// subsession. Subsession id is logged to the db.
pub struct Subsession {
    pub id: u64,
    pub query: String,
    pub created_at: jiff::Timestamp,
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

    /// Check if impressions should be logged and log them if so
    /// - force: if true, log even if <200ms old
    /// - Returns Ok(()) even if logging is disabled
    pub fn check_and_log_impressions(
        &mut self,
        force: bool,
        top_n_files: Vec<FileMetadata>,
    ) -> Result<()> {
        if self.no_logging {
            return Ok(());
        }

        // Extract values we need before borrowing subsession mutably
        let (subsession_id, subsession_query, created_at, already_logged) =
            match &self.current_subsession {
                Some(s) => (
                    s.id,
                    s.query.clone(),
                    s.created_at,
                    s.events_have_been_logged,
                ),
                None => return Ok(()),
            };

        // Skip if already logged
        if already_logged {
            return Ok(());
        }

        // Check if we should log: either forced or >200ms old
        let elapsed = jiff::Timestamp::now().duration_since(created_at);
        let threshold = jiff::SignedDuration::from_millis(200);
        let should_log = force || elapsed >= threshold;
        if !should_log {
            return Ok(());
        }

        // Log impressions
        if !top_n_files.is_empty() {
            self.db
                .log_impressions(&subsession_query, &top_n_files, subsession_id, &self.session_id)?;

            // Mark as logged
            if let Some(ref mut s) = self.current_subsession {
                s.events_have_been_logged = true;
            }
        }

        Ok(())
    }

    /// Log a scroll event for a file (deduplicated by query + full_path)
    pub fn log_scroll(&mut self, query: &str, event_data: EventData) -> Result<()> {
        if self.no_logging {
            return Ok(());
        }

        let key = (query.to_string(), event_data.full_path.to_string());

        if !self.scrolled_files.contains(&key) {
            self.db.log_event(event_data)?;
            self.scrolled_files.insert(key);
        }

        Ok(())
    }

    /// Log a click event
    pub fn log_click(&self, event_data: EventData) -> Result<()> {
        if self.no_logging {
            return Ok(());
        }

        self.db.log_event(event_data)
    }

    /// Create a new subsession (when query changes)
    pub fn new_subsession(&mut self, query_id: u64, query: String) {
        self.current_subsession = Some(Subsession {
            id: query_id,
            query,
            created_at: jiff::Timestamp::now(),
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
