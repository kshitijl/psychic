//! Episode tracking module
//!
//! An episode represents a sequence of queries (subsessions) leading up to an engagement event
//! (click or scroll). This module tracks all unique queries seen during an episode.
//!
//! Example:
//! - User types "tc" → "todo" → "todo-current" → clicks a file
//! - Episode queries: ["tc", "todo", "todo-current"]
//! - All three queries get credit for that file selection

use anyhow::Result;

/// Tracks all unique queries in the current episode
#[derive(Debug, Clone)]
pub struct Episode {
    queries: Vec<String>,
}

impl Episode {
    /// Create a new empty episode
    pub fn new() -> Self {
        Self {
            queries: Vec::new(),
        }
    }

    /// Add a query to the episode if not already present
    pub fn add_query(&mut self, query: &str) {
        if !self.queries.contains(&query.to_string()) {
            self.queries.push(query.to_string());
        }
    }

    /// Serialize queries to JSON for database storage
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(&self.queries)?)
    }

    /// Clear all queries (called after engagement event)
    pub fn clear(&mut self) {
        self.queries.clear();
    }

    /// Get reference to queries (testing only)
    #[cfg(test)]
    pub fn queries(&self) -> &[String] {
        &self.queries
    }
}

impl Default for Episode {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_episode_is_empty() {
        let episode = Episode::new();
        assert_eq!(
            episode.queries(),
            &[] as &[String],
            "New episode should be empty"
        );
    }

    #[test]
    fn test_add_single_query() {
        let mut episode = Episode::new();
        episode.add_query("test");
        assert_eq!(episode.queries(), &["test"], "Should contain single query");
    }

    #[test]
    fn test_add_duplicate_query_ignored() {
        let mut episode = Episode::new();
        episode.add_query("test");
        episode.add_query("test");
        assert_eq!(
            episode.queries(),
            &["test"],
            "Duplicate query should not be added"
        );
    }

    #[test]
    fn test_add_multiple_unique_queries() {
        let mut episode = Episode::new();
        episode.add_query("tc");
        episode.add_query("todo");
        episode.add_query("todo-current");
        assert_eq!(
            episode.queries(),
            &["tc", "todo", "todo-current"],
            "Should contain all unique queries in order"
        );
    }

    #[test]
    fn test_to_json_empty() {
        let episode = Episode::new();
        let json = episode
            .to_json()
            .expect("JSON serialization should succeed");
        assert_eq!(json, "[]", "Empty episode should serialize to empty array");
    }

    #[test]
    fn test_to_json_with_queries() {
        let mut episode = Episode::new();
        episode.add_query("tc");
        episode.add_query("todo");
        let json = episode
            .to_json()
            .expect("JSON serialization should succeed");
        assert_eq!(
            json, "[\"tc\",\"todo\"]",
            "Episode should serialize to JSON array"
        );
    }

    #[test]
    fn test_clear_empties_episode() {
        let mut episode = Episode::new();
        episode.add_query("test");
        episode.add_query("another");
        episode.clear();
        assert_eq!(
            episode.queries(),
            &[] as &[String],
            "Clear should remove all queries"
        );
    }

    #[test]
    fn test_clear_allows_reuse() {
        let mut episode = Episode::new();
        episode.add_query("first");
        episode.clear();
        episode.add_query("second");
        assert_eq!(
            episode.queries(),
            &["second"],
            "After clear, episode should start fresh"
        );
    }
}
