//! History navigation module
//!
//! Manages directory navigation history with branch-point semantics.
//!
//! ## Model - Invariant: cwd == dirs[current_index]
//! - `dirs` is a Vec<PathBuf> where dirs[i+1] came after dirs[i] in time
//! - `current_index` is our position in history
//! - **Invariant**: The current working directory is ALWAYS `dirs[current_index]`
//! - At startup: dirs = [starting_dir], current_index = 0
//! - Display shows: [dirs[len-1], dirs[len-2], ..., dirs[0]] (most recent first)

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct History {
    /// Directory history, last item is most recent
    dirs: Vec<PathBuf>,
    /// Current position in history
    /// Invariant: cwd == dirs[current_index]
    current_index: usize,
}

impl History {
    pub fn new(starting_dir: PathBuf) -> Self {
        Self {
            dirs: vec![starting_dir],
            current_index: 0,
        }
    }

    /// Get all items for display (in reverse order: most recent first)
    pub fn items_for_display(&self) -> Vec<PathBuf> {
        let mut items = Vec::new();

        // Add all history items in reverse order (most recent first)
        for i in (0..self.dirs.len()).rev() {
            items.push(self.dirs[i].clone());
        }

        items
    }

    /// Navigate to a directory from the history display
    ///
    /// # Arguments
    /// * `display_index` - Index in the display list (0 = most recent)
    ///
    /// # Returns
    /// The directory to navigate to (from dirs), or None if out of bounds
    pub fn navigate_to_display_index(&mut self, display_index: usize) -> Option<PathBuf> {
        // Map display index to history index
        // display[0] = dirs[len-1] (most recent)
        // display[1] = dirs[len-2]
        // ...
        // display[len-1] = dirs[0] (oldest)
        //
        // So: display[i] = dirs[len - 1 - i]

        if display_index >= self.dirs.len() {
            return None;
        }

        let history_index = self.dirs.len() - 1 - display_index;

        // Just update current_index, don't modify dirs
        self.current_index = history_index;

        Some(self.dirs[history_index].clone())
    }

    /// Navigate to a new directory (or existing one in history)
    ///
    /// # Arguments
    /// * `new_dir` - The directory we're navigating to
    ///
    /// This handles two cases:
    /// 1. If new_dir is the next item in history (dirs[current_index + 1]), just increment current_index
    /// 2. Otherwise, truncate history after current_index and push new_dir
    pub fn navigate_to(&mut self, new_dir: PathBuf) {
        // Check if new_dir is the next item in history
        if self.current_index + 1 < self.dirs.len() && self.dirs[self.current_index + 1] == new_dir
        {
            // Just move forward in history
            self.current_index += 1;
        } else {
            // Branch point: truncate and push new
            self.dirs.truncate(self.current_index + 1);
            self.dirs.push(new_dir);
            self.current_index = self.dirs.len() - 1;
        }
    }

    /// Get the display index for the current position
    /// This is used to set the cursor when opening history view
    pub fn current_display_index(&self) -> usize {
        // current_index maps to display as:
        // current_index = len - 1 -> display[0]
        // current_index = len - 2 -> display[1]
        // ...
        // current_index = 0 -> display[len - 1]
        //
        // So: display_index = len - 1 - current_index
        self.dirs.len() - 1 - self.current_index
    }

    /// Get the underlying history vec (for debugging/testing)
    #[cfg(test)]
    pub fn dirs(&self) -> &[PathBuf] {
        &self.dirs
    }

    /// Get current index (for debugging/testing)
    #[cfg(test)]
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Go back in history (like browser back button)
    /// Returns Some(PathBuf) if we can go back, None if already at oldest
    pub fn go_back(&mut self) -> Option<PathBuf> {
        if self.current_index > 0 {
            self.current_index -= 1;
            Some(self.dirs[self.current_index].clone())
        } else {
            None
        }
    }

    /// Go forward in history (like browser forward button)
    /// Returns Some(PathBuf) if we can go forward, None if already at newest
    pub fn go_forward(&mut self) -> Option<PathBuf> {
        if self.current_index + 1 < self.dirs.len() {
            self.current_index += 1;
            Some(self.dirs[self.current_index].clone())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn test_new_history_starts_with_initial_dir() {
        let history = History::new(path("/A"));
        assert_eq!(history.dirs(), &[path("/A")]);
        assert_eq!(history.current_index(), 0);
    }

    #[test]
    fn test_navigate_to_new_directories() {
        let mut history = History::new(path("/A"));

        // Navigate A -> B
        history.navigate_to(path("/B"));
        assert_eq!(history.dirs(), &[path("/A"), path("/B")]);
        assert_eq!(history.current_index(), 1);

        // Navigate B -> C
        history.navigate_to(path("/C"));
        assert_eq!(history.dirs(), &[path("/A"), path("/B"), path("/C")]);
        assert_eq!(history.current_index(), 2);
    }

    #[test]
    fn test_display_order() {
        let mut history = History::new(path("/A"));
        history.navigate_to(path("/B"));
        history.navigate_to(path("/C"));

        // Display should show [C, B, A] (most recent first)
        let display = history.items_for_display();
        assert_eq!(
            display,
            vec![path("/C"), path("/B"), path("/A")],
            "Display should show most recent first"
        );
    }

    #[test]
    fn test_navigate_backward_in_history() {
        let mut history = History::new(path("/A"));
        history.navigate_to(path("/B"));
        history.navigate_to(path("/C"));

        // Current: C (index 2), Display: [C, B, A]
        assert_eq!(history.current_index(), 2);

        // Select display[1] (B)
        let result = history.navigate_to_display_index(1);
        assert_eq!(result, Some(path("/B")));

        // History vec should be unchanged
        assert_eq!(
            history.dirs(),
            &[path("/A"), path("/B"), path("/C")],
            "History should not change when navigating backward"
        );

        // current_index should be 1
        assert_eq!(history.current_index(), 1);
    }

    #[test]
    fn test_navigate_to_oldest() {
        let mut history = History::new(path("/A"));
        history.navigate_to(path("/B"));
        history.navigate_to(path("/C"));

        // Select display[2] (A, the oldest)
        let result = history.navigate_to_display_index(2);
        assert_eq!(result, Some(path("/A")));

        assert_eq!(
            history.dirs(),
            &[path("/A"), path("/B"), path("/C")],
            "History should not change"
        );
        assert_eq!(history.current_index(), 0);
    }

    #[test]
    fn test_bug_scenario_history_disappears() {
        // User bug: A -> B -> C, select B in history, then navigate to D
        // C should NOT disappear

        let mut history = History::new(path("/A"));
        history.navigate_to(path("/B"));
        history.navigate_to(path("/C"));

        // Display: [C, B, A]
        assert_eq!(
            history.items_for_display(),
            vec![path("/C"), path("/B"), path("/A")]
        );

        // Select B (display[1])
        let _ = history.navigate_to_display_index(1);
        assert_eq!(history.current_index(), 1);

        // History should still be [A, B, C]
        assert_eq!(history.dirs(), &[path("/A"), path("/B"), path("/C")]);

        // Now navigate to D (new directory)
        history.navigate_to(path("/D"));

        // Should truncate at current_index (1), keeping [A, B], then push D
        assert_eq!(
            history.dirs(),
            &[path("/A"), path("/B"), path("/D")],
            "Should branch from B, keeping A and B but losing C"
        );
        assert_eq!(history.current_index(), 2);
    }

    #[test]
    fn test_navigate_forward_after_backward() {
        let mut history = History::new(path("/A"));
        history.navigate_to(path("/B"));
        history.navigate_to(path("/C"));

        // Go back to B
        let _ = history.navigate_to_display_index(1);
        assert_eq!(history.current_index(), 1);

        // Navigate to C (which is the next item in history)
        history.navigate_to(path("/C"));

        // Should just increment current_index, not add duplicate
        assert_eq!(
            history.dirs(),
            &[path("/A"), path("/B"), path("/C")],
            "Should not duplicate C"
        );
        assert_eq!(history.current_index(), 2);
    }

    #[test]
    fn test_current_display_index() {
        let mut history = History::new(path("/A"));
        history.navigate_to(path("/B"));
        history.navigate_to(path("/C"));

        // At C (index 2), display_index should be 0
        assert_eq!(history.current_display_index(), 0);

        // Navigate to B (index 1)
        let _ = history.navigate_to_display_index(1);
        assert_eq!(history.current_display_index(), 1);

        // Navigate to A (index 0)
        let _ = history.navigate_to_display_index(2);
        assert_eq!(history.current_display_index(), 2);
    }
}
