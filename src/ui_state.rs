//! UI state machine module
//!
//! This module manages all UI state transitions as a pure state machine.
//! No IO, just state transitions and queries that can be tested with expect tests.

/// Debug pane display mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugPaneMode {
    Hidden,
    Small,
    Expanded,
}

/// UI state machine
#[derive(Debug, Clone)]
pub struct UiState {
    /// Are we in history navigation mode?
    pub history_mode: bool,
    /// Is the filter picker visible?
    pub filter_picker_visible: bool,
    /// Debug pane mode
    pub debug_pane_mode: DebugPaneMode,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            history_mode: false,
            filter_picker_visible: false,
            debug_pane_mode: DebugPaneMode::Hidden,
        }
    }

    /// Cycle debug pane mode: Small -> Expanded -> Hidden -> Small
    pub fn cycle_debug_pane_mode(&mut self) {
        self.debug_pane_mode = match self.debug_pane_mode {
            DebugPaneMode::Small => DebugPaneMode::Expanded,
            DebugPaneMode::Expanded => DebugPaneMode::Hidden,
            DebugPaneMode::Hidden => DebugPaneMode::Small,
        };
    }

    /// Is the debug pane expanded (taking more space)?
    pub fn is_debug_pane_expanded(&self) -> bool {
        matches!(self.debug_pane_mode, DebugPaneMode::Expanded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let state = UiState::new();
        assert_eq!(state.history_mode, false);
        assert_eq!(state.filter_picker_visible, false);
        assert_eq!(state.debug_pane_mode, DebugPaneMode::Hidden);
        assert_eq!(state.is_debug_pane_expanded(), false);
    }

    #[test]
    fn test_history_mode_field() {
        let mut state = UiState::new();

        state.history_mode = true;
        assert_eq!(state.history_mode, true);

        state.history_mode = false;
        assert_eq!(state.history_mode, false);
    }

    #[test]
    fn test_filter_picker_field() {
        let mut state = UiState::new();

        state.filter_picker_visible = true;
        assert_eq!(state.filter_picker_visible, true);

        state.filter_picker_visible = false;
        assert_eq!(state.filter_picker_visible, false);
    }

    #[test]
    fn test_cycle_debug_pane_mode() {
        let mut state = UiState::new();

        // Initial: Hidden
        assert_eq!(state.debug_pane_mode, DebugPaneMode::Hidden);
        assert_eq!(state.is_debug_pane_expanded(), false);

        // Cycle to Small
        state.cycle_debug_pane_mode();
        assert_eq!(state.debug_pane_mode, DebugPaneMode::Small);
        assert_eq!(state.is_debug_pane_expanded(), false);

        // Cycle to Expanded
        state.cycle_debug_pane_mode();
        assert_eq!(state.debug_pane_mode, DebugPaneMode::Expanded);
        assert_eq!(state.is_debug_pane_expanded(), true);

        // Cycle back to Hidden
        state.cycle_debug_pane_mode();
        assert_eq!(state.debug_pane_mode, DebugPaneMode::Hidden);
        assert_eq!(state.is_debug_pane_expanded(), false);
    }

    #[test]
    fn test_combined_state_transitions() {
        let mut state = UiState::new();

        // User opens history mode
        state.history_mode = true;
        assert_eq!(state.history_mode, true);

        // User toggles filter picker while in history
        state.filter_picker_visible = true;
        assert_eq!(state.filter_picker_visible, true);
        assert_eq!(
            state.history_mode, true,
            "History mode should remain active"
        );

        // User cycles debug pane (Hidden -> Small)
        state.cycle_debug_pane_mode();
        assert_eq!(state.debug_pane_mode, DebugPaneMode::Small);
        assert_eq!(
            state.history_mode, true,
            "History mode should remain active"
        );
        assert_eq!(
            state.filter_picker_visible, true,
            "Filter picker should remain visible"
        );

        // User exits history mode
        state.history_mode = false;
        assert_eq!(state.history_mode, false);
        assert_eq!(
            state.filter_picker_visible, true,
            "Filter picker should remain visible"
        );
        assert_eq!(
            state.debug_pane_mode,
            DebugPaneMode::Small,
            "Debug pane should remain in small mode"
        );
    }
}
