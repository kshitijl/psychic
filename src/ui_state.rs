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
    /// Is the help screen covering everything else?
    pub help_visible: bool,
    /// First visible line of the help screen, for terminals too short to show it all
    pub help_scroll: u16,
    /// Largest useful `help_scroll`, measured while rendering the last frame
    pub help_scroll_max: u16,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            history_mode: false,
            filter_picker_visible: false,
            debug_pane_mode: DebugPaneMode::Hidden,
            help_visible: false,
            help_scroll: 0,
            help_scroll_max: 0,
        }
    }

    /// Show or hide the help screen. Always reopens at the top.
    pub fn toggle_help(&mut self) {
        self.help_visible = !self.help_visible;
        self.help_scroll = 0;
    }

    pub fn hide_help(&mut self) {
        self.help_visible = false;
        self.help_scroll = 0;
    }

    /// Scroll the help screen, clamped to the content measured while rendering.
    pub fn scroll_help(&mut self, delta: i16) {
        self.help_scroll = self
            .help_scroll
            .saturating_add_signed(delta)
            .min(self.help_scroll_max);
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
    fn test_toggle_help() {
        let mut state = UiState::new();
        assert_eq!(state.help_visible, false);

        state.toggle_help();
        assert_eq!(state.help_visible, true);

        state.toggle_help();
        assert_eq!(state.help_visible, false);
    }

    #[test]
    fn test_help_reopens_at_the_top() {
        let mut state = UiState::new();
        state.help_scroll_max = 10;

        state.toggle_help();
        state.scroll_help(4);
        assert_eq!(state.help_scroll, 4);

        state.toggle_help(); // close
        state.toggle_help(); // reopen
        assert_eq!(state.help_scroll, 0, "Reopening starts from the top again");
    }

    #[test]
    fn test_scroll_help_is_clamped_both_ways() {
        let mut state = UiState::new();
        state.help_scroll_max = 3;

        state.scroll_help(-1);
        assert_eq!(state.help_scroll, 0, "Cannot scroll above the first line");

        state.scroll_help(10);
        assert_eq!(state.help_scroll, 3, "Cannot scroll past the last line");

        state.scroll_help(-1);
        assert_eq!(state.help_scroll, 2);
    }

    #[test]
    fn test_scroll_help_with_nothing_to_scroll() {
        let mut state = UiState::new();
        // help_scroll_max stays 0 when the whole screen fits.
        state.scroll_help(5);
        assert_eq!(state.help_scroll, 0);
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
