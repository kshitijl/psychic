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

    /// Toggle history mode on/off
    pub fn toggle_history_mode(&mut self) {
        self.history_mode = !self.history_mode;
    }

    /// Enter history mode
    pub fn enter_history_mode(&mut self) {
        self.history_mode = true;
    }

    /// Exit history mode
    pub fn exit_history_mode(&mut self) {
        self.history_mode = false;
    }

    /// Toggle filter picker on/off
    pub fn toggle_filter_picker(&mut self) {
        self.filter_picker_visible = !self.filter_picker_visible;
    }

    /// Hide filter picker
    pub fn hide_filter_picker(&mut self) {
        self.filter_picker_visible = false;
    }

    /// Cycle debug pane mode: Small -> Expanded -> Hidden -> Small
    pub fn cycle_debug_pane_mode(&mut self) {
        self.debug_pane_mode = match self.debug_pane_mode {
            DebugPaneMode::Small => DebugPaneMode::Expanded,
            DebugPaneMode::Expanded => DebugPaneMode::Hidden,
            DebugPaneMode::Hidden => DebugPaneMode::Small,
        };
    }

    /// Get eza command flags based on preview pane width
    ///
    /// # Arguments
    /// * `width` - Available width for the preview pane
    ///
    /// # Returns
    /// The flags to pass to eza (everything after "eza -al")
    pub fn get_eza_flags(&self, width: u16) -> &'static str {
        // If width is small, omit user and permissions to save space
        if width < 80 {
            " --no-user --no-permissions"
        } else {
            ""
        }
    }

    /// Should the debug pane be visible?
    pub fn is_debug_pane_visible(&self) -> bool {
        !matches!(self.debug_pane_mode, DebugPaneMode::Hidden)
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
        assert_eq!(state.is_debug_pane_visible(), false);
        assert_eq!(state.is_debug_pane_expanded(), false);
    }

    #[test]
    fn test_toggle_history_mode() {
        let mut state = UiState::new();

        state.toggle_history_mode();
        assert_eq!(state.history_mode, true);

        state.toggle_history_mode();
        assert_eq!(state.history_mode, false);
    }

    #[test]
    fn test_enter_exit_history_mode() {
        let mut state = UiState::new();

        state.enter_history_mode();
        assert_eq!(state.history_mode, true);

        state.enter_history_mode(); // Should stay true
        assert_eq!(state.history_mode, true);

        state.exit_history_mode();
        assert_eq!(state.history_mode, false);
    }

    #[test]
    fn test_toggle_filter_picker() {
        let mut state = UiState::new();

        state.toggle_filter_picker();
        assert_eq!(state.filter_picker_visible, true);

        state.toggle_filter_picker();
        assert_eq!(state.filter_picker_visible, false);
    }

    #[test]
    fn test_hide_filter_picker() {
        let mut state = UiState::new();

        state.toggle_filter_picker();
        assert_eq!(state.filter_picker_visible, true);

        state.hide_filter_picker();
        assert_eq!(state.filter_picker_visible, false);

        state.hide_filter_picker(); // Should stay false
        assert_eq!(state.filter_picker_visible, false);
    }

    #[test]
    fn test_cycle_debug_pane_mode() {
        let mut state = UiState::new();

        // Initial: Hidden
        assert_eq!(state.debug_pane_mode, DebugPaneMode::Hidden);
        assert_eq!(state.is_debug_pane_visible(), false);
        assert_eq!(state.is_debug_pane_expanded(), false);

        // Cycle to Small
        state.cycle_debug_pane_mode();
        assert_eq!(state.debug_pane_mode, DebugPaneMode::Small);
        assert_eq!(state.is_debug_pane_visible(), true);
        assert_eq!(state.is_debug_pane_expanded(), false);

        // Cycle to Expanded
        state.cycle_debug_pane_mode();
        assert_eq!(state.debug_pane_mode, DebugPaneMode::Expanded);
        assert_eq!(state.is_debug_pane_visible(), true);
        assert_eq!(state.is_debug_pane_expanded(), true);

        // Cycle back to Hidden
        state.cycle_debug_pane_mode();
        assert_eq!(state.debug_pane_mode, DebugPaneMode::Hidden);
        assert_eq!(state.is_debug_pane_visible(), false);
        assert_eq!(state.is_debug_pane_expanded(), false);
    }

    #[test]
    fn test_get_eza_flags_wide_screen() {
        let state = UiState::new();

        // Wide screens get full eza output
        assert_eq!(state.get_eza_flags(80), "", "80 width should use full eza");
        assert_eq!(
            state.get_eza_flags(100),
            "",
            "100 width should use full eza"
        );
        assert_eq!(
            state.get_eza_flags(120),
            "",
            "120 width should use full eza"
        );
    }

    #[test]
    fn test_get_eza_flags_narrow_screen() {
        let state = UiState::new();

        // Narrow screens get compact eza output
        assert_eq!(
            state.get_eza_flags(79),
            " --no-user --no-permissions",
            "79 width should use compact eza"
        );
        assert_eq!(
            state.get_eza_flags(60),
            " --no-user --no-permissions",
            "60 width should use compact eza"
        );
        assert_eq!(
            state.get_eza_flags(40),
            " --no-user --no-permissions",
            "40 width should use compact eza"
        );
    }

    #[test]
    fn test_combined_state_transitions() {
        let mut state = UiState::new();

        // User opens history mode
        state.enter_history_mode();
        assert_eq!(state.history_mode, true);

        // User toggles filter picker while in history
        state.toggle_filter_picker();
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
        state.exit_history_mode();
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
