//! Keybinding registry - THE SINGLE SOURCE OF TRUTH for input.
//!
//! Every key and mouse binding lives in [`KEYMAP`] exactly once, together with
//! the help text shown to the user. `input.rs` turns raw crossterm events into
//! an [`Action`] through [`lookup`] and then does the work; the help screen
//! renders the same table. Nothing is kept in sync by hand.
//!
//! **A binding that isn't documented is unrepresentable:**
//!
//! * [`Binding::new`] is the only way to build a binding, and it asserts that the
//!   help text and trigger list are non-empty. `KEYMAP` is a `static`, so those
//!   asserts run at compile time - an undocumented binding fails to build.
//! * `input.rs` never inspects a raw key. The only path from a keypress to
//!   behaviour is [`lookup`], which can only return actions that are in `KEYMAP`.
//!   So a binding cannot be implemented without a row here.
//! * The dispatch `match` in `input.rs` is exhaustive over [`Action`], so a new
//!   action must be handled, and `test_every_action_is_bound` checks the other
//!   direction: every action has a row.
//! * Every row names a [`Section`], and the help screen renders all of
//!   [`Section::ALL`], so every row reaches the screen.
//!
//! Same idea as `feature_defs/registry.rs`: one list, everything else derived.

use crossterm::event::{KeyCode, KeyModifiers, MouseEventKind};
use strum_macros::EnumIter;

/// What a binding does. `input.rs` matches exhaustively over this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum Action {
    // Search
    AppendToQuery,
    DeleteFromQuery,
    ClearQuery,

    // Moving around
    MoveUp,
    MoveDown,
    Activate,
    ParentDir,
    HistoryBack,
    HistoryForward,
    ToggleHistoryMode,

    // Leaving for a shell
    VisitCurrentDir,
    VisitSelectedDir,

    // Filters
    CycleFilterForward,
    CycleFilterBackward,
    ToggleFilterPicker,
    SetFilterNone,
    SetFilterCwd,
    SetFilterDirectCwd,
    SetFilterDirs,
    SetFilterFiles,

    // Panes
    CycleDebugPane,
    ScrollPreviewUp,
    ScrollPreviewDown,

    // Help and exit
    ToggleHelp,
    Escape,
    Quit,
}

/// Which UI mode a binding applies to.
///
/// Bindings are looked up in the active context first, then in `Global`, so the
/// filter picker can claim plain letters without losing the global bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Context {
    Global,
    FilterPicker,
}

/// A key combination. Shift is not tracked: it is already baked into the
/// character crossterm reports (and into `BackTab`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chord {
    pub code: KeyCode,
    pub ctrl: bool,
    pub alt: bool,
}

/// What fires a binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    Key(Chord),
    /// Any printable character not claimed by a `Key` trigger. Checked last.
    AnyChar,
    Mouse(MouseEventKind),
}

/// Where a binding appears on the help screen. Order here is display order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Search,
    Moving,
    Shell,
    Filters,
    Panes,
    HelpAndExit,
}

impl Section {
    /// Every section, in the order the help screen shows them.
    ///
    /// The help screen renders this list, so a section missing from it would
    /// hide its bindings. `test_every_section_is_displayed` guards that.
    pub const ALL: &'static [Section] = &[
        Section::Search,
        Section::Moving,
        Section::Shell,
        Section::Filters,
        Section::Panes,
        Section::HelpAndExit,
    ];

    pub fn title(&self) -> &'static str {
        match self {
            Section::Search => "Search",
            Section::Moving => "Moving around",
            Section::Shell => "Leaving for a shell",
            Section::Filters => "Filters",
            Section::Panes => "Panes",
            Section::HelpAndExit => "Help and exit",
        }
    }
}

/// One row of the keymap: what fires it, what it does, and how to explain it.
///
/// Fields are private so [`Binding::new`] is the only constructor; it rejects a
/// binding with no help text or no trigger.
pub struct Binding {
    action: Action,
    triggers: &'static [Trigger],
    context: Context,
    section: Section,
    description: &'static str,
}

impl Binding {
    /// Declare a binding.
    ///
    /// `description` is the help text the user reads: lowercase, imperative, no
    /// trailing period. The asserts are evaluated at compile time because every
    /// caller is inside the `KEYMAP` static, so an undocumented or untriggerable
    /// binding is a build failure rather than a bug.
    const fn new(
        action: Action,
        triggers: &'static [Trigger],
        context: Context,
        section: Section,
        description: &'static str,
    ) -> Self {
        assert!(
            !description.is_empty(),
            "every keybinding needs help text: it is what the user reads on the help screen"
        );
        assert!(
            !triggers.is_empty(),
            "a binding with no trigger can never fire, but would still be listed as if it could"
        );

        Self {
            action,
            triggers,
            context,
            section,
            description,
        }
    }

    pub fn description(&self) -> &'static str {
        self.description
    }

    /// The key column for this binding on the help screen, e.g. `Up / Ctrl-P`.
    pub fn keys(&self) -> String {
        self.triggers
            .iter()
            .map(describe_trigger)
            .collect::<Vec<_>>()
            .join(" / ")
    }

    fn has_trigger(&self, trigger: &Trigger) -> bool {
        self.triggers.contains(trigger)
    }

    fn matches_key(&self, code: KeyCode, modifiers: KeyModifiers) -> bool {
        self.triggers.iter().any(|trigger| match trigger {
            Trigger::Key(chord) => chord_matches(chord, code, modifiers),
            _ => false,
        })
    }
}

const fn key(code: KeyCode) -> Trigger {
    Trigger::Key(Chord {
        code,
        ctrl: false,
        alt: false,
    })
}

const fn ctrl_key(code: KeyCode) -> Trigger {
    Trigger::Key(Chord {
        code,
        ctrl: true,
        alt: false,
    })
}

const fn alt_key(code: KeyCode) -> Trigger {
    Trigger::Key(Chord {
        code,
        ctrl: false,
        alt: true,
    })
}

const fn chr(c: char) -> Trigger {
    key(KeyCode::Char(c))
}

const fn ctrl(c: char) -> Trigger {
    ctrl_key(KeyCode::Char(c))
}

/// Every binding psychic responds to.
pub static KEYMAP: &[Binding] = &[
    // ---- Search ----
    Binding::new(
        Action::AppendToQuery,
        &[Trigger::AnyChar],
        Context::Global,
        Section::Search,
        "add to the search query",
    ),
    Binding::new(
        Action::DeleteFromQuery,
        &[key(KeyCode::Backspace)],
        Context::Global,
        Section::Search,
        "delete the last character",
    ),
    Binding::new(
        Action::ClearQuery,
        &[ctrl('u')],
        Context::Global,
        Section::Search,
        "clear the query",
    ),
    // ---- Moving around ----
    Binding::new(
        Action::MoveUp,
        &[key(KeyCode::Up), ctrl('p')],
        Context::Global,
        Section::Moving,
        "move selection up",
    ),
    Binding::new(
        Action::MoveDown,
        &[key(KeyCode::Down), ctrl('n')],
        Context::Global,
        Section::Moving,
        "move selection down",
    ),
    Binding::new(
        Action::Activate,
        &[key(KeyCode::Enter)],
        Context::Global,
        Section::Moving,
        "open file in your editor, or enter dir",
    ),
    Binding::new(
        Action::ParentDir,
        &[alt_key(KeyCode::Up)],
        Context::Global,
        Section::Moving,
        "go to the parent directory",
    ),
    Binding::new(
        Action::HistoryBack,
        &[key(KeyCode::Left)],
        Context::Global,
        Section::Moving,
        "back to the previous directory",
    ),
    Binding::new(
        Action::HistoryForward,
        &[key(KeyCode::Right)],
        Context::Global,
        Section::Moving,
        "forward to the next directory",
    ),
    Binding::new(
        Action::ToggleHistoryMode,
        &[ctrl('h')],
        Context::Global,
        Section::Moving,
        "browse every directory visited",
    ),
    // ---- Leaving for a shell ----
    Binding::new(
        Action::VisitCurrentDir,
        &[ctrl('j')],
        Context::Global,
        Section::Shell,
        "visit the current directory",
    ),
    Binding::new(
        Action::VisitSelectedDir,
        &[ctrl_key(KeyCode::Enter)],
        Context::Global,
        Section::Shell,
        "visit the selected directory",
    ),
    // ---- Filters ----
    Binding::new(
        Action::CycleFilterForward,
        &[key(KeyCode::Tab)],
        Context::Global,
        Section::Filters,
        "next filter",
    ),
    Binding::new(
        Action::CycleFilterBackward,
        &[key(KeyCode::BackTab)],
        Context::Global,
        Section::Filters,
        "previous filter",
    ),
    Binding::new(
        Action::ToggleFilterPicker,
        &[ctrl('f')],
        Context::Global,
        Section::Filters,
        "open the filter picker",
    ),
    Binding::new(
        Action::SetFilterNone,
        &[chr('0')],
        Context::FilterPicker,
        Section::Filters,
        "in picker: no filter",
    ),
    Binding::new(
        Action::SetFilterCwd,
        &[chr('c')],
        Context::FilterPicker,
        Section::Filters,
        "in picker: under cwd, recursive",
    ),
    Binding::new(
        Action::SetFilterDirectCwd,
        &[chr('i')],
        Context::FilterPicker,
        Section::Filters,
        "in picker: direct children of cwd",
    ),
    Binding::new(
        Action::SetFilterDirs,
        &[chr('d')],
        Context::FilterPicker,
        Section::Filters,
        "in picker: directories only",
    ),
    Binding::new(
        Action::SetFilterFiles,
        &[chr('f')],
        Context::FilterPicker,
        Section::Filters,
        "in picker: files only",
    ),
    // ---- Panes ----
    Binding::new(
        Action::CycleDebugPane,
        &[ctrl('o')],
        Context::Global,
        Section::Panes,
        "cycle debug pane: off, small, large",
    ),
    Binding::new(
        Action::ScrollPreviewDown,
        &[Trigger::Mouse(MouseEventKind::ScrollDown)],
        Context::Global,
        Section::Panes,
        "scroll the preview down",
    ),
    Binding::new(
        Action::ScrollPreviewUp,
        &[Trigger::Mouse(MouseEventKind::ScrollUp)],
        Context::Global,
        Section::Panes,
        "scroll the preview up",
    ),
    // ---- Help and exit ----
    Binding::new(
        Action::ToggleHelp,
        &[ctrl('g'), key(KeyCode::F(1))],
        Context::Global,
        Section::HelpAndExit,
        "show this help",
    ),
    Binding::new(
        Action::Escape,
        &[key(KeyCode::Esc)],
        Context::Global,
        Section::HelpAndExit,
        "close popup or history, else quit",
    ),
    Binding::new(
        Action::Quit,
        &[ctrl('c'), ctrl('d')],
        Context::Global,
        Section::HelpAndExit,
        "quit",
    ),
];

/// Does this key event match `chord`?
///
/// Shift is deliberately ignored: crossterm reports it both as a modifier and in
/// the character itself, so requiring an exact modifier set would break
/// `Shift-Tab` and capital letters.
fn chord_matches(chord: &Chord, code: KeyCode, modifiers: KeyModifiers) -> bool {
    chord.code == code
        && chord.ctrl == modifiers.contains(KeyModifiers::CONTROL)
        && chord.alt == modifiers.contains(KeyModifiers::ALT)
}

/// Is this a plain printable character (no ctrl or alt)?
fn is_plain_char(code: KeyCode, modifiers: KeyModifiers) -> bool {
    matches!(code, KeyCode::Char(_))
        && !modifiers.contains(KeyModifiers::CONTROL)
        && !modifiers.contains(KeyModifiers::ALT)
}

/// Resolve a key event to an action.
///
/// Bindings in `context` win over global ones, and explicit chords win over the
/// catch-all [`Trigger::AnyChar`], so the filter picker can bind plain letters
/// without swallowing `Ctrl-C` or making typing impossible everywhere else.
pub fn lookup(code: KeyCode, modifiers: KeyModifiers, context: Context) -> Option<Action> {
    if context != Context::Global
        && let Some(binding) = KEYMAP
            .iter()
            .find(|b| b.context == context && b.matches_key(code, modifiers))
    {
        return Some(binding.action);
    }

    if let Some(binding) = KEYMAP
        .iter()
        .find(|b| b.context == Context::Global && b.matches_key(code, modifiers))
    {
        return Some(binding.action);
    }

    if is_plain_char(code, modifiers) {
        return KEYMAP
            .iter()
            .find(|b| b.context == Context::Global && b.has_trigger(&Trigger::AnyChar))
            .map(|b| b.action);
    }

    None
}

/// Resolve a mouse event to an action.
pub fn lookup_mouse(kind: MouseEventKind) -> Option<Action> {
    KEYMAP
        .iter()
        .find(|b| b.has_trigger(&Trigger::Mouse(kind)))
        .map(|b| b.action)
}

/// Human-readable name for a key combination, e.g. `Ctrl-J`, `Shift-Tab`, `Up`.
fn describe_chord(chord: &Chord) -> String {
    let base = match chord.code {
        // BackTab *is* Shift-Tab; that is what the user pressed and expects to read.
        KeyCode::BackTab => "Shift-Tab".to_string(),
        KeyCode::Char(' ') => "Space".to_string(),
        KeyCode::Char(c) if chord.ctrl || chord.alt => c.to_ascii_uppercase().to_string(),
        KeyCode::Char(c) => c.to_string(),
        KeyCode::F(n) => format!("F{}", n),
        other => format!("{:?}", other),
    };

    match (chord.ctrl, chord.alt) {
        (true, true) => format!("Ctrl-Alt-{}", base),
        (true, false) => format!("Ctrl-{}", base),
        (false, true) => format!("Alt-{}", base),
        (false, false) => base,
    }
}

/// Human-readable name for a trigger.
pub fn describe_trigger(trigger: &Trigger) -> String {
    match trigger {
        Trigger::Key(chord) => describe_chord(chord),
        Trigger::AnyChar => "any character".to_string(),
        Trigger::Mouse(MouseEventKind::ScrollUp) => "Wheel up".to_string(),
        Trigger::Mouse(MouseEventKind::ScrollDown) => "Wheel down".to_string(),
        Trigger::Mouse(kind) => format!("{:?}", kind),
    }
}

/// How to describe triggering `action` in a one-line UI hint.
///
/// This is the first trigger declared for the action, so declaration order is
/// preference order: put the key you want advertised first.
///
/// Returns `None` only if nothing in the keymap produces that action.
pub fn primary_trigger(action: Action) -> Option<String> {
    KEYMAP
        .iter()
        .find(|b| b.action == action)?
        .triggers
        .first()
        .map(describe_trigger)
}

/// All bindings in a section, in declaration order.
pub fn bindings_in(section: Section) -> impl Iterator<Item = &'static Binding> {
    KEYMAP.iter().filter(move |b| b.section == section)
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::IntoEnumIterator;

    #[test]
    fn test_lookup_ctrl_binding() {
        assert_eq!(
            lookup(KeyCode::Char('u'), KeyModifiers::CONTROL, Context::Global),
            Some(Action::ClearQuery)
        );
    }

    #[test]
    fn test_plain_char_falls_through_to_query() {
        assert_eq!(
            lookup(KeyCode::Char('u'), KeyModifiers::NONE, Context::Global),
            Some(Action::AppendToQuery),
            "Without ctrl, 'u' is just text"
        );
    }

    #[test]
    fn test_capital_letters_reach_the_query() {
        assert_eq!(
            lookup(KeyCode::Char('U'), KeyModifiers::SHIFT, Context::Global),
            Some(Action::AppendToQuery),
            "Shift must not stop a character from being typed"
        );
    }

    #[test]
    fn test_filter_picker_claims_plain_letters() {
        assert_eq!(
            lookup(
                KeyCode::Char('d'),
                KeyModifiers::NONE,
                Context::FilterPicker
            ),
            Some(Action::SetFilterDirs)
        );
        assert_eq!(
            lookup(KeyCode::Char('d'), KeyModifiers::NONE, Context::Global),
            Some(Action::AppendToQuery),
            "Outside the picker the same key types instead"
        );
    }

    #[test]
    fn test_ctrl_beats_filter_picker() {
        assert_eq!(
            lookup(
                KeyCode::Char('c'),
                KeyModifiers::CONTROL,
                Context::FilterPicker
            ),
            Some(Action::Quit),
            "Ctrl-C must quit even while the picker claims plain 'c'"
        );
        assert_eq!(
            lookup(
                KeyCode::Char('c'),
                KeyModifiers::NONE,
                Context::FilterPicker
            ),
            Some(Action::SetFilterCwd)
        );
    }

    #[test]
    fn test_alt_up_is_not_up() {
        assert_eq!(
            lookup(KeyCode::Up, KeyModifiers::ALT, Context::Global),
            Some(Action::ParentDir)
        );
        assert_eq!(
            lookup(KeyCode::Up, KeyModifiers::NONE, Context::Global),
            Some(Action::MoveUp)
        );
    }

    #[test]
    fn test_ctrl_enter_is_not_enter() {
        assert_eq!(
            lookup(KeyCode::Enter, KeyModifiers::CONTROL, Context::Global),
            Some(Action::VisitSelectedDir)
        );
        assert_eq!(
            lookup(KeyCode::Enter, KeyModifiers::NONE, Context::Global),
            Some(Action::Activate)
        );
    }

    #[test]
    fn test_unbound_key_is_ignored() {
        assert_eq!(
            lookup(KeyCode::Insert, KeyModifiers::NONE, Context::Global),
            None
        );
    }

    #[test]
    fn test_lookup_mouse() {
        assert_eq!(
            lookup_mouse(MouseEventKind::ScrollUp),
            Some(Action::ScrollPreviewUp)
        );
        assert_eq!(lookup_mouse(MouseEventKind::Moved), None);
    }

    /// The help screen's key column is generated from the triggers, so this is
    /// also a test of what the user reads.
    #[test]
    fn test_keys_are_described_the_way_users_say_them() {
        let keys = |action: Action| KEYMAP.iter().find(|b| b.action == action).unwrap().keys();

        assert_eq!(keys(Action::MoveUp), "Up / Ctrl-P");
        assert_eq!(keys(Action::ParentDir), "Alt-Up");
        assert_eq!(keys(Action::VisitSelectedDir), "Ctrl-Enter");
        assert_eq!(keys(Action::CycleFilterBackward), "Shift-Tab");
        assert_eq!(keys(Action::ToggleHelp), "Ctrl-G / F1");
        assert_eq!(keys(Action::SetFilterNone), "0");
        assert_eq!(keys(Action::AppendToQuery), "any character");
        assert_eq!(keys(Action::ScrollPreviewUp), "Wheel up");
        assert_eq!(keys(Action::DeleteFromQuery), "Backspace");
    }

    #[test]
    fn test_primary_trigger_is_the_first_declared() {
        assert_eq!(
            primary_trigger(Action::ToggleHelp),
            Some("Ctrl-G".to_string()),
            "The hint on the main screen must not advertise F1: on macOS it is a \
             brightness key unless the user has changed that setting"
        );
        assert_eq!(primary_trigger(Action::Quit), Some("Ctrl-C".to_string()));
    }

    /// The half of the round trip the compiler cannot check: `input.rs` must
    /// handle every action, but nothing stops an action from having no row here,
    /// which would leave it unreachable and undocumented.
    #[test]
    fn test_every_action_is_bound() {
        for action in Action::iter() {
            assert!(
                KEYMAP.iter().any(|b| b.action == action),
                "Action {:?} has no row in KEYMAP, so no key can trigger it and it \
                 never appears on the help screen",
                action
            );
        }
    }

    #[test]
    fn test_every_action_is_bound_once() {
        for (i, binding) in KEYMAP.iter().enumerate() {
            for other in KEYMAP.iter().skip(i + 1) {
                assert_ne!(
                    binding.action, other.action,
                    "Action {:?} has two rows; put all of its triggers in one row so \
                     the help screen lists it once",
                    binding.action
                );
            }
        }
    }

    #[test]
    fn test_no_shadowed_triggers() {
        for (i, binding) in KEYMAP.iter().enumerate() {
            for other in KEYMAP.iter().skip(i + 1) {
                if binding.context != other.context {
                    continue;
                }
                for trigger in binding.triggers {
                    assert!(
                        !other.has_trigger(trigger),
                        "{} is bound to both {:?} and {:?} in the same context",
                        describe_trigger(trigger),
                        binding.action,
                        other.action
                    );
                }
            }
        }
    }

    #[test]
    fn test_every_section_is_displayed() {
        for binding in KEYMAP {
            assert!(
                Section::ALL.contains(&binding.section),
                "Section {:?} is missing from Section::ALL, so {:?} would never appear \
                 on the help screen",
                binding.section,
                binding.action
            );
        }

        for section in Section::ALL {
            assert!(
                bindings_in(*section).next().is_some(),
                "Section {:?} is displayed but has no bindings",
                section
            );
        }
    }
}
