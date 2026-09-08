//! Input handling module - keyboard and mouse event processing
//!
//! This module provides a clean interface for handling user input events:
//! - Keyboard shortcuts (Ctrl-C, Ctrl-H, etc.)
//! - Mouse scrolling
//! - Text input for search
//! - Directory navigation
//! - Terminal suspension for editor/shell
//!
//! Deep implementation hiding complexity of terminal management, event dispatching,
//! and state updates behind a simple `handle_input` function.
//!
//! Which key does what is not decided here. Events are resolved to a
//! `keymap::Action` by the registry in `keymap.rs`, and this module only says
//! what each action does. That keeps the keys, their help text, and their
//! behaviour in one place: see the module docs in `keymap.rs`.

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use crossterm::{
    cursor::Show,
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{Terminal, backend::CrosstermBackend};

use crate::app::App;
use crate::cli::{OnCwdVisitAction, OnDirClickAction};
use crate::db::{EventData, UserInteraction};
use crate::keymap::{self, Action, Context};
use crate::search_worker::{FilterType, UpdateQueryRequest, WorkerRequest};

/// Action to take after handling input
pub enum InputAction {
    /// Continue running the event loop
    Continue,
    /// Exit the application
    Exit,
    /// Print a path to stdout and exit (for shell integration)
    PrintAndExit(String),
}

/// Handle a single input event (keyboard or mouse)
///
/// Returns an InputAction indicating what the main loop should do next.
pub fn handle_input(
    app: &mut App,
    event: Event,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<InputAction> {
    match event {
        Event::Mouse(mouse_event) => {
            let Some(action) = keymap::lookup_mouse(mouse_event.kind) else {
                return Ok(InputAction::Continue);
            };

            // The help screen covers the preview, so the wheel scrolls what the
            // user can actually see.
            if app.ui_state.help_visible {
                match action {
                    Action::ScrollPreviewUp => app.ui_state.scroll_help(-3),
                    Action::ScrollPreviewDown => app.ui_state.scroll_help(3),
                    _ => {}
                }
                return Ok(InputAction::Continue);
            }

            dispatch(app, action, KeyCode::Null, terminal)
        }
        Event::Key(key) if key.kind == KeyEventKind::Press => {
            handle_key_press(app, key.code, key.modifiers, terminal)
        }
        _ => Ok(InputAction::Continue),
    }
}

/// Handle a key press event
fn handle_key_press(
    app: &mut App,
    code: KeyCode,
    modifiers: KeyModifiers,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<InputAction> {
    // A status message lasts until the user does something else. Cleared before
    // dispatch, so a handler below can set a fresh one for this keypress.
    app.status_message = None;

    // The help screen is a cheat sheet, not a mode: it scrolls, and anything
    // else dismisses it rather than acting on whatever was underneath.
    if app.ui_state.help_visible {
        return Ok(handle_help_key(app, code, modifiers));
    }

    let context = if app.ui_state.filter_picker_visible {
        Context::FilterPicker
    } else {
        Context::Global
    };

    let Some(action) = keymap::lookup(code, modifiers, context) else {
        return Ok(InputAction::Continue);
    };

    dispatch(app, action, code, terminal)
}

/// Do what an action says.
///
/// `code` is the key that produced the action, needed only by
/// `AppendToQuery`, which has to know which character was typed.
///
/// This match is exhaustive on purpose: adding an action to the keymap without
/// implementing it here does not compile.
fn dispatch(
    app: &mut App,
    action: Action,
    code: KeyCode,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<InputAction> {
    match action {
        Action::AppendToQuery => {
            if let KeyCode::Char(c) = code {
                handle_char_input(app, c);
            }
            Ok(InputAction::Continue)
        }
        Action::DeleteFromQuery => {
            handle_backspace(app);
            Ok(InputAction::Continue)
        }
        Action::ClearQuery => {
            handle_ctrl_u(app);
            Ok(InputAction::Continue)
        }
        Action::MoveUp => {
            handle_navigation(app, -1);
            Ok(InputAction::Continue)
        }
        Action::MoveDown => {
            handle_navigation(app, 1);
            Ok(InputAction::Continue)
        }
        Action::Activate => handle_enter(app, terminal),
        Action::ParentDir => handle_parent_dir(app),
        Action::HistoryBack => handle_history_back(app),
        Action::HistoryForward => handle_history_forward(app),
        Action::ToggleHistoryMode => {
            handle_ctrl_h(app);
            Ok(InputAction::Continue)
        }
        Action::VisitCurrentDir => handle_ctrl_j(app, terminal),
        Action::VisitSelectedDir => handle_ctrl_enter(app, terminal),
        Action::CycleFilterForward => {
            cycle_filter(app, true);
            Ok(InputAction::Continue)
        }
        Action::CycleFilterBackward => {
            cycle_filter(app, false);
            Ok(InputAction::Continue)
        }
        Action::ToggleFilterPicker => {
            app.ui_state.filter_picker_visible = !app.ui_state.filter_picker_visible;
            Ok(InputAction::Continue)
        }
        Action::SetFilterNone => {
            set_filter(app, FilterType::None);
            Ok(InputAction::Continue)
        }
        Action::SetFilterCwd => {
            set_filter(app, FilterType::OnlyCwd);
            Ok(InputAction::Continue)
        }
        Action::SetFilterDirectCwd => {
            set_filter(app, FilterType::DirectCwd);
            Ok(InputAction::Continue)
        }
        Action::SetFilterDirs => {
            set_filter(app, FilterType::OnlyDirs);
            Ok(InputAction::Continue)
        }
        Action::SetFilterFiles => {
            set_filter(app, FilterType::OnlyFiles);
            Ok(InputAction::Continue)
        }
        Action::CycleDebugPane => {
            app.ui_state.cycle_debug_pane_mode();
            // The pane shows database counts; fetch them the first time it opens
            // rather than making every launch pay for them.
            if app.ui_state.debug_pane_mode != crate::ui_state::DebugPaneMode::Hidden {
                app.request_db_stats();
            }
            Ok(InputAction::Continue)
        }
        Action::ScrollPreviewUp => {
            app.preview.scroll(-3);
            let _ = app.log_preview_scroll();
            Ok(InputAction::Continue)
        }
        Action::ScrollPreviewDown => {
            app.preview.scroll(3);
            let _ = app.log_preview_scroll();
            Ok(InputAction::Continue)
        }
        Action::ToggleHelp => {
            app.ui_state.toggle_help();
            Ok(InputAction::Continue)
        }
        Action::Escape => handle_escape(app),
        Action::Quit => Ok(InputAction::Exit),
    }
}

/// Handle a key while the help screen is up.
///
/// Scroll keys scroll it, quitting still quits, and everything else closes it -
/// so no key can act on the UI hidden behind the help screen.
fn handle_help_key(app: &mut App, code: KeyCode, modifiers: KeyModifiers) -> InputAction {
    match keymap::lookup(code, modifiers, Context::Global) {
        Some(Action::Quit) => return InputAction::Exit,
        Some(Action::MoveUp) => {
            app.ui_state.scroll_help(-1);
            return InputAction::Continue;
        }
        Some(Action::MoveDown) => {
            app.ui_state.scroll_help(1);
            return InputAction::Continue;
        }
        _ => {}
    }

    app.ui_state.hide_help();
    InputAction::Continue
}

/// Execute on-cwd-visit action for a given directory
fn execute_cwd_visit_action(
    app: &App,
    dir_path: &std::path::Path,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<InputAction> {
    match app.options.on_cwd_visit {
        OnCwdVisitAction::PrintToStdout => {
            cleanup_terminal(terminal)?;
            Ok(InputAction::PrintAndExit(dir_path.display().to_string()))
        }
        OnCwdVisitAction::DropIntoShell => {
            suspend_tui_and_run_shell(app, dir_path, terminal)?;
            Ok(InputAction::Continue)
        }
    }
}

/// Handle Ctrl-J (execute on-cwd-visit action for current directory)
fn handle_ctrl_j(
    app: &App,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<InputAction> {
    execute_cwd_visit_action(app, &app.cwd, terminal)
}

/// The selected row, confirmed to still exist on disk.
///
/// Produced only by `resolve_selection`, so holding one is evidence the check
/// was made.
struct Selection {
    display_name: String,
    full_path: std::path::PathBuf,
    mtime: Option<i64>,
    atime: Option<i64>,
    file_size: Option<i64>,
    is_dir: bool,
}

/// Resolve the selected row, dropping it if it is no longer on disk.
///
/// The file registry is a cache of the filesystem, validated once at startup
/// (`search_worker::WorkerState::new` skips historical paths that do not
/// exist). Nothing revalidates it afterwards, so a file deleted mid-session
/// stays in the results. Acting on such a row used to log a click on a
/// nonexistent path and then hand that path to the shell, which failed the
/// `cd` after psychic had already exited.
///
/// This is the one place a path becomes actionable, so it is the one place the
/// check belongs: every action that touches the selection - open, navigate,
/// print-and-exit, drop-into-shell - goes through here first. `Ok(None)` means
/// the caller should do nothing; the row has been evicted and the user told.
///
/// One `stat` per keypress on one path, so the cost is not measurable. It is
/// deliberately not done at render time: `App::get_file_at_index` is called for
/// every visible row on every frame and must stay IO-free.
fn resolve_selection(app: &mut App) -> Option<Selection> {
    let display_info = app.get_file_at_index(app.selected_index)?;

    // Clone what we need before the mutable borrows below.
    let selection = Selection {
        display_name: display_info.display_name.clone(),
        full_path: display_info.full_path.clone(),
        mtime: display_info.mtime,
        atime: display_info.atime,
        file_size: display_info.file_size,
        is_dir: display_info.is_dir,
    };

    // `exists()` follows symlinks, matching the startup filter: a broken
    // symlink is not something the user can open or cd into either.
    if selection.full_path.exists() {
        return Some(selection);
    }

    log::info!(
        "Selected path no longer exists, evicting: {:?}",
        selection.full_path
    );

    let query_id = app.next_query_id();
    let _ = app.worker_tx.send(WorkerRequest::Evict {
        path: selection.full_path.clone(),
        query_id,
    });

    app.status_message = Some(format!("Gone: {}", selection.display_name));

    None
}

/// Log a click on the selected row.
///
/// Shared by the actions that count as a click so that they agree on what gets
/// written to the events table - this is training data, and a click logged by
/// one path but not another would be a silent hole in it.
fn log_selection_click(app: &mut App, selection: &Selection) -> Result<()> {
    let subsession_id = app.analytics.current_subsession_id();
    let session_id = app.analytics.session_id().to_string();
    app.analytics.log_click(EventData {
        query: &app.query,
        file_path: &selection.display_name,
        full_path: &selection.full_path.to_string_lossy(),
        mtime: selection.mtime,
        atime: selection.atime,
        file_size: selection.file_size,
        subsession_id,
        action: UserInteraction::Click,
        session_id: &session_id,
        episode_queries: None,
    })
}

/// Handle Ctrl-Enter (execute on-cwd-visit action for selected directory)
fn handle_ctrl_enter(
    app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<InputAction> {
    if app.ui_state.history_mode {
        // In history mode, same as regular Enter
        app.handle_history_enter()?;
        return Ok(InputAction::Continue);
    }

    if app.total_results == 0 {
        return Ok(InputAction::Continue);
    }

    // Log impressions before action
    app.check_and_log_impressions(true)?;

    let Some(selection) = resolve_selection(app) else {
        return Ok(InputAction::Continue);
    };

    // Only handle directories
    if !selection.is_dir {
        return Ok(InputAction::Continue);
    }

    log_selection_click(app, &selection)?;

    // Execute the on-cwd-visit action for the selected directory
    execute_cwd_visit_action(app, &selection.full_path, terminal)
}

/// Handle Ctrl-H (toggle history mode)
fn handle_ctrl_h(app: &mut App) {
    if app.ui_state.history_mode {
        app.ui_state.history_mode = false;
        app.query.clear();
    } else {
        app.ui_state.history_mode = true;
        app.history_selected = app.history.current_display_index();
        app.query.clear();
        app.preview.clear();
    }
}

/// Handle Alt-Up (navigate to parent directory)
fn handle_parent_dir(app: &mut App) -> Result<InputAction> {
    if let Some(parent) = app.cwd.parent() {
        let parent = parent.to_path_buf();
        log::info!("Navigating to parent directory: {:?}", parent);
        app.history.navigate_to(parent.clone());
        app.cwd = parent.clone();
        app.query.clear();

        let query_id = app.next_query_id();
        let _ = app.worker_tx.send(WorkerRequest::ChangeCwd {
            new_cwd: parent,
            query_id,
        });
    }
    Ok(InputAction::Continue)
}

/// Handle Left arrow (go back in history)
fn handle_history_back(app: &mut App) -> Result<InputAction> {
    if let Some(dir) = app.history.go_back() {
        log::info!("Navigating back in history to: {:?}", dir);
        app.cwd = dir.clone();
        app.query.clear();

        let query_id = app.next_query_id();
        let _ = app.worker_tx.send(WorkerRequest::ChangeCwd {
            new_cwd: dir,
            query_id,
        });
    }
    Ok(InputAction::Continue)
}

/// Handle Ctrl-Right (go forward in history)
fn handle_history_forward(app: &mut App) -> Result<InputAction> {
    if let Some(dir) = app.history.go_forward() {
        log::info!("Navigating forward in history to: {:?}", dir);
        app.cwd = dir.clone();
        app.query.clear();

        let query_id = app.next_query_id();
        let _ = app.worker_tx.send(WorkerRequest::ChangeCwd {
            new_cwd: dir,
            query_id,
        });
    }
    Ok(InputAction::Continue)
}

/// Handle Ctrl-U (clear search query)
fn handle_ctrl_u(app: &mut App) {
    app.query.clear();
    send_query_update(app);
}

/// Handle Escape key
fn handle_escape(app: &mut App) -> Result<InputAction> {
    if app.ui_state.filter_picker_visible {
        app.ui_state.filter_picker_visible = false;
        Ok(InputAction::Continue)
    } else if app.ui_state.history_mode {
        app.ui_state.history_mode = false;
        app.query.clear();
        Ok(InputAction::Continue)
    } else {
        Ok(InputAction::Exit)
    }
}

/// Handle up/down navigation
fn handle_navigation(app: &mut App, delta: isize) {
    if app.ui_state.history_mode {
        app.move_history_selection(delta);
    } else {
        app.move_selection(delta);
    }
}

/// Handle character input (adds to search query)
fn handle_char_input(app: &mut App, c: char) {
    app.query.push(c);
    send_query_update(app);
}

/// Handle backspace (removes from search query)
fn handle_backspace(app: &mut App) {
    app.query.pop();
    send_query_update(app);
}

/// Handle Enter key (select file/directory or navigate history)
fn handle_enter(
    app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<InputAction> {
    if app.ui_state.history_mode {
        app.handle_history_enter()?;
        return Ok(InputAction::Continue);
    }

    if app.total_results == 0 {
        return Ok(InputAction::Continue);
    }

    // Log impressions before click (analytics module handles no_logging flag)
    app.check_and_log_impressions(true)?;

    let Some(selection) = resolve_selection(app) else {
        return Ok(InputAction::Continue);
    };

    // Log the click event (analytics module handles no_logging flag)
    log_selection_click(app, &selection)?;

    if selection.is_dir {
        handle_directory_click(app, selection.full_path, terminal)
    } else {
        handle_file_click(app, selection.full_path, terminal)?;
        Ok(InputAction::Continue)
    }
}

/// Handle clicking on a directory
fn handle_directory_click(
    app: &mut App,
    dir_path: std::path::PathBuf,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<InputAction> {
    match app.options.on_dir_click {
        OnDirClickAction::Navigate => {
            if dir_path == app.cwd {
                log::info!("Already in {:?}, not navigating", dir_path);
                return Ok(InputAction::Continue);
            }

            log::info!("Navigating to directory: {:?}", dir_path);
            app.history.navigate_to(dir_path.clone());
            app.cwd = dir_path.clone();
            app.query.clear();

            let query_id = app.next_query_id();
            let _ = app.worker_tx.send(WorkerRequest::ChangeCwd {
                new_cwd: dir_path,
                query_id,
            });
            Ok(InputAction::Continue)
        }
        OnDirClickAction::PrintToStdout => {
            cleanup_terminal(terminal)?;
            Ok(InputAction::PrintAndExit(dir_path.display().to_string()))
        }
        OnDirClickAction::DropIntoShell => {
            suspend_tui_and_run_shell(app, &dir_path, terminal)?;
            Ok(InputAction::Continue)
        }
    }
}

/// Handle clicking on a file (open in editor)
fn handle_file_click(
    app: &mut App,
    file_path: std::path::PathBuf,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<()> {
    suspend_tui_for_editor(app, &file_path, terminal)?;

    // Reload model and rerank after editing
    let query_id_model = app.next_query_id();
    if let Err(e) = app.reload_model(query_id_model) {
        log::error!("Failed to reload model: {}", e);
    }

    let query_id_clicks = app.next_query_id();
    if let Err(e) = app.reload_and_rerank(query_id_clicks) {
        log::error!("Failed to reload and rerank: {}", e);
    }

    Ok(())
}

/// Set the current filter and trigger query update
fn set_filter(app: &mut App, filter: FilterType) {
    app.current_filter = filter;
    app.ui_state.filter_picker_visible = false;
    send_query_update(app);
}

/// Cycle to next or previous filter
fn cycle_filter(app: &mut App, forward: bool) {
    let new_filter = if forward {
        app.current_filter.next()
    } else {
        app.current_filter.prev()
    };
    app.current_filter = new_filter;
    send_query_update(app);
}

/// Send query update to worker
fn send_query_update(app: &mut App) {
    let query_id = app.next_query_id();
    let _ = app
        .worker_tx
        .send(WorkerRequest::UpdateQuery(UpdateQueryRequest {
            query: app.query.clone(),
            query_id,
            filter: app.current_filter,
        }));
}

/// Cleanup terminal before exiting
fn cleanup_terminal(terminal: &mut Terminal<CrosstermBackend<std::fs::File>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        crossterm::event::DisableMouseCapture
    )?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    execute!(terminal.backend_mut(), Show)?;
    Ok(())
}

/// Suspend TUI and run a shell in the given directory
fn suspend_tui_and_run_shell(
    app: &App,
    dir: &std::path::Path,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<()> {
    // Pause crossterm thread
    let _ = app.input_control_tx.send(false);

    // Pause tick thread to prevent event queue buildup
    app.tick_paused
        .store(true, std::sync::atomic::Ordering::Relaxed);

    // Give the input thread time to enter paused state to avoid race condition
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Suspend TUI
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        crossterm::event::DisableMouseCapture
    )?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    execute!(terminal.backend_mut(), Show)?;

    // Run shell
    use std::process::Stdio;
    let tty_in = std::fs::OpenOptions::new().read(true).open("/dev/tty")?;
    let tty_out = std::fs::OpenOptions::new().write(true).open("/dev/tty")?;
    let tty_err = std::fs::OpenOptions::new().write(true).open("/dev/tty")?;

    let shell = std::env::var("SHELL").unwrap_or_else(|_| "sh".to_string());
    let status = std::process::Command::new(&shell)
        .current_dir(dir)
        .stdin(Stdio::from(tty_in))
        .stdout(Stdio::from(tty_out))
        .stderr(Stdio::from(tty_err))
        .status();

    // Resume TUI
    log::info!("Resuming TUI after editor/shell");
    enable_raw_mode()?;
    execute!(terminal.backend_mut(), EnterAlternateScreen)?;
    execute!(terminal.backend_mut(), crossterm::event::EnableMouseCapture)?;
    // Re-enable enhanced keyboard protocol
    execute!(
        terminal.backend_mut(),
        crossterm::event::PushKeyboardEnhancementFlags(
            crossterm::event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
        )
    )?;
    terminal.clear()?;

    // Resume crossterm thread
    log::info!("Sending resume signal to input thread");
    let _ = app.input_control_tx.send(true);

    // Resume tick thread
    app.tick_paused
        .store(false, std::sync::atomic::Ordering::Relaxed);

    if let Err(e) = status {
        log::error!("Failed to launch shell: {}", e);
    }

    Ok(())
}

/// Suspend TUI and run editor for the given file
fn suspend_tui_for_editor(
    app: &App,
    file_path: &std::path::Path,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<()> {
    // Pause crossterm thread
    let _ = app.input_control_tx.send(false);

    // Pause tick thread to prevent event queue buildup
    app.tick_paused
        .store(true, std::sync::atomic::Ordering::Relaxed);

    // Give the input thread time to enter paused state to avoid race condition
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Suspend TUI
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        crossterm::event::DisableMouseCapture
    )?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    execute!(terminal.backend_mut(), Show)?;

    // Run editor
    use std::process::Stdio;
    let tty_in = std::fs::OpenOptions::new().read(true).open("/dev/tty")?;
    let tty_out = std::fs::OpenOptions::new().write(true).open("/dev/tty")?;
    let tty_err = std::fs::OpenOptions::new().write(true).open("/dev/tty")?;

    let status = std::process::Command::new(&app.options.editor)
        .arg(file_path)
        .stdin(Stdio::from(tty_in))
        .stdout(Stdio::from(tty_out))
        .stderr(Stdio::from(tty_err))
        .status();

    // Resume TUI
    log::info!("Resuming TUI after editor/shell");
    enable_raw_mode()?;
    execute!(terminal.backend_mut(), EnterAlternateScreen)?;
    execute!(terminal.backend_mut(), crossterm::event::EnableMouseCapture)?;
    // Re-enable enhanced keyboard protocol
    execute!(
        terminal.backend_mut(),
        crossterm::event::PushKeyboardEnhancementFlags(
            crossterm::event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
        )
    )?;
    terminal.clear()?;

    // Resume crossterm thread
    log::info!("Sending resume signal to input thread");
    let _ = app.input_control_tx.send(true);

    // Resume tick thread
    app.tick_paused
        .store(false, std::sync::atomic::Ordering::Relaxed);

    if let Err(e) = status {
        log::error!("Failed to launch editor: {}", e);
    }

    Ok(())
}
