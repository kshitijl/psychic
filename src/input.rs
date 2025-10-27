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
            use crossterm::event::MouseEventKind;
            match mouse_event.kind {
                MouseEventKind::ScrollDown => {
                    app.preview.scroll(3);
                    let _ = app.log_preview_scroll();
                }
                MouseEventKind::ScrollUp => {
                    app.preview.scroll(-3);
                    let _ = app.log_preview_scroll();
                }
                _ => {}
            }
            Ok(InputAction::Continue)
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
    match code {
        KeyCode::Char('j') if modifiers.contains(KeyModifiers::CONTROL) => {
            handle_ctrl_j(app, terminal)
        }
        KeyCode::Char('h') if modifiers.contains(KeyModifiers::CONTROL) => {
            handle_ctrl_h(app);
            Ok(InputAction::Continue)
        }
        KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => Ok(InputAction::Exit),
        KeyCode::Char('d') if modifiers.contains(KeyModifiers::CONTROL) => Ok(InputAction::Exit),
        KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
            handle_ctrl_u(app);
            Ok(InputAction::Continue)
        }
        KeyCode::Char('o') if modifiers.contains(KeyModifiers::CONTROL) => {
            app.ui_state.cycle_debug_pane_mode();
            Ok(InputAction::Continue)
        }
        KeyCode::Char('f') if modifiers.contains(KeyModifiers::CONTROL) => {
            app.ui_state.filter_picker_visible = !app.ui_state.filter_picker_visible;
            Ok(InputAction::Continue)
        }
        KeyCode::Char('0') if app.ui_state.filter_picker_visible => {
            set_filter(app, FilterType::None);
            Ok(InputAction::Continue)
        }
        KeyCode::Char('c') if app.ui_state.filter_picker_visible => {
            set_filter(app, FilterType::OnlyCwd);
            Ok(InputAction::Continue)
        }
        KeyCode::Char('i') if app.ui_state.filter_picker_visible => {
            set_filter(app, FilterType::DirectCwd);
            Ok(InputAction::Continue)
        }
        KeyCode::Char('d') if app.ui_state.filter_picker_visible => {
            set_filter(app, FilterType::OnlyDirs);
            Ok(InputAction::Continue)
        }
        KeyCode::Char('f') if app.ui_state.filter_picker_visible => {
            set_filter(app, FilterType::OnlyFiles);
            Ok(InputAction::Continue)
        }
        KeyCode::BackTab => {
            cycle_filter(app, false);
            Ok(InputAction::Continue)
        }
        KeyCode::Tab => {
            cycle_filter(app, true);
            Ok(InputAction::Continue)
        }
        KeyCode::Esc => handle_escape(app),
        KeyCode::Up => {
            handle_navigation(app, -1);
            Ok(InputAction::Continue)
        }
        KeyCode::Down => {
            handle_navigation(app, 1);
            Ok(InputAction::Continue)
        }
        KeyCode::Left => handle_history_back(app),
        KeyCode::Right => handle_history_forward(app),
        KeyCode::Char('p') if modifiers.contains(KeyModifiers::CONTROL) => {
            handle_navigation(app, -1);
            Ok(InputAction::Continue)
        }
        KeyCode::Char('n') if modifiers.contains(KeyModifiers::CONTROL) => {
            handle_navigation(app, 1);
            Ok(InputAction::Continue)
        }
        KeyCode::Char(c) => {
            handle_char_input(app, c);
            Ok(InputAction::Continue)
        }
        KeyCode::Backspace => {
            handle_backspace(app);
            Ok(InputAction::Continue)
        }
        KeyCode::Enter if modifiers.contains(KeyModifiers::CONTROL) => {
            handle_ctrl_enter(app, terminal)
        }
        KeyCode::Enter => handle_enter(app, terminal),
        _ => Ok(InputAction::Continue),
    }
}

/// Execute on-cwd-visit action for a given directory
fn execute_cwd_visit_action(
    app: &App,
    dir_path: &std::path::Path,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<InputAction> {
    match app.on_cwd_visit {
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

    let Some(display_info) = app.get_file_at_index(app.selected_index) else {
        return Ok(InputAction::Continue);
    };

    // Only handle directories
    if !display_info.is_dir {
        return Ok(InputAction::Continue);
    }

    // Log the click event
    let subsession_id = app.analytics.current_subsession_id();
    let session_id = app.analytics.session_id().to_string();
    app.analytics.log_click(EventData {
        query: &app.query,
        file_path: &display_info.display_name,
        full_path: &display_info.full_path.to_string_lossy(),
        mtime: display_info.mtime,
        atime: display_info.atime,
        file_size: display_info.file_size,
        subsession_id,
        action: UserInteraction::Click,
        session_id: &session_id,
    })?;

    // Execute the on-cwd-visit action for the selected directory
    execute_cwd_visit_action(app, &display_info.full_path, terminal)
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

/// Handle Ctrl-Left (go back in history)
fn handle_history_back(app: &mut App) -> Result<InputAction> {
    if let Some(dir) = app.history.go_back() {
        log::info!("Navigating back in history to: {:?}", dir);
        app.cwd = dir.clone();
        app.query.clear();

        let query_id = app.analytics.next_subsession_id();
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

        let query_id = app.analytics.next_subsession_id();
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

    let Some(display_info) = app.get_file_at_index(app.selected_index) else {
        return Ok(InputAction::Continue);
    };

    // Log the click event (analytics module handles no_logging flag)
    let subsession_id = app.analytics.current_subsession_id();
    let session_id = app.analytics.session_id().to_string();
    app.analytics.log_click(EventData {
        query: &app.query,
        file_path: &display_info.display_name,
        full_path: &display_info.full_path.to_string_lossy(),
        mtime: display_info.mtime,
        atime: display_info.atime,
        file_size: display_info.file_size,
        subsession_id,
        action: UserInteraction::Click,
        session_id: &session_id,
    })?;

    if display_info.is_dir {
        handle_directory_click(app, display_info.full_path.clone(), terminal)
    } else {
        handle_file_click(app, display_info.full_path.clone(), terminal)?;
        Ok(InputAction::Continue)
    }
}

/// Handle clicking on a directory
fn handle_directory_click(
    app: &mut App,
    dir_path: std::path::PathBuf,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<InputAction> {
    match app.on_dir_click {
        OnDirClickAction::Navigate => {
            if dir_path == app.cwd {
                log::info!("Already in {:?}, not navigating", dir_path);
                return Ok(InputAction::Continue);
            }

            log::info!("Navigating to directory: {:?}", dir_path);
            app.history.navigate_to(dir_path.clone());
            app.cwd = dir_path.clone();
            app.query.clear();

            let query_id = app.analytics.next_subsession_id();
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
    let query_id_model = app.analytics.next_subsession_id();
    if let Err(e) = app.reload_model(query_id_model) {
        log::error!("Failed to reload model: {}", e);
    }

    let query_id_clicks = app.analytics.next_subsession_id();
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
    let query_id = app.analytics.next_subsession_id();
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

    let status = std::process::Command::new("hx")
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

    if let Err(e) = status {
        log::error!("Failed to launch editor: {}", e);
    }

    Ok(())
}
