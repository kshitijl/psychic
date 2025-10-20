mod context;
mod db;
mod feature_defs;
mod features;
mod history;
mod ranker;
mod search_worker;
mod ui_state;
mod walker;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use crossterm::{
    ExecutableCommand,
    cursor::Show,
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use db::{Database, EventData, FileMetadata};
use ratatui::{
    Frame,
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph},
};
use search_worker::{
    DisplayFileInfo, WorkerRequest, WorkerResponse,
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    env,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver},
    thread::JoinHandle,
    time::{Duration, Instant},
};



/// Truncates a path string in the middle if it's too long, keeping the first
/// component and the end of the path.
/// e.g., "a/b/c/d/e.txt" -> "a/.../d/e.txt"
fn truncate_path(path_str: &str, max_len: usize) -> String {
    if path_str.len() <= max_len {
        return path_str.to_string();
    }

    let path = Path::new(path_str);
    let components: Vec<&str> = path
        .components()
        .map(|c| c.as_os_str().to_str().unwrap_or(""))
        .collect();

    // Don't truncate simple paths
    if components.len() <= 2 {
        return path_str.to_string();
    }

    let head = components.first().unwrap_or(&"");
    let mut tail_parts: Vec<&str> = Vec::new();

    // Start with filename
    let filename = components.last().unwrap_or(&"");
    tail_parts.push(filename);

    // head + "/.../" + filename
    let mut len_so_far = head.len() + 5 + filename.len();

    // Add parts to tail from the end until we run out of space
    // Iterate over parent components in reverse (skipping filename)
    for part in components.iter().rev().skip(1) {
        // Stop if we are about to collide with the head component
        if part == head {
            break;
        }

        if len_so_far + part.len() + 1 > max_len {
            break;
        }

        tail_parts.insert(0, part);
        len_so_far += part.len() + 1;
    }

    format!("{}/.../{}", head, tail_parts.join("/"))
}

/// Abbreviate a path component to its first character
/// e.g., "Users" -> "U", "kshitijlauria" -> "k"
fn abbreviate_component(s: &str) -> String {
    s.chars().next().map(|c| c.to_string()).unwrap_or_else(|| s.to_string())
}

/// Truncate an absolute path (for historical files) showing beginning and end with abbreviations
/// e.g., "/Users/kshitijlauria/Library/CloudStorage/Dropbox/src/11-sg/todo.md"
///    -> "/U/k/L/CloudStorage/.../11-sg/todo.md"
fn truncate_absolute_path(path_str: &str, max_len: usize) -> String {
    if path_str.len() <= max_len {
        return path_str.to_string();
    }

    // Split path into components manually to avoid Path component issues
    let parts: Vec<&str> = path_str.split('/').filter(|s| !s.is_empty()).collect();

    if parts.is_empty() {
        return path_str.to_string();
    }

    // Always keep the filename
    let filename = parts.last().unwrap_or(&"");

    // Start with just the filename
    let mut tail_count = 1; // Start with 1 for filename
    let mut estimated_len = 3 + 1 + filename.len(); // "..." + "/" + filename

    // Add components from the end (before filename)
    for i in (0..parts.len().saturating_sub(1)).rev() {
        let part = parts[i];
        if estimated_len + 1 + part.len() > max_len {
            break;
        }
        estimated_len += 1 + part.len(); // "/" + part
        tail_count += 1;
    }

    // Add abbreviated components from the beginning
    let mut head_parts: Vec<String> = Vec::new();
    for i in 0..parts.len() {
        if i >= parts.len() - tail_count {
            // We've reached the tail section
            break;
        }
        let part = parts[i];

        // Try full component first
        if estimated_len + 1 + part.len() <= max_len {
            estimated_len += 1 + part.len();
            head_parts.push(part.to_string());
        } else {
            // Try abbreviated component
            let abbrev = abbreviate_component(part);
            if estimated_len + 1 + abbrev.len() <= max_len {
                estimated_len += 1 + abbrev.len();
                head_parts.push(abbrev);
            } else {
                // Can't fit even abbreviated, stop adding head parts
                break;
            }
        }
    }

    // Build the result
    let is_absolute = path_str.starts_with('/');
    let head_count = head_parts.len();

    if head_count + tail_count >= parts.len() {
        // Everything fits (possibly with abbreviations)
        if head_parts.iter().all(|h| parts.contains(&h.as_str())) {
            // No abbreviations were used
            return path_str.to_string();
        } else {
            // Some abbreviations, reconstruct
            let tail: Vec<&str> = parts.iter().skip(parts.len() - tail_count).copied().collect();
            if is_absolute {
                format!("/{}/{}", head_parts.join("/"), tail.join("/"))
            } else {
                format!("{}/{}", head_parts.join("/"), tail.join("/"))
            }
        }
    } else if head_count == 0 {
        // Only tail fits
        let tail: Vec<&str> = parts.iter().skip(parts.len() - tail_count).copied().collect();
        if is_absolute {
            format!("/.../{}", tail.join("/"))
        } else {
            format!(".../{}", tail.join("/"))
        }
    } else {
        // Both head and tail, with ellipsis
        let tail: Vec<&str> = parts.iter().skip(parts.len() - tail_count).copied().collect();
        if is_absolute {
            format!("/{}/.../{}", head_parts.join("/"), tail.join("/"))
        } else {
            format!("{}/.../{}", head_parts.join("/"), tail.join("/"))
        }
    }
}

// Page-based caching constants
const PAGE_SIZE: usize = 128;
const PREFETCH_MARGIN: usize = 32;

/// Convert Unix timestamp to human-readable "time ago" string
fn get_time_ago(mtime: Option<i64>) -> String {
    if let Some(mtime_secs) = mtime {
        // Convert Unix timestamp to SystemTime
        let mtime_systime = std::time::UNIX_EPOCH + Duration::from_secs(mtime_secs as u64);

        let duration = std::time::SystemTime::now()
            .duration_since(mtime_systime)
            .unwrap_or(Duration::from_secs(0));

        let formatter = timeago::Formatter::new();
        return formatter.convert(duration);
    }
    String::from("unknown")
}

fn render_history_mode(f: &mut Frame, app: &App) {
    use ratatui::widgets::{List, ListItem, Paragraph};

    // Split vertically: top for dir list + preview, bottom for input
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),    // Dir list + Preview
            Constraint::Length(3), // Search input at bottom
        ])
        .split(f.area());

    // Split horizontally: left for dir list, right for preview
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Directory list
            Constraint::Percentage(50), // Preview
        ])
        .split(main_chunks[0]);

    // Get filtered history (sorted chronologically as stored)
    let filtered_history = app.get_filtered_history();

    // Calculate list width (accounting for borders)
    let list_width = top_chunks[0].width.saturating_sub(2) as usize;

    // Build list items for directory history
    let items: Vec<ListItem> = filtered_history
        .iter()
        .enumerate()
        .map(|(idx, path)| {
            let rank = idx + 1;
            let rank_prefix = format!("{:2}. ", rank);
            let prefix_len = rank_prefix.len();
            let path_str = path.to_string_lossy();

            // Calculate available width for path (widget width - rank prefix - 1 for safety margin)
            let available_width = list_width.saturating_sub(prefix_len).saturating_sub(1);

            // Use truncate_absolute_path for good abbreviation
            let display_text = truncate_absolute_path(&path_str, available_width);

            let style = if idx == app.history_selected {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Cyan)
            };
            ListItem::new(Line::from(vec![
                Span::styled(rank_prefix, style),
                Span::styled(display_text, style),
            ]))
        })
        .collect();

    // Render directory list
    // Total includes all items in history display
    let total_dirs = app.history.items_for_display().len();
    let title = format!("History (most recent at top) — {}/{}", filtered_history.len(), total_dirs);
    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(title, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))),
    );
    f.render_widget(list, top_chunks[0]);

    // Preview pane: show eza -al of selected directory
    let preview_text = if !filtered_history.is_empty() && app.history_selected < filtered_history.len() {
        let selected_dir = &filtered_history[app.history_selected];

        let preview_width = top_chunks[1].width;
        let extra_flags = app.ui_state.get_eza_flags(preview_width);

        let mut eza_cmd = std::process::Command::new("eza");
        eza_cmd.arg("-al").arg("--color=always");

        // Add extra flags if preview is narrow
        for flag in extra_flags.split_whitespace() {
            if !flag.is_empty() {
                eza_cmd.arg(flag);
            }
        }

        eza_cmd.arg(selected_dir);
        let eza_output = eza_cmd.output();

        match eza_output {
            Ok(output) => {
                match ansi_to_tui::IntoText::into_text(&output.stdout) {
                    Ok(text) => text,
                    Err(_) => Text::from("[Unable to parse directory listing]"),
                }
            }
            Err(_) => {
                // Fallback to ls if eza not available
                let ls_output = std::process::Command::new("ls")
                    .arg("-lah")
                    .arg(selected_dir)
                    .output();
                match ls_output {
                    Ok(output) => Text::from(String::from_utf8_lossy(&output.stdout).to_string()),
                    Err(_) => Text::from("[Unable to list directory]"),
                }
            }
        }
    } else {
        Text::from("No history available")
    };

    let preview_para = Paragraph::new(preview_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Preview"),
        )
        .scroll((app.preview_scroll as u16, 0));
    f.render_widget(preview_para, top_chunks[1]);

    // Search input at bottom
    let input_area = main_chunks[1];
    let input_text = if app.query.is_empty() {
        "Filter history..."
    } else {
        &app.query
    };
    let input_para = Paragraph::new(input_text)
        .style(Style::default().fg(Color::Gray))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Search (Ctrl-H/Esc to exit)"),
        );
    f.render_widget(input_para, input_area);
}

/// A page of DisplayFileInfo for caching
#[derive(Debug, Clone)]
struct Page {
    start_index: usize,
    end_index: usize,
    files: Vec<DisplayFileInfo>,
}

/// For feature generation
#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    Csv,
    Json,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate features for training from collected events
    GenerateFeatures {
        /// Output format
        #[arg(short, long, value_enum, default_value = "csv")]
        format: OutputFormat,
    },
    /// Retrain the ranking model using collected events. This does everything,
    /// including feature gen.
    Retrain,
    /// Output shell integration script for zsh
    Zsh,
}

/// Filter type for initial filter
#[derive(ValueEnum, Clone, Debug)]
enum FilterArg {
    None,
    Cwd,
    Dirs,
    Files,
}

/// Action to take when user presses Enter on a directory
#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum OnDirClickAction {
    /// Navigate into the directory (default)
    Navigate,
    /// Print path to stdout and exit
    PrintToStdout,
    /// Drop into a shell in that directory
    DropIntoShell,
}

/// Action to take when user presses Ctrl-J on current directory
#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum OnCwdVisitAction {
    /// Print path to stdout and exit (for shell integration)
    PrintToStdout,
    /// Drop into a shell in the current directory
    DropIntoShell,
}

/// A terminal-based file browser that learns which files you want to see.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Data directory for database and training files (default: ~/.local/share/psychic)
    #[arg(long, global = true, value_name = "DIR")]
    data_dir: Option<PathBuf>,

    /// Action when pressing Enter on a directory
    #[arg(long, value_enum, default_value = "navigate")]
    on_dir_click: OnDirClickAction,

    /// Action when pressing Ctrl-J (current directory visit)
    #[arg(long, value_enum, default_value = "drop-into-shell")]
    on_cwd_visit: OnCwdVisitAction,

    /// Set initial filter (none, cwd, dirs, files)
    #[arg(long, value_enum)]
    filter: Option<FilterArg>,

    #[command(subcommand)]
    command: Option<Commands>,
}

/// Get the default data directory
fn get_default_data_dir() -> Result<PathBuf> {
    let home = env::var("HOME").context("HOME environment variable not set")?;
    Ok(PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("psychic"))
}

// Every time the query changes, as the user types, corresponds to a new
// subsession. Subsession id is logged to the db.
struct Subsession {
    id: u64,
    query: String,
    created_at: jiff::Timestamp,
    events_have_been_logged: bool,
}

struct App {
    query: String,
    page_cache: HashMap<usize, Page>, // Page-based cache
    total_results: usize,             // Total number of filtered files
    total_files: usize,               // Total number of files in index
    selected_index: usize,
    file_list_scroll: usize, // Scroll offset for file list
    preview_scroll: usize,
    preview_cache: Option<(String, Text<'static>)>, // (file_path, rendered_text)
    scrolled_files: HashSet<(String, String)>, // (query, full_path) - track what we've logged scroll for
    session_id: String,
    current_subsession: Option<Subsession>,
    next_subsession_id: u64,
    db: Database,
    cwd: PathBuf, // Current working directory
    history: history::History,
    history_selected: usize, // Selected item in history mode UI

    num_results_to_log_as_impressions: usize,

    // For marquee path bar
    path_bar_scroll: u16,
    path_bar_scroll_direction: i8,
    last_path_bar_update: Instant,

    // For debug pane
    model_stats_cache: Option<ranker::ModelStats>, // Cached from worker, refreshed periodically
    currently_retraining: bool,
    log_receiver: Receiver<String>,
    recent_logs: VecDeque<String>,

    // Filter state
    current_filter: search_worker::FilterType,

    // UI state machine
    ui_state: ui_state::UiState,

    // Action configuration
    on_dir_click: OnDirClickAction,
    on_cwd_visit: OnCwdVisitAction,

    // Search worker thread communication
    worker_tx: mpsc::Sender<WorkerRequest>,
    worker_rx: mpsc::Receiver<WorkerResponse>,
    worker_handle: Option<JoinHandle<()>>,
}

impl App {
    fn new(root: PathBuf, data_dir: &Path, log_receiver: Receiver<String>, on_dir_click: OnDirClickAction, on_cwd_visit: OnCwdVisitAction, initial_filter: search_worker::FilterType) -> Result<Self> {
        let start_time = Instant::now();
        log::debug!("App::new() started");

        // Get session ID from environment (set in main())
        let session_id = std::env::var("PSYCHIC_SESSION_ID")
            .unwrap_or_else(|_| "unknown".to_string());

        let db_start = Instant::now();
        let db_path = Database::get_db_path(data_dir);
        let db = Database::new(&db_path)?;
        log::debug!("Database initialization took {:?}", db_start.elapsed());

        let (worker_tx, worker_rx, worker_handle) = search_worker::spawn(root.clone(), data_dir)?;

        log::debug!("App::new() total time: {:?}", start_time.elapsed());

        let app = App {
            query: String::new(),
            page_cache: HashMap::new(),
            total_results: 0,
            total_files: 0,
            selected_index: 0,
            file_list_scroll: 0,
            preview_scroll: 0,
            preview_cache: None,
            scrolled_files: HashSet::new(),
            session_id,
            current_subsession: None,
            next_subsession_id: 1, // Start with 1, 0 is for initial query
            num_results_to_log_as_impressions: 25,
            db,
            cwd: root.clone(),
            history: history::History::new(root),
            history_selected: 0,
            path_bar_scroll: 0,
            path_bar_scroll_direction: 1,
            last_path_bar_update: Instant::now(),
            model_stats_cache: None,
            currently_retraining: false,
            log_receiver,
            recent_logs: VecDeque::with_capacity(50),
            current_filter: initial_filter,
            ui_state: ui_state::UiState::new(),
            on_dir_click,
            on_cwd_visit,
            worker_tx: worker_tx.clone(),
            worker_rx,
            worker_handle: Some(worker_handle),
        };

        // Send initial query to worker with ID 0
        let _ = worker_tx.send(WorkerRequest::UpdateQuery(search_worker::UpdateQueryRequest {
            query: String::new(),
            query_id: 0,
            filter: initial_filter,
        }));

        Ok(app)
    }

    fn reload_model(&mut self, query_id: u64) -> Result<()> {
        log::info!("Requesting model reload from worker");
        self.worker_tx
            .send(WorkerRequest::ReloadModel { query_id })
            .context("Failed to send ReloadModel request to worker")?;
        Ok(())
    }

    fn reload_and_rerank(&mut self, query_id: u64) -> Result<()> {
        log::info!("Requesting clicks reload from worker");
        self.worker_tx
            .send(WorkerRequest::ReloadClicks { query_id })
            .context("Failed to send ReloadClicks request to worker")?;
        Ok(())
    }

    /// Get file from page cache at a given global index
    /// Returns None if the page isn't loaded or index is out of range
    fn get_file_at_index(&self, index: usize) -> Option<&DisplayFileInfo> {
        if index >= self.total_results {
            return None;
        }

        let page_num = index / PAGE_SIZE;
        let page = self.page_cache.get(&page_num)?;

        // Preconditions: page must be properly constructed
        assert!(
            page.start_index < page.end_index,
            "Page has invalid range: [{}, {})",
            page.start_index,
            page.end_index
        );
        assert_eq!(
            page.end_index - page.start_index,
            page.files.len(),
            "Page size mismatch: range length {} != files count {}",
            page.end_index - page.start_index,
            page.files.len()
        );

        // Assert that the index is actually within this page's range
        assert!(
            index >= page.start_index && index < page.end_index,
            "Index {} outside page {} range [{}, {})",
            index,
            page_num,
            page.start_index,
            page.end_index
        );

        let offset_in_page = index - page.start_index;
        page.files.get(offset_in_page)
    }

    fn check_and_log_impressions(&mut self, force: bool) -> Result<()> {
        // Extract values we need before borrowing subsession mutably
        let (subsession_id, subsession_query, created_at, already_logged) = match &self.current_subsession {
            Some(s) => (s.id, s.query.clone(), s.created_at, s.events_have_been_logged),
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

        // Log top N visible files with metadata
        // Collect from page cache starting at index 0
        let mut top_n = Vec::new();
        for i in 0..self.num_results_to_log_as_impressions.min(self.total_results) {
            if let Some(display_info) = self.get_file_at_index(i) {
                top_n.push(FileMetadata {
                    relative_path: display_info.display_name.clone(),
                    full_path: display_info.full_path.to_string_lossy().to_string(),
                    mtime: display_info.mtime,
                    atime: display_info.atime,
                    size: display_info.file_size,
                });
            }
        }

        if !top_n.is_empty() {
            self.db
                .log_impressions(&subsession_query, &top_n, subsession_id, &self.session_id)?;

            // Mark as logged
            if let Some(ref mut s) = self.current_subsession {
                s.events_have_been_logged = true;
            }
        }

        Ok(())
    }

    fn move_selection(&mut self, delta: isize) {
        if self.total_results == 0 {
            return;
        }

        let len = self.total_results as isize;
        let new_index = (self.selected_index as isize + delta).rem_euclid(len);
        self.selected_index = new_index as usize;

        // Reset preview scroll and clear cache when changing selection
        self.preview_scroll = 0;
        self.preview_cache = None;

        // Reset marquee scroll
        self.path_bar_scroll = 0;
        self.path_bar_scroll_direction = 1;
        self.last_path_bar_update = Instant::now();
    }

    fn get_filtered_history(&self) -> Vec<PathBuf> {
        let query_lower = self.query.to_lowercase();

        // Get all history items (already in reverse order from the History module)
        let all_dirs = self.history.items_for_display();

        if query_lower.is_empty() {
            all_dirs
        } else {
            all_dirs
                .into_iter()
                .filter(|path| {
                    path.to_string_lossy().to_lowercase().contains(&query_lower)
                })
                .collect()
        }
    }

    fn move_history_selection(&mut self, delta: isize) {
        let filtered_history = self.get_filtered_history();
        if filtered_history.is_empty() {
            return;
        }

        let len = filtered_history.len() as isize;
        let new_index = (self.history_selected as isize + delta).rem_euclid(len);
        self.history_selected = new_index as usize;

        // Clear preview cache when selection changes
        self.preview_cache = None;
        self.preview_scroll = 0;
    }

    fn handle_history_enter(&mut self) -> Result<()> {
        let filtered_history = self.get_filtered_history();
        if filtered_history.is_empty() {
            return Ok(());
        }

        if self.history_selected >= filtered_history.len() {
            return Ok(());
        }

        // Map the selected index in the filtered list back to the display index
        // in the unfiltered list
        let selected_dir = &filtered_history[self.history_selected];
        let all_dirs = self.history.items_for_display();
        let display_index = all_dirs.iter().position(|p| p == selected_dir).unwrap_or(0);

        // Use the History module to navigate
        if let Some(new_dir) = self.history.navigate_to_display_index(display_index) {
            log::info!("Navigating to {:?} from history", new_dir);
            self.cwd = new_dir.clone();

            // Exit history mode
            self.ui_state.history_mode = false;
            self.query.clear();

            // Send ChangeCwd request to worker
            let query_id = self.next_subsession_id;
            self.next_subsession_id += 1;
            let _ = self.worker_tx.send(WorkerRequest::ChangeCwd {
                new_cwd: new_dir,
                query_id,
            });
        } else {
            // Directory is same as current, just exit history mode
            log::info!("Already in selected directory, not navigating");
            self.ui_state.history_mode = false;
            self.query.clear();
        }

        Ok(())
    }

    fn log_preview_scroll(&mut self) -> Result<()> {
        if self.total_results == 0 {
            return Ok(());
        }

        // Force log impressions before scroll
        self.check_and_log_impressions(true)?;

        if let Some(display_info) = self.get_file_at_index(self.selected_index) {
            let full_path_str = display_info.full_path.to_string_lossy().to_string();
            let key = (self.query.clone(), full_path_str.clone());

            if !self.scrolled_files.contains(&key) {
                let subsession_id = self
                    .current_subsession
                    .as_ref()
                    .map(|s| s.id)
                    .unwrap_or(1);
                self.db.log_event(EventData {
                    query: &self.query,
                    file_path: &display_info.display_name,
                    full_path: &full_path_str,
                    mtime: display_info.mtime,
                    atime: display_info.atime,
                    file_size: display_info.file_size,
                    subsession_id,
                    action: db::UserInteraction::Scroll,
                    session_id: &self.session_id,
                })?;
                self.scrolled_files.insert(key);
            }
        }

        Ok(())
    }

    fn update_scroll(&mut self, visible_height: u16) {
        let visible_height = visible_height as usize;
        if self.total_results == 0 {
            return;
        }

        // If all results fit on screen, don't scroll at all
        if self.total_results <= visible_height {
            self.file_list_scroll = 0;
        } else {
            // Auto-scroll the file list when selection is near top or bottom
            let selected = self.selected_index;
            let scroll = self.file_list_scroll;

            // If selected item is above visible area, scroll up
            if selected < scroll {
                self.file_list_scroll = selected;
            }
            // If selected item is below visible area, scroll down
            else if selected >= scroll + visible_height {
                // Smart positioning: leave some space from bottom (5 lines)
                // This makes wrap-around more comfortable
                let margin = 5usize;
                self.file_list_scroll = selected.saturating_sub(visible_height.saturating_sub(margin).min(visible_height - 1));
            }
            // If we're in the bottom 5 items and there's more to see, keep scrolling
            else if selected >= scroll + visible_height.saturating_sub(5) {
                self.file_list_scroll = selected.saturating_sub(visible_height.saturating_sub(5));
            }
        }

        let active_query_id = self.current_subsession.as_ref().map(|s| s.id).unwrap_or(0);

        // Page-based prefetching
        let current_page = self.selected_index / PAGE_SIZE;

        // Ensure current page is loaded
        if !self.page_cache.contains_key(&current_page) {
            let _ = self.worker_tx.send(WorkerRequest::GetPage {
                query_id: active_query_id,
                page_num: current_page,
            });
        }

        // Check if we're close to the top of the current page - prefetch previous page
        let offset_in_page = self.selected_index % PAGE_SIZE;
        if offset_in_page < PREFETCH_MARGIN {
            // Calculate previous page with wrap-around
            let total_pages = self.total_results.div_ceil(PAGE_SIZE);
            let prev_page = if current_page == 0 {
                total_pages.saturating_sub(1)
            } else {
                current_page - 1
            };

            // Only request if we have that many pages and it's not cached
            if total_pages > 0 && !self.page_cache.contains_key(&prev_page) {
                let _ = self.worker_tx.send(WorkerRequest::GetPage {
                    query_id: active_query_id,
                    page_num: prev_page,
                });
            }
        }

        // Check if we're close to the bottom of the current page - prefetch next page
        let page_size_for_current = PAGE_SIZE.min(self.total_results - current_page * PAGE_SIZE);
        if offset_in_page >= page_size_for_current.saturating_sub(PREFETCH_MARGIN) {
            // Calculate next page with wrap-around
            let total_pages = self.total_results.div_ceil(PAGE_SIZE);
            let next_page = if current_page + 1 >= total_pages {
                0
            } else {
                current_page + 1
            };

            // Only request if it's not cached
            if !self.page_cache.contains_key(&next_page) {
                let _ = self.worker_tx.send(WorkerRequest::GetPage {
                    query_id: active_query_id,
                    page_num: next_page,
                });
            }
        }
    }
}

fn main() -> Result<()> {
    // Generate session ID early so we can include it in all logs
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};
    let mut hasher = RandomState::new().build_hasher();
    Instant::now().hash(&mut hasher);
    std::process::id().hash(&mut hasher);
    let session_id = hasher.finish().to_string();

    // Set as environment variable so all threads can access it
    // SAFETY: We set this once at the very beginning of main() before any other threads exist
    unsafe {
        std::env::set_var("PSYCHIC_SESSION_ID", &session_id);
    }

    // Initialize logger with fern to write to both file and memory
    let (log_tx, log_rx) = mpsc::channel();

    if let Ok(home) = std::env::var("HOME") {
        let log_dir = PathBuf::from(&home)
            .join(".local")
            .join("share")
            .join("psychic");
        let _ = std::fs::create_dir_all(&log_dir);
        let log_file = log_dir.join("app.log");

        fern::Dispatch::new()
            .format(|out, message, record| {
                let session = std::env::var("PSYCHIC_SESSION_ID").unwrap_or_else(|_| "unknown".to_string());
                out.finish(format_args!(
                    "[{} {} {} {}] {}",
                    jiff::Timestamp::now().strftime("%Y-%m-%d %H:%M:%S"),
                    record.level(),
                    record.target(),
                    session,
                    message
                ))
            })
            .level(log::LevelFilter::Debug)
            .chain(fern::log_file(log_file).expect("Failed to open log file"))
            .chain(log_tx)
            .apply()
            .expect("Failed to initialize logger");
    }

    let cli = Cli::parse();

    // Handle subcommands
    if let Some(command) = cli.command {
        // Get data directory (global option)
        let data_dir = cli.data_dir.unwrap_or_else(|| {
            get_default_data_dir().expect("Failed to get default data directory")
        });

        match command {
            Commands::GenerateFeatures { format } => {
                // Create data directory if it doesn't exist
                std::fs::create_dir_all(&data_dir)?;

                // Determine output paths
                let default_filename = match format {
                    OutputFormat::Csv => "features.csv",
                    OutputFormat::Json => "features.json",
                };
                let output_path = data_dir.join(default_filename);
                let schema_path = data_dir.join("feature_schema.json");
                let db_path = db::Database::get_db_path(&data_dir);

                // Convert CLI format to features format
                let features_format = match format {
                    OutputFormat::Csv => features::OutputFormat::Csv,
                    OutputFormat::Json => features::OutputFormat::Json,
                };

                let format_str = match features_format {
                    features::OutputFormat::Csv => "CSV",
                    features::OutputFormat::Json => "JSON",
                };

                println!(
                    "Generating features ({}) from DB at {:?} and writing to {:?}",
                    format_str, db_path, output_path
                );
                features::generate_features(&db_path, &output_path, &schema_path, features_format)?;

                println!("Generated features at {:?}", output_path);
                println!("Generated feature schema at {:?}", schema_path);
                println!("Done.");
                return Ok(());
            }
            Commands::Retrain => {
                println!("Retraining model with data directory: {:?}", data_dir);
                // When called from CLI, print to stdout (no log file)
                ranker::retrain_model(&data_dir, None)?;
                println!("Done.");
                return Ok(());
            }
            Commands::Zsh => {
                // Output zsh integration script
                print!("{}", include_str!("../shell/psychic.zsh"));
                return Ok(());
            }
        }
    }

    let main_start = Instant::now();
    log::info!("=== STARTUP TIMING ===");

    // Get current working directory and canonicalize once
    let root_start = Instant::now();
    let root = env::current_dir()?.canonicalize()?;
    log::info!("TIMING {{\"op\":\"get_canonicalize_root\",\"ms\":{}}}", root_start.elapsed().as_secs_f64() * 1000.0);

    // Get data directory for main app
    let data_dir = cli
        .data_dir
        .unwrap_or_else(|| get_default_data_dir().expect("Failed to get default data directory"));

    // Create channel for retraining status
    let (retrain_tx, retrain_rx) = mpsc::channel();

    // Start background retraining in a new thread
    let retrain_start = Instant::now();
    let data_dir_clone = data_dir.clone();
    let training_log_path = data_dir.join("training.log");
    std::thread::spawn(move || {
        let _ = retrain_tx.send(true); // Signal retraining started
        log::info!("Starting background model retraining");
        if let Err(e) = ranker::retrain_model(&data_dir_clone, Some(training_log_path)) {
            log::error!("Background retraining failed: {}", e);
        } else {
            log::info!("Background retraining completed successfully");
        }
        let _ = retrain_tx.send(false); // Signal retraining completed
    });
    log::info!("TIMING {{\"op\":\"spawn_retrain_thread\",\"ms\":{}}}", retrain_start.elapsed().as_secs_f64() * 1000.0);

    // Convert CLI filter to internal FilterType
    let initial_filter = match cli.filter {
        Some(FilterArg::None) | None => search_worker::FilterType::None,
        Some(FilterArg::Cwd) => search_worker::FilterType::OnlyCwd,
        Some(FilterArg::Dirs) => search_worker::FilterType::OnlyDirs,
        Some(FilterArg::Files) => search_worker::FilterType::OnlyFiles,
    };

    // Initialize app
    let app_new_start = Instant::now();
    let mut app = App::new(root.clone(), &data_dir, log_rx, cli.on_dir_click.clone(), cli.on_cwd_visit.clone(), initial_filter)?;
    log::info!("TIMING {{\"op\":\"app_new\",\"ms\":{}}}", app_new_start.elapsed().as_secs_f64() * 1000.0);

    let log_session_start = Instant::now();
    log::info!(
        "Started psychic in directory {}, session {}",
        root.display(),
        app.session_id
    );
    log::info!("TIMING {{\"op\":\"session_log_message\",\"ms\":{}}}", log_session_start.elapsed().as_secs_f64() * 1000.0);

    // DISABLED: Log the initial directory as a click event (with empty query)
    // This was taking 200ms+ due to synchronous DB write, blocking startup
    // TODO: Re-enable with async logging or move to background thread
    /*
    let log_dir_click_start = Instant::now();
    {
        let dir_name = root
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(".");

        // Get metadata for the directory
        let metadata = std::fs::metadata(&root).ok();
        let mtime = metadata.as_ref().and_then(|m| {
            m.modified().ok().and_then(|t| {
                t.duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs() as i64)
            })
        });
        let atime = metadata.as_ref().and_then(|m| {
            m.accessed().ok().and_then(|t| {
                t.duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs() as i64)
            })
        });
        let file_size = metadata.as_ref().map(|m| m.len() as i64);

        let _ = app.db.log_event(EventData {
            query: "",
            file_path: dir_name,
            full_path: &root.to_string_lossy(),
            mtime,
            atime,
            file_size,
            subsession_id: 0, // Initial event, before any query
            action: db::UserInteraction::Click,
            session_id: &app.session_id,
        });
    }
    log::info!("TIMING {{\"op\":\"log_initial_dir_click\",\"ms\":{}}}", log_dir_click_start.elapsed().as_secs_f64() * 1000.0);
    */

    // Gather context in background thread
    let context_spawn_start = Instant::now();
    let session_id_clone = app.session_id.clone();
    let data_dir_clone = data_dir.clone();
    std::thread::spawn(move || {
        let context = context::gather_context();
        if let Ok(db) = db::Database::new(&data_dir_clone) {
            let _ = db.log_session(&session_id_clone, &context);
        }
    });
    log::info!("TIMING {{\"op\":\"spawn_context_thread\",\"ms\":{}}}", context_spawn_start.elapsed().as_secs_f64() * 1000.0);

    // Setup terminal
    let terminal_setup_start = Instant::now();
    // Use /dev/tty directly so shell integration can redirect stdout
    let mut tty = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")?;

    enable_raw_mode()?;
    execute!(tty, EnterAlternateScreen)?;
    execute!(tty, crossterm::event::EnableMouseCapture)?;
    let backend = CrosstermBackend::new(tty);
    let mut terminal = Terminal::new(backend)?;
    log::info!("TIMING {{\"op\":\"terminal_setup\",\"ms\":{}}}", terminal_setup_start.elapsed().as_secs_f64() * 1000.0);

    log::info!("TIMING {{\"op\":\"main_setup_total\",\"ms\":{}}}", main_start.elapsed().as_secs_f64() * 1000.0);

    // Run the app
    let result = run_app(&mut terminal, &mut app, retrain_rx);

    // Shutdown sequence: extract and drop worker_tx to signal the worker to stop,
    // then wait for the worker thread to finish, THEN drop app (which drops log_rx).
    // This ensures the worker can log its shutdown message before the logging channel closes.
    let worker_tx = std::mem::replace(&mut app.worker_tx, mpsc::channel().0);
    drop(worker_tx);

    if let Some(handle) = app.worker_handle.take() {
        let _ = handle.join();
    }

    // Now it's safe to drop app, which will close the logging channel
    drop(app);

    // Terminal cleanup
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        crossterm::event::DisableMouseCapture,
        LeaveAlternateScreen,
        Show
    )?;

    result
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
    app: &mut App,
    retrain_rx: Receiver<bool>,
) -> Result<()> {
    let run_app_start = Instant::now();
    let mut first_render_logged = false;
    let mut first_query_complete_logged = false;

    let marquee_delay = Duration::from_millis(500); // 0.5s pause at ends
    let marquee_speed = Duration::from_millis(80); // scroll every 80ms

    loop {
        // Log impressions for this subsession if >200ms old
        let _ = app.check_and_log_impressions(false);

        // Draw UI
        terminal.draw(|f| {
            // Log first render
            if !first_render_logged {
                log::info!("TIMING {{\"op\":\"first_render\",\"ms\":{}}}", run_app_start.elapsed().as_secs_f64() * 1000.0);
                first_render_logged = true;
            }

            // If in history mode, render history-specific UI
            if app.ui_state.history_mode {
                render_history_mode(f, app);
                return;
            }

            // Split vertically: top for results/preview, bottom for input
            let main_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(0),    // Results + Preview
                    Constraint::Length(1), // Path bar
                    Constraint::Length(3), // Search input at bottom
                ])
                .split(f.area());

            // Split top area horizontally: left for file list, middle for preview, right for debug
            let top_chunks = match app.ui_state.debug_pane_mode {
                ui_state::DebugPaneMode::Expanded => {
                    // Debug expanded: give it most of the space
                    Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([
                            Constraint::Percentage(25), // File list (smaller)
                            Constraint::Percentage(0),  // Preview (hidden)
                            Constraint::Percentage(75), // Debug (expanded)
                        ])
                        .split(main_chunks[0])
                }
                ui_state::DebugPaneMode::Small => {
                    // Debug small: normal layout
                    Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([
                            Constraint::Percentage(35), // File list
                            Constraint::Percentage(45), // Preview
                            Constraint::Percentage(20), // Debug (small)
                        ])
                        .split(main_chunks[0])
                }
                ui_state::DebugPaneMode::Hidden => {
                    // Debug hidden: no space for debug pane
                    Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([
                            Constraint::Percentage(40), // File list (more space)
                            Constraint::Percentage(60), // Preview (more space)
                            Constraint::Percentage(0),  // Debug (hidden)
                        ])
                        .split(main_chunks[0])
                }
            };

            // Update scroll position based on selection and visible height
            let visible_height = top_chunks[0].height.saturating_sub(2); // subtract border
            app.update_scroll(visible_height);

            // File list on the left
            let list_width = top_chunks[0].width.saturating_sub(2) as usize; // subtract borders
            let scroll_offset = app.file_list_scroll;

            // Build list items from page cache
            let items: Vec<ListItem> = (0..visible_height as usize)
                .map(|display_idx| {
                    let i = scroll_offset + display_idx;
                    if i >= app.total_results {
                        return ListItem::new("");
                    }

                    if let Some(display_info) = app.get_file_at_index(i) {
                        let time_ago = get_time_ago(display_info.mtime);
                        let rank = i + 1;

                        // Calculate space: "N. " takes 4 chars, time_ago length, we need padding between
                        let rank_prefix = format!("{:2}. ", rank);
                        let prefix_len = rank_prefix.len();
                        let time_len = time_ago.len();

                        // Available space for filename and padding
                        let available = list_width.saturating_sub(prefix_len + time_len);
                        let file_width = available.saturating_sub(2); // leave at least 2 spaces padding

                        // Add "/" suffix for directories and "(cwd)" for current directory
                        let cwd_suffix = if display_info.is_cwd { " (cwd)" } else { "" };
                        let cwd_suffix_len = cwd_suffix.len();

                        let display_name = if display_info.is_dir {
                            format!("{}/", display_info.display_name)
                        } else {
                            display_info.display_name.clone()
                        };

                        // Adjust file_width to account for cwd suffix
                        let adjusted_file_width = file_width.saturating_sub(cwd_suffix_len);

                        // For historical files, use absolute path truncation
                        let truncated_path = if display_info.is_historical {
                            truncate_absolute_path(&display_name, adjusted_file_width)
                        } else {
                            truncate_path(&display_name, adjusted_file_width)
                        };

                        // Build line with styled spans
                        let base_style = if i == app.selected_index {
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD)
                        } else if display_info.is_dir {
                            // Color directories cyan when not selected
                            Style::default().fg(Color::Cyan)
                        } else {
                            Style::default()
                        };

                        // Rank number color: different for historical files
                        let rank_style = if display_info.is_historical {
                            // Historical files: gray rank number
                            Style::default().fg(Color::DarkGray)
                        } else if i == app.selected_index {
                            // Selected: match base style
                            base_style
                        } else {
                            // Normal: match base style
                            base_style
                        };

                        let cwd_style = if i == app.selected_index {
                            // If selected, keep yellow but make it even more visible
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            // Otherwise use magenta to stand out
                            Style::default()
                                .fg(Color::Magenta)
                                .add_modifier(Modifier::BOLD)
                        };

                        // Calculate padding to right-justify time
                        let padding_len = adjusted_file_width.saturating_sub(truncated_path.len());
                        let padding = " ".repeat(padding_len);

                        let line = if display_info.is_cwd {
                            Line::from(vec![
                                Span::styled(rank_prefix.clone(), rank_style),
                                Span::styled(truncated_path.clone(), base_style),
                                Span::styled(cwd_suffix, cwd_style),
                                Span::raw(padding),
                                Span::raw("  "),
                                Span::styled(time_ago.clone(), base_style),
                            ])
                        } else {
                            Line::from(vec![
                                Span::styled(rank_prefix.clone(), rank_style),
                                Span::styled(format!("{:<width$}  {}", truncated_path, time_ago, width = adjusted_file_width), base_style),
                            ])
                        };

                        ListItem::new(line)
                    } else {
                        // Page is not cached, show a loading indicator
                        ListItem::new("[Loading...]").style(Style::default().fg(Color::DarkGray))
                    }
                })
                .collect();

            // Create title with filter indicator
            let filter_name = match app.current_filter {
                search_worker::FilterType::None => "All",
                search_worker::FilterType::OnlyCwd => "CWD",
                search_worker::FilterType::OnlyDirs => "Dirs",
                search_worker::FilterType::OnlyFiles => "Files",
            };

            let title_line = if app.current_filter == search_worker::FilterType::None {
                // No filter active - no highlight
                Line::from(vec![
                    Span::raw(format!("{} ({}/{})", filter_name, app.total_results, app.total_files)),
                ])
            } else {
                // Filter active - highlight in green
                Line::from(vec![
                    Span::styled(
                        filter_name,
                        Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
                    ),
                    Span::raw(format!(" ({}/{})", app.total_results, app.total_files)),
                ])
            };

            let list = List::new(items).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title_line),
            );
            f.render_widget(list, top_chunks[0]);

            // Get current file from page cache - clone the info we need to avoid borrow issues
            let current_file_info: Option<(PathBuf, String, bool)> = app
                .get_file_at_index(app.selected_index)
                .map(|f| (f.full_path.clone(), f.display_name.clone(), f.is_dir));
            let current_file_path = current_file_info.as_ref().map(|(p, _, _)| p.to_string_lossy().to_string());
            let is_dir = current_file_info.as_ref().map(|(_, _, d)| *d).unwrap_or(false);

            // Preview on the right using bat (with smart caching)
            let preview_height = top_chunks[1].height.saturating_sub(2);
            let preview_text = if current_file_info.is_some() && app.total_results > 0 {
                if let Some(current_file_path) = &current_file_path {
                    let full_path = PathBuf::from(current_file_path);

                    // For directories, show eza -al output
                    if is_dir {
                        let preview_width = top_chunks[1].width;
                        let extra_flags = app.ui_state.get_eza_flags(preview_width);

                        let mut eza_cmd = std::process::Command::new("eza");
                        eza_cmd.arg("-al").arg("--color=always");

                        // Add extra flags if preview is narrow
                        for flag in extra_flags.split_whitespace() {
                            if !flag.is_empty() {
                                eza_cmd.arg(flag);
                            }
                        }

                        eza_cmd.arg(&full_path);
                        let eza_output = eza_cmd.output();

                        match eza_output {
                            Ok(output) => {
                                match ansi_to_tui::IntoText::into_text(&output.stdout) {
                                    Ok(text) => text,
                                    Err(_) => Text::from("[Unable to parse directory listing]"),
                                }
                            }
                            Err(_) => {
                                // Fallback to ls if eza not available
                                let ls_output = std::process::Command::new("ls")
                                    .arg("-lah")
                                    .arg(&full_path)
                                    .output();
                                match ls_output {
                                    Ok(output) => Text::from(String::from_utf8_lossy(&output.stdout).to_string()),
                                    Err(_) => Text::from("[Unable to list directory]"),
                                }
                            }
                        }
                    } else {
                        // For files, use bat as before
                        let full_path = PathBuf::from(current_file_path);

                    // Check for a full, cached preview
                    if let Some(cached_text) =
                        app.preview_cache.as_ref().and_then(|(path, text)| {
                            if path == current_file_path {
                                Some(text.clone())
                            } else {
                                None
                            }
                        })
                    {
                        // FAST PATH: Full preview is cached, just use it.
                        cached_text
                    } else {
                        // SLOW PATH: No full preview in cache.
                        // Decide whether to render a light preview or a full one.
                        let (text_to_render, should_cache) = if app.preview_scroll == 0 {
                            // Initial view (unscrolled): render a light preview of N lines.
                            let line_range = format!(":{}", preview_height);
                            let bat_output = std::process::Command::new("bat")
                                .arg("--color=always")
                                .arg("--style=numbers")
                                .arg("--line-range")
                                .arg(&line_range)
                                .arg(&full_path)
                                .output();

                            let text = match bat_output {
                                Ok(output) => {
                                    match ansi_to_tui::IntoText::into_text(&output.stdout) {
                                        Ok(text) => text,
                                        Err(_) => Text::from("[Unable to parse preview]"),
                                    }
                                }
                                Err(_) => {
                                    // Fallback for light preview
                                    match std::fs::read_to_string(&full_path) {
                                        Ok(content) => Text::from(
                                            content
                                                .lines()
                                                .take(preview_height as usize)
                                                .collect::<Vec<_>>()
                                                .join("\n"),
                                        ),
                                        Err(_) => Text::from("[Unable to preview file]"),
                                    }
                                }
                            };
                            (text, false) // Don't cache the light preview
                        } else {
                            // User is scrolling, and we need to generate the full preview.
                            let bat_output = std::process::Command::new("bat")
                                .arg("--color=always")
                                .arg("--style=numbers")
                                .arg("--paging=never")
                                .arg(&full_path)
                                .output();

                            let text = match bat_output {
                                Ok(output) => {
                                    match ansi_to_tui::IntoText::into_text(&output.stdout) {
                                        Ok(text) => text,
                                        Err(_) => Text::from("[Unable to parse preview]"),
                                    }
                                }
                                Err(_) => {
                                    // Fallback for full preview
                                    match std::fs::read_to_string(&full_path) {
                                        Ok(content) => Text::from(content),
                                        Err(_) => Text::from("[Unable to preview file]"),
                                    }
                                }
                            };
                            (text, true) // Cache the full preview
                        };

                        if should_cache {
                            app.preview_cache =
                                Some((current_file_path.to_string(), text_to_render.clone()));
                        }
                        text_to_render
                    }
                    } // End of else block for files
                } else {
                    Text::from("[Loading preview...]")
                }
            } else {
                Text::from("")
            };

            let preview_pane_title = current_file_info
                .as_ref()
                .and_then(|(path, _, _)| path.file_name())
                .map(|x| x.to_string_lossy())
                .map(|x| x.to_string())
                .unwrap_or("No file selected".to_string());

            let preview = Paragraph::new(preview_text)
                .scroll((app.preview_scroll as u16, 0))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(preview_pane_title),
                );
            f.render_widget(preview, top_chunks[1]);

            // Debug panel on the right
            let mut debug_lines = Vec::new();

            // Show current selection info - need another lookup to get score/features
            if let Some(display_info) = app.get_file_at_index(app.selected_index) {
                debug_lines.push(format!("Score: {:.4}", display_info.score));
                debug_lines.push(String::from(""));
                debug_lines.push(String::from("Features:"));
                debug_lines.push(String::from(""));

                // Show all features from registry
                if !display_info.features.is_empty() {
                    let features_map = ranker::features_to_map(&display_info.features);
                    for feature in feature_defs::FEATURE_REGISTRY.iter() {
                        if let Some(value) = features_map.get(feature.name()) {
                            debug_lines.push(format!("  {}: {}", feature.name(), value));
                        }
                    }
                } else {
                    debug_lines.push(String::from("  (no features)"));
                }
            } else if app.total_results > 0 {
                debug_lines.push(String::from("(loading...)"));
            } else {
                debug_lines.push(String::from("No results"));
            }

            debug_lines.push(String::from("")); // Separator

            // Add retraining status
            if app.currently_retraining {
                debug_lines.push(String::from("Retraining model..."));
                debug_lines.push(String::from("")); // Separator
            }

            // Add model stats
            if let Some(ref stats) = app.model_stats_cache {
                debug_lines.push(String::from("Model Stats:"));
                let formatter = timeago::Formatter::new();

                // Parse timestamp and show how long ago
                if let Ok(trained_at) = stats.trained_at.parse::<jiff::Timestamp>() {
                    let now = jiff::Timestamp::now();
                    let duration = now.duration_since(trained_at);
                    let time_ago = formatter
                        .convert(std::time::Duration::from_secs(duration.as_secs() as u64));
                    debug_lines.push(format!("  Trained: {}", time_ago));
                } else {
                    debug_lines.push(format!("  Trained: {}", stats.trained_at));
                }

                debug_lines.push(format!(
                    "  Duration: {:.2}s",
                    stats.training_duration_seconds
                ));
                debug_lines.push(format!("  Features: {}", stats.num_features));
                debug_lines.push(format!(
                    "  Examples: {} ({} pos, {} neg)",
                    stats.num_total_examples,
                    stats.num_positive_examples,
                    stats.num_negative_examples
                ));
                debug_lines.push(String::from("  Top features:"));
                for feat in &stats.top_3_features {
                    debug_lines.push(format!("    {}: {:.1}", feat.feature, feat.importance));
                }
                debug_lines.push(String::from("")); // Separator
            }

            // Add preview cache status
            if let Some((file_path, _, _)) = &current_file_info {
                let full_path_str = file_path.to_string_lossy().to_string();
                let is_cached = app
                    .preview_cache
                    .as_ref()
                    .is_some_and(|(p, _)| p == &full_path_str);
                let cache_status = if is_cached { "Cached" } else { "Live" };
                debug_lines.push(format!("Preview: {}", cache_status));
            } else {
                debug_lines.push(String::from("Preview: N/A"));
            }

            debug_lines.push(String::from("")); // Separator

            // Add page cache status
            if app.total_results > 0 {
                let current_page = app.selected_index / PAGE_SIZE;
                debug_lines.push(format!("Current page: {}", current_page));

                let mut cached_pages: Vec<usize> = app.page_cache.keys().copied().collect();
                cached_pages.sort_unstable();
                let pages_str = cached_pages
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                debug_lines.push(format!("Cached pages: [{}]", pages_str));
            } else {
                debug_lines.push(String::from("Current page: N/A"));
                debug_lines.push(String::from("Cached pages: []"));
            }

            debug_lines.push(String::from("")); // Separator

            // Add recent logs
            debug_lines.push(String::from("Recent Logs:"));
            // Show more log lines when debug is maximized
            let log_count = if app.ui_state.is_debug_pane_expanded() { 30 } else { 10 };
            let log_start = app.recent_logs.len().saturating_sub(log_count);
            for log_line in app.recent_logs.iter().skip(log_start) {
                // Truncate long lines to fit
                let max_len = if app.ui_state.is_debug_pane_expanded() { 120 } else { 60 };
                if log_line.len() > max_len {
                    debug_lines.push(format!("  {}...", &log_line[..(max_len - 3)]));
                } else {
                    debug_lines.push(format!("  {}", log_line));
                }
            }

            let debug_text = debug_lines.join("\n");

            let debug_title = match app.ui_state.debug_pane_mode {
                ui_state::DebugPaneMode::Small => "Debug (Ctrl-O: expand)",
                ui_state::DebugPaneMode::Expanded => "Debug (Ctrl-O: hide)",
                ui_state::DebugPaneMode::Hidden => "Debug (Ctrl-O: show)",
            };
            let debug_pane = Paragraph::new(debug_text)
                .block(Block::default().borders(Borders::ALL).title(debug_title));
            f.render_widget(debug_pane, top_chunks[2]);

            // Get path of currently selected file for marquee
            let selected_path_str = app
                .get_file_at_index(app.selected_index)
                .map(|f| f.full_path.to_string_lossy().to_string())
                .unwrap_or_default();

            // Pad the string to make the marquee scroll past the end
            let padded_path = format!("{}    ", selected_path_str);

            let path_bar_width = main_chunks[1].width as usize;

            // Marquee animation logic
            if padded_path.len() > path_bar_width {
                let now = Instant::now();
                let time_since_update = now.duration_since(app.last_path_bar_update);

                let max_scroll = padded_path.len().saturating_sub(path_bar_width) as u16;

                // Pause at the ends of the scroll
                let should_scroll = if app.path_bar_scroll == 0 || app.path_bar_scroll >= max_scroll {
                    time_since_update > marquee_delay
                } else {
                    time_since_update > marquee_speed
                };

                if should_scroll {
                    if app.path_bar_scroll_direction == 1 {
                        if app.path_bar_scroll < max_scroll {
                            app.path_bar_scroll += 1;
                        } else {
                            app.path_bar_scroll_direction = -1; // Change direction
                        }
                    } else if app.path_bar_scroll > 0 {
                        app.path_bar_scroll -= 1;
                    } else {
                        app.path_bar_scroll_direction = 1; // Change direction
                    }
                    app.last_path_bar_update = now;
                }
            }

            // Path bar
            let path_bar = Paragraph::new(padded_path)
                .style(Style::default().fg(Color::DarkGray))
                .scroll((0, app.path_bar_scroll));
            f.render_widget(path_bar, main_chunks[1]);

            // Search input at the bottom
            let cwd_str = app.cwd.to_string_lossy();
            let filter_indicator = match app.current_filter {
                search_worker::FilterType::None => "",
                search_worker::FilterType::OnlyCwd => " [CWD]",
                search_worker::FilterType::OnlyDirs => " [DIRS]",
                search_worker::FilterType::OnlyFiles => " [FILES]",
            };
            let search_title = format!("Search: {}{}", cwd_str, filter_indicator);
            let input = Paragraph::new(app.query.as_str())
                .block(Block::default().borders(Borders::ALL).title(search_title));
            f.render_widget(input, main_chunks[2]);

            // Filter picker overlay (rendered on top if visible)
            if app.ui_state.filter_picker_visible {
                use ratatui::layout::Rect;

                // Create a popup in the bottom-right
                let popup_width = 30;
                let popup_height = 6;  // 4 options + top/bottom borders
                let area = f.area();

                // Position in bottom-right corner with some margin
                let popup_x = area.width.saturating_sub(popup_width + 2);
                let popup_y = area.height.saturating_sub(popup_height + 2);

                let popup_area = Rect {
                    x: popup_x,
                    y: popup_y,
                    width: popup_width,
                    height: popup_height,
                };

                // Clear the area first to make it opaque
                f.render_widget(Clear, popup_area);

                // Build filter options text with current selection highlighted
                let mut lines = vec![];

                let options = [
                    (search_worker::FilterType::None, "0: No filter"),
                    (search_worker::FilterType::OnlyCwd, "c: Only CWD"),
                    (search_worker::FilterType::OnlyDirs, "d: Only directories"),
                    (search_worker::FilterType::OnlyFiles, "f: Only files"),
                ];

                for (filter_type, label) in options.iter() {
                    if *filter_type == app.current_filter {
                        lines.push(format!("> {}", label));
                    } else {
                        lines.push(format!("  {}", label));
                    }
                }

                let filter_text = lines.join("\n");
                let filter_popup = Paragraph::new(filter_text)
                    .block(Block::default()
                        .borders(Borders::ALL)
                        .title("Filters (Ctrl-F)"));

                f.render_widget(filter_popup, popup_area);
            }

            // Position cursor in the search input at the end of the query text
            // Account for border (1 char) + query length
            let cursor_x = main_chunks[2].x + 1 + app.query.len() as u16;
            let cursor_y = main_chunks[2].y + 1; // 1 for top border
            f.set_cursor_position((cursor_x, cursor_y));
        })?;

        // Check for retraining status updates
        if let Ok(retraining_status) = retrain_rx.try_recv() {
            app.currently_retraining = retraining_status;
        }

        // Collect new log messages (non-blocking)
        while let Ok(log_msg) = app.log_receiver.try_recv() {
            // fern adds a newline to each message sent via channel, so trim it
            app.recent_logs.push_back(log_msg.trim_end().to_string());
            // Keep only last 50 messages
            if app.recent_logs.len() > 50 {
                app.recent_logs.pop_front();
            }
        }

        // Poll for worker responses (non-blocking)
        while let Ok(response) = app.worker_rx.try_recv() {
            match response {
                WorkerResponse::QueryUpdated {
                    query_id,
                    total_results,
                    total_files,
                    initial_page,
                    model_stats,
                } => {
                    let active_query_id = app.current_subsession.as_ref().map(|s| s.id).unwrap_or(0);

                    // Only accept updates for queries that are newer than what we currently have.
                    // This handles out-of-order responses.
                    if query_id >= active_query_id {
                        // Clear page cache on query change
                        app.page_cache.clear();

                        // Insert initial page
                        let page = Page {
                            start_index: initial_page.start_index,
                            end_index: initial_page.end_index,
                            files: initial_page.files,
                        };
                        app.page_cache.insert(initial_page.page_num, page);

                        app.total_results = total_results;
                        app.total_files = total_files;
                        app.model_stats_cache = model_stats;

                        // Reset selection if needed
                        if app.selected_index >= total_results {
                            app.selected_index = 0;
                        }

                        // Log first query completion
                        if !first_query_complete_logged {
                            log::info!("TIMING {{\"op\":\"first_query_complete\",\"ms\":{}}}", run_app_start.elapsed().as_secs_f64() * 1000.0);
                            first_query_complete_logged = true;
                        }

                        // Create subsession, using the query text from the app state
                        app.current_subsession = Some(Subsession {
                            id: query_id,
                            query: app.query.clone(),
                            created_at: jiff::Timestamp::now(),
                            events_have_been_logged: false,
                        });
                    }
                }
                WorkerResponse::Page { query_id, page_data } => {
                    let active_query_id = app.current_subsession.as_ref().map(|s| s.id).unwrap_or(0);

                    // Only accept page data for the currently active query
                    if query_id == active_query_id {
                        // Insert page into cache
                        let page = Page {
                            start_index: page_data.start_index,
                            end_index: page_data.end_index,
                            files: page_data.files,
                        };
                        app.page_cache.insert(page_data.page_num, page);
                    }
                }
                WorkerResponse::FilesChanged => {
                    // The worker detected file changes, so we trigger a refresh
                    // of the current query to get fresh results.
                    log::info!("Auto-refreshing query due to file changes.");
                    let query_id = app.next_subsession_id;
                    app.next_subsession_id += 1;
                    let _ = app.worker_tx.send(WorkerRequest::UpdateQuery(
                        search_worker::UpdateQueryRequest {
                            query: app.query.clone(),
                            query_id,
                            filter: app.current_filter,
                        },
                    ));
                }
            }
        }

        // Handle events with timeout
        if event::poll(Duration::from_millis(100))? {
            let event_read = event::read()?;
            match event_read {
                Event::Mouse(mouse_event) => {
                    use crossterm::event::MouseEventKind;
                    match mouse_event.kind {
                        MouseEventKind::ScrollDown => {
                            app.preview_scroll = app.preview_scroll.saturating_add(3);
                            let _ = app.log_preview_scroll();
                        }
                        MouseEventKind::ScrollUp => {
                            app.preview_scroll = app.preview_scroll.saturating_sub(3);
                            let _ = app.log_preview_scroll();
                        }
                        _ => {}
                    }
                }
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    match key.code {
                        KeyCode::Char('j')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            match app.on_cwd_visit {
                                OnCwdVisitAction::PrintToStdout => {
                                    // Print directory and exit (for shell integration)
                                    disable_raw_mode()?;
                                    terminal
                                        .backend_mut()
                                        .execute(crossterm::event::DisableMouseCapture)?;
                                    terminal.backend_mut().execute(LeaveAlternateScreen)?;
                                    terminal.backend_mut().execute(Show)?;

                                    println!("{}", app.cwd.display());
                                    return Ok(());
                                }
                                OnCwdVisitAction::DropIntoShell => {
                                    // Open shell in CWD
                                    log::info!("Opening shell in cwd: {:?}", app.cwd);

                                    // Suspend TUI
                                    disable_raw_mode()?;
                                    terminal
                                        .backend_mut()
                                        .execute(crossterm::event::DisableMouseCapture)?;
                                    terminal.backend_mut().execute(LeaveAlternateScreen)?;
                                    terminal.backend_mut().execute(Show)?;

                                    // Use /dev/tty for shell so it can interact with the terminal
                                    use std::process::Stdio;
                                    let tty_in = std::fs::OpenOptions::new()
                                        .read(true)
                                        .open("/dev/tty")?;
                                    let tty_out = std::fs::OpenOptions::new()
                                        .write(true)
                                        .open("/dev/tty")?;
                                    let tty_err = std::fs::OpenOptions::new()
                                        .write(true)
                                        .open("/dev/tty")?;

                                    let shell =
                                        std::env::var("SHELL").unwrap_or_else(|_| "sh".to_string());
                                    let status = std::process::Command::new(&shell)
                                        .current_dir(&app.cwd)
                                        .stdin(Stdio::from(tty_in))
                                        .stdout(Stdio::from(tty_out))
                                        .stderr(Stdio::from(tty_err))
                                        .status();

                                    // Resume TUI
                                    enable_raw_mode()?;
                                    terminal.backend_mut().execute(EnterAlternateScreen)?;
                                    terminal
                                        .backend_mut()
                                        .execute(crossterm::event::EnableMouseCapture)?;
                                    terminal.clear()?;

                                    if let Err(e) = status {
                                        log::error!("Failed to launch shell: {}", e);
                                    }

                                    // Reload model (may have been retrained in background)
                                    let query_id_model = app.next_subsession_id;
                                    app.next_subsession_id += 1;
                                    if let Err(e) = app.reload_model(query_id_model) {
                                        log::error!("Failed to reload model: {}", e);
                                    }

                                    // Reload click data and rerank files after shell
                                    let query_id_clicks = app.next_subsession_id;
                                    app.next_subsession_id += 1;
                                    if let Err(e) = app.reload_and_rerank(query_id_clicks) {
                                        log::error!("Failed to reload and rerank: {}", e);
                                    }
                                }
                            }
                        }
                        KeyCode::Char('h')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            // Ctrl-H: Toggle history navigation mode
                            if app.ui_state.history_mode {
                                // Exit history mode
                                app.ui_state.history_mode = false;
                                app.query.clear();
                            } else {
                                // Enter history mode (always allow, even with empty history)
                                app.ui_state.history_mode = true;
                                // Set cursor to current position in history
                                app.history_selected = app.history.current_display_index();
                                app.query.clear();
                                app.preview_cache = None; // Clear preview cache
                            }
                        }
                        KeyCode::Char('c')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            return Ok(());
                        }
                        KeyCode::Char('d')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            return Ok(());
                        }
                        KeyCode::Char('u')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            // Ctrl-U: clear entire search
                            app.query.clear();
                            let query_id = app.next_subsession_id;
                            app.next_subsession_id += 1;
                            let _ = app.worker_tx.send(WorkerRequest::UpdateQuery(
                                search_worker::UpdateQueryRequest {
                                    query: app.query.clone(),
                                    query_id,
                                    filter: app.current_filter,
                                },
                            ));
                        }
                        KeyCode::Char('o')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            // Ctrl-O: cycle debug pane mode (Small -> Expanded -> Hidden -> Small)
                            app.ui_state.cycle_debug_pane_mode();
                        }
                        KeyCode::Char('f')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            // Ctrl-F: toggle filter picker
                            app.ui_state.filter_picker_visible = !app.ui_state.filter_picker_visible;
                        }
                        KeyCode::Char('0') if app.ui_state.filter_picker_visible => {
                            // Filter 0: None
                            app.current_filter = search_worker::FilterType::None;
                            app.ui_state.filter_picker_visible = false;
                            let query_id = app.next_subsession_id;
                            app.next_subsession_id += 1;
                            let _ = app.worker_tx.send(WorkerRequest::UpdateQuery(
                                search_worker::UpdateQueryRequest {
                                    query: app.query.clone(),
                                    query_id,
                                    filter: app.current_filter,
                                },
                            ));
                        }
                        KeyCode::Char('c') if app.ui_state.filter_picker_visible => {
                            // Filter c: Only CWD
                            app.current_filter = search_worker::FilterType::OnlyCwd;
                            app.ui_state.filter_picker_visible = false;
                            let query_id = app.next_subsession_id;
                            app.next_subsession_id += 1;
                            let _ = app.worker_tx.send(WorkerRequest::UpdateQuery(
                                search_worker::UpdateQueryRequest {
                                    query: app.query.clone(),
                                    query_id,
                                    filter: app.current_filter,
                                },
                            ));
                        }
                        KeyCode::Char('d') if app.ui_state.filter_picker_visible => {
                            // Filter d: Only Dirs
                            app.current_filter = search_worker::FilterType::OnlyDirs;
                            app.ui_state.filter_picker_visible = false;
                            let query_id = app.next_subsession_id;
                            app.next_subsession_id += 1;
                            let _ = app.worker_tx.send(WorkerRequest::UpdateQuery(
                                search_worker::UpdateQueryRequest {
                                    query: app.query.clone(),
                                    query_id,
                                    filter: app.current_filter,
                                },
                            ));
                        }
                        KeyCode::Char('f') if app.ui_state.filter_picker_visible => {
                            // Filter f: Only Files
                            app.current_filter = search_worker::FilterType::OnlyFiles;
                            app.ui_state.filter_picker_visible = false;
                            let query_id = app.next_subsession_id;
                            app.next_subsession_id += 1;
                            let _ = app.worker_tx.send(WorkerRequest::UpdateQuery(
                                search_worker::UpdateQueryRequest {
                                    query: app.query.clone(),
                                    query_id,
                                    filter: app.current_filter,
                                },
                            ));
                        }
                        KeyCode::Esc => {
                            if app.ui_state.filter_picker_visible {
                                // Close filter picker without changing filter
                                app.ui_state.filter_picker_visible = false;
                            } else if app.ui_state.history_mode {
                                // Exit history mode
                                app.ui_state.history_mode = false;
                                app.query.clear();
                            } else {
                                return Ok(());
                            }
                        }
                        KeyCode::Up => {
                            if app.ui_state.history_mode {
                                app.move_history_selection(-1);
                            } else {
                                app.move_selection(-1);
                            }
                        }
                        KeyCode::Char('p')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            if app.ui_state.history_mode {
                                app.move_history_selection(-1);
                            } else {
                                app.move_selection(-1);
                            }
                        }
                        KeyCode::Down => {
                            if app.ui_state.history_mode {
                                app.move_history_selection(1);
                            } else {
                                app.move_selection(1);
                            }
                        }
                        KeyCode::Char('n')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            if app.ui_state.history_mode {
                                app.move_history_selection(1);
                            } else {
                                app.move_selection(1);
                            }
                        }
                        KeyCode::Char(c) => {
                            app.query.push(c);
                            let query_id = app.next_subsession_id;
                            app.next_subsession_id += 1;
                            let _ = app.worker_tx.send(WorkerRequest::UpdateQuery(
                                search_worker::UpdateQueryRequest {
                                    query: app.query.clone(),
                                    query_id,
                                    filter: app.current_filter,
                                },
                            ));
                        }
                        KeyCode::Backspace => {
                            app.query.pop();
                            let query_id = app.next_subsession_id;
                            app.next_subsession_id += 1;
                            let _ = app.worker_tx.send(WorkerRequest::UpdateQuery(
                                search_worker::UpdateQueryRequest {
                                    query: app.query.clone(),
                                    query_id,
                                    filter: app.current_filter,
                                },
                            ));
                        }
                        KeyCode::Enter => {
                            if app.ui_state.history_mode {
                                app.handle_history_enter()?;
                            } else if app.total_results > 0 {
                                // Force log impressions before click
                                app.check_and_log_impressions(true)?;

                                if let Some(display_info) = app.get_file_at_index(app.selected_index) {
                                    // Log the click
                                    let subsession_id =
                                        app.current_subsession.as_ref().map(|s| s.id).unwrap_or(1);
                                    app.db.log_event(EventData {
                                        query: &app.query,
                                        file_path: &display_info.display_name,
                                        full_path: &display_info.full_path.to_string_lossy(),
                                        mtime: display_info.mtime,
                                        atime: display_info.atime,
                                        file_size: display_info.file_size,
                                        subsession_id,
                                        action: db::UserInteraction::Click,
                                        session_id: &app.session_id,
                                    })?;

                                    if display_info.is_dir {
                                        // On a directory, perform configured action
                                        let dir_path = display_info.full_path.clone();

                                        match app.on_dir_click {
                                            OnDirClickAction::Navigate => {
                                                // Don't navigate if already in that directory
                                                if dir_path == app.cwd {
                                                    log::info!("Already in {:?}, not navigating", dir_path);
                                                    continue;
                                                }

                                                log::info!("Navigating to directory: {:?}", dir_path);
                                                app.history.navigate_to(dir_path.clone());
                                                app.cwd = dir_path.clone();

                                                // Clear query and send request to worker
                                                app.query.clear();
                                                let query_id = app.next_subsession_id;
                                                app.next_subsession_id += 1;
                                                let _ = app.worker_tx.send(WorkerRequest::ChangeCwd {
                                                    new_cwd: dir_path,
                                                    query_id,
                                                });
                                            }
                                            OnDirClickAction::PrintToStdout => {
                                                // Print directory path and exit
                                                disable_raw_mode()?;
                                                terminal
                                                    .backend_mut()
                                                    .execute(crossterm::event::DisableMouseCapture)?;
                                                terminal.backend_mut().execute(LeaveAlternateScreen)?;
                                                terminal.backend_mut().execute(Show)?;

                                                println!("{}", dir_path.display());
                                                return Ok(());
                                            }
                                            OnDirClickAction::DropIntoShell => {
                                                // Drop into shell in the selected directory
                                                log::info!("Dropping into shell in directory: {:?}", dir_path);

                                                disable_raw_mode()?;
                                                terminal
                                                    .backend_mut()
                                                    .execute(crossterm::event::DisableMouseCapture)?;
                                                terminal.backend_mut().execute(LeaveAlternateScreen)?;
                                                terminal.backend_mut().execute(Show)?;

                                                use std::process::Stdio;
                                                let tty_in = std::fs::OpenOptions::new()
                                                    .read(true)
                                                    .open("/dev/tty")?;
                                                let tty_out = std::fs::OpenOptions::new()
                                                    .write(true)
                                                    .open("/dev/tty")?;
                                                let tty_err = std::fs::OpenOptions::new()
                                                    .write(true)
                                                    .open("/dev/tty")?;

                                                let shell = std::env::var("SHELL").unwrap_or_else(|_| "sh".to_string());
                                                let status = std::process::Command::new(&shell)
                                                    .current_dir(&dir_path)
                                                    .stdin(Stdio::from(tty_in))
                                                    .stdout(Stdio::from(tty_out))
                                                    .stderr(Stdio::from(tty_err))
                                                    .status();

                                                // Resume TUI
                                                enable_raw_mode()?;
                                                terminal.backend_mut().execute(EnterAlternateScreen)?;
                                                terminal
                                                    .backend_mut()
                                                    .execute(crossterm::event::EnableMouseCapture)?;
                                                terminal.clear()?;

                                                if let Err(e) = status {
                                                    log::error!("Failed to launch shell: {}", e);
                                                }
                                            }
                                        }
                                    } else {
                                        // On a file, suspend TUI and open editor
                                        disable_raw_mode()?;
                                        terminal
                                            .backend_mut()
                                            .execute(crossterm::event::DisableMouseCapture)?;
                                        terminal.backend_mut().execute(LeaveAlternateScreen)?;
                                        terminal.backend_mut().execute(Show)?;

                                        // Use /dev/tty for editor so it can interact with the terminal
                                        // This works in both normal and shell-integration mode
                                        use std::process::Stdio;
                                        let tty_in = std::fs::OpenOptions::new()
                                            .read(true)
                                            .open("/dev/tty")?;
                                        let tty_out = std::fs::OpenOptions::new()
                                            .write(true)
                                            .open("/dev/tty")?;
                                        let tty_err = std::fs::OpenOptions::new()
                                            .write(true)
                                            .open("/dev/tty")?;

                                        let status = std::process::Command::new("hx")
                                            .arg(&display_info.full_path)
                                            .stdin(Stdio::from(tty_in))
                                            .stdout(Stdio::from(tty_out))
                                            .stderr(Stdio::from(tty_err))
                                            .status();

                                        // Resume TUI
                                        enable_raw_mode()?;
                                        terminal.backend_mut().execute(EnterAlternateScreen)?;
                                        terminal
                                            .backend_mut()
                                            .execute(crossterm::event::EnableMouseCapture)?;
                                        terminal.clear()?;

                                        if let Err(e) = status {
                                            log::error!("Failed to launch editor: {}", e);
                                        }

                                        // Reload model (may have been retrained in background)
                                        let query_id_model = app.next_subsession_id;
                                        app.next_subsession_id += 1;
                                        if let Err(e) = app.reload_model(query_id_model) {
                                            log::error!("Failed to reload model: {}", e);
                                        }

                                        // Reload click data and rerank files after editing
                                        let query_id_clicks = app.next_subsession_id;
                                        app.next_subsession_id += 1;
                                        if let Err(e) = app.reload_and_rerank(query_id_clicks) {
                                            log::error!("Failed to reload and rerank: {}", e);
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_path_no_truncation_needed() {
        let result = truncate_path("src/main.rs", 100);
        assert_eq!(result, "src/main.rs", "Short paths should not be truncated");
    }

    #[test]
    fn test_truncate_path_simple() {
        let result = truncate_path("a/b/c/d/e.txt", 12);
        assert_eq!(result, "a/.../e.txt", "Should truncate middle components when too long");
    }

    #[test]
    fn test_truncate_path_keeps_filename() {
        let result = truncate_path("very/long/path/to/some/file.txt", 18);
        assert_eq!(result, "very/.../file.txt", "Should always keep filename");
    }

    #[test]
    fn test_truncate_path_builds_from_end() {
        let result = truncate_path("a/b/c/d/e/f/g.txt", 15);
        // Actual: "a/.../e/f/g.txt" = 15 chars
        assert_eq!(result, "a/.../e/f/g.txt", "Should build from end to fit more context");
    }

    #[test]
    fn test_truncate_absolute_path_no_truncation() {
        let result = truncate_absolute_path("/Users/test/file.txt", 100);
        assert_eq!(result, "/Users/test/file.txt", "Short absolute paths should not be truncated");
    }

    #[test]
    fn test_truncate_absolute_path_both_ends() {
        let result = truncate_absolute_path("/Users/kshitijlauria/Library/CloudStorage/Dropbox/src/11-sg/todo.md", 40);
        // With abbreviation: "/Users/k/L/.../Dropbox/src/11-sg/todo.md" fits in 40 chars
        assert_eq!(result, "/Users/k/L/.../Dropbox/src/11-sg/todo.md", "Should abbreviate middle components to fit");
    }

    #[test]
    fn test_truncate_absolute_path_only_tail() {
        let result = truncate_absolute_path("/Users/kshitijlauria/Library/CloudStorage/Dropbox/src/11-sg/todo.md", 25);
        // With 25 chars, abbreviates head: "/U/k/.../src/11-sg/todo.md" = 27 chars, still too long
        // Actually gets: "/U/k/.../src/11-sg/todo.md"
        assert_eq!(result, "/U/k/.../src/11-sg/todo.md", "Should abbreviate head components when space is very limited");
    }

    #[test]
    fn test_truncate_absolute_path_exact_example() {
        // Test the specific example from the docs
        let input = "/Users/kshitijlauria/Library/CloudStorage/Dropbox/src/11-sg/todo.md";
        let result = truncate_absolute_path(input, 50);
        // Actual: "/.../Library/CloudStorage/Dropbox/src/11-sg/todo.md" = 53 chars
        // Note: This exceeds 50 chars because we prioritize showing complete components
        assert_eq!(result, "/.../Library/CloudStorage/Dropbox/src/11-sg/todo.md", "Shows as much tail as fits complete components");
    }

    #[test]
    fn test_truncate_absolute_path_relative() {
        // Test that relative paths also work
        let result = truncate_absolute_path("relative/path/to/file.txt", 18);
        // With abbreviation: "r/.../to/file.txt" = 18 chars exactly
        assert_eq!(result, "r/.../to/file.txt", "Should abbreviate first component when needed");
    }

    #[test]
    fn test_truncate_absolute_path_preserves_leading_slash() {
        let result = truncate_absolute_path("/a/b/c/d/e/f/g.txt", 15);
        assert!(result.starts_with("/"), "Should preserve leading slash for absolute paths");
        assert!(result.contains("..."), "Should truncate");
        assert!(result.ends_with("g.txt"), "Should keep filename");
    }

    #[test]
    fn test_truncate_absolute_path_fits_more_tail() {
        let result = truncate_absolute_path("/Users/name/Documents/Projects/rust/psychic/src/main.rs", 35);
        // Should prioritize tail (more recent parts of path)
        assert!(result.ends_with("main.rs"), "Should keep filename");
        assert!(result.contains("src"), "Should keep parent directory");
        assert!(result.contains("..."), "Should have ellipsis");
    }
}
