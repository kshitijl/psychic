mod context;
mod db;
mod walker;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use db::Database;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::Text,
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Terminal,
};
use std::{
    collections::HashSet,
    env,
    io::{self, stdout},
    path::PathBuf,
    sync::mpsc::{self, Receiver},
    time::{Duration, Instant},
};
use walker::start_file_walker;

struct App {
    query: String,
    files: Vec<PathBuf>,
    filtered_files: Vec<String>,
    selected_index: usize,
    preview_scroll: u16,
    preview_cache: Option<(String, Text<'static>)>, // (file_path, rendered_text)
    scrolled_files: HashSet<(String, String)>, // (query, full_path) - track what we've logged scroll for
    root: PathBuf,
    session_id: String,
    db: Database,
    last_impression_log: Instant,
}

impl App {
    fn new(root: PathBuf) -> Result<Self> {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hash, Hasher};

        // Generate random 64-bit session ID
        let mut hasher = RandomState::new().build_hasher();
        Instant::now().hash(&mut hasher);
        std::process::id().hash(&mut hasher);
        let session_id = hasher.finish().to_string();

        Ok(App {
            query: String::new(),
            files: Vec::new(),
            filtered_files: Vec::new(),
            selected_index: 0,
            preview_scroll: 0,
            preview_cache: None,
            scrolled_files: HashSet::new(),
            root,
            session_id,
            db: Database::new()?,
            last_impression_log: Instant::now(),
        })
    }

    fn update_filtered_files(&mut self) {
        let query_lower = self.query.to_lowercase();

        self.filtered_files = self
            .files
            .iter()
            .filter_map(|path| {
                let relative = path
                    .strip_prefix(&self.root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string();

                if query_lower.is_empty() || relative.to_lowercase().contains(&query_lower) {
                    Some(relative)
                } else {
                    None
                }
            })
            .collect();

        // Reset selection if out of bounds
        if self.selected_index >= self.filtered_files.len() && !self.filtered_files.is_empty() {
            self.selected_index = 0;
        }
    }

    fn log_impressions_debounced(&mut self) -> Result<()> {
        // Debounce impressions by 200ms
        if self.last_impression_log.elapsed() < Duration::from_millis(200) {
            return Ok(());
        }

        // Log top 10 visible files with metadata
        let top_10: Vec<(String, String, Option<i64>, Option<i64>)> = self
            .filtered_files
            .iter()
            .take(10)
            .map(|relative| {
                let full_path = self.root.join(relative);
                let (mtime, atime) = get_file_times(&full_path);
                (
                    relative.clone(),
                    full_path.to_string_lossy().to_string(),
                    mtime,
                    atime,
                )
            })
            .collect();

        if !top_10.is_empty() {
            self.db.log_impressions(&self.query, &top_10, &self.session_id)?;
            self.last_impression_log = Instant::now();
        }

        Ok(())
    }

    fn move_selection(&mut self, delta: isize) {
        if self.filtered_files.is_empty() {
            return;
        }

        let len = self.filtered_files.len() as isize;
        let new_index = (self.selected_index as isize + delta).rem_euclid(len);
        self.selected_index = new_index as usize;

        // Reset preview scroll and clear cache when changing selection
        self.preview_scroll = 0;
        self.preview_cache = None;
    }
}

fn get_file_times(path: &PathBuf) -> (Option<i64>, Option<i64>) {
    if let Ok(metadata) = std::fs::metadata(path) {
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64);

        let atime = metadata
            .accessed()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64);

        (mtime, atime)
    } else {
        (None, None)
    }
}

fn get_time_ago(path: &PathBuf) -> String {
    if let Ok(metadata) = std::fs::metadata(path) {
        if let Ok(modified) = metadata.modified() {
            let duration = std::time::SystemTime::now()
                .duration_since(modified)
                .unwrap_or(Duration::from_secs(0));

            let formatter = timeago::Formatter::new();
            return formatter.convert(duration);
        }
    }
    String::from("unknown")
}

fn main() -> Result<()> {
    // Get current working directory
    let root = env::current_dir()?;

    // Initialize app
    let mut app = App::new(root.clone())?;

    // Gather context in background thread
    let session_id_clone = app.session_id.clone();
    std::thread::spawn(move || {
        let context = context::gather_context();
        if let Ok(db) = db::Database::new() {
            let _ = db.log_session(&session_id_clone, &context);
        }
    });

    // Start file walker
    let (tx, rx) = mpsc::channel();
    start_file_walker(root.clone(), tx);

    // Setup terminal
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    stdout().execute(crossterm::event::EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    // Run the app
    let result = run_app(&mut terminal, &mut app, rx);

    // Cleanup
    disable_raw_mode()?;
    stdout().execute(crossterm::event::DisableMouseCapture)?;
    stdout().execute(LeaveAlternateScreen)?;

    result
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    rx: Receiver<PathBuf>,
) -> Result<()> {
    loop {
        // Receive new files from walker (non-blocking)
        while let Ok(path) = rx.try_recv() {
            app.files.push(path);
            app.update_filtered_files();
        }

        // Draw UI
        terminal.draw(|f| {
            // Split vertically: top for results/preview, bottom for input
            let main_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(0),    // Results + Preview
                    Constraint::Length(3), // Search input at bottom
                ])
                .split(f.area());

            // Split top area horizontally: left for file list, right for preview
            let top_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(50), // File list
                    Constraint::Percentage(50), // Preview
                ])
                .split(main_chunks[0]);

            // File list on the left
            let items: Vec<ListItem> = app
                .filtered_files
                .iter()
                .enumerate()
                .map(|(i, file)| {
                    let full_path = app.root.join(file);
                    let time_ago = get_time_ago(&full_path);
                    let rank = i + 1;

                    // Format: "1. src/main.rs                      2 days ago"
                    let line = format!("{:2}. {:<50} {}", rank, file, time_ago);

                    let style = if i == app.selected_index {
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };
                    ListItem::new(line).style(style)
                })
                .collect();

            let list = List::new(items)
                .block(Block::default().borders(Borders::ALL).title(format!(
                    "Files ({}/{})",
                    app.filtered_files.len(),
                    app.files.len()
                )));
            f.render_widget(list, top_chunks[0]);

            // Preview on the right using bat (with caching)
            let preview_text = if !app.filtered_files.is_empty() {
                let selected_file = &app.filtered_files[app.selected_index];
                let full_path = app.root.join(selected_file);
                let full_path_str = full_path.to_string_lossy().to_string();

                // Check cache
                let cached = app.preview_cache.as_ref()
                    .filter(|(path, _)| path == &full_path_str)
                    .map(|(_, text)| text.clone());

                if let Some(text) = cached {
                    text
                } else {
                    // Cache miss - fetch full file with bat
                    let text = match std::process::Command::new("bat")
                        .arg("--color=always")
                        .arg("--style=numbers")
                        .arg("--paging=never")
                        .arg(&full_path)
                        .output()
                    {
                        Ok(output) => {
                            // Convert ANSI codes to ratatui Text
                            match ansi_to_tui::IntoText::into_text(&output.stdout) {
                                Ok(text) => text,
                                Err(_) => Text::from("[Unable to parse preview]"),
                            }
                        }
                        Err(_) => {
                            // Fallback to plain file read if bat is not available
                            match std::fs::read_to_string(&full_path) {
                                Ok(content) => Text::from(content),
                                Err(_) => Text::from("[Unable to preview file]"),
                            }
                        }
                    };

                    // Cache the result
                    app.preview_cache = Some((full_path_str, text.clone()));
                    text
                }
            } else {
                Text::from("")
            };

            let preview = Paragraph::new(preview_text)
                .scroll((app.preview_scroll, 0))
                .block(Block::default().borders(Borders::ALL).title("Preview"));
            f.render_widget(preview, top_chunks[1]);

            // Search input at the bottom
            let input = Paragraph::new(app.query.as_str())
                .block(Block::default().borders(Borders::ALL).title("Search"));
            f.render_widget(input, main_chunks[1]);
        })?;

        // Handle events with timeout
        if event::poll(Duration::from_millis(100))? {
            let event_read = event::read()?;
            match event_read {
                Event::Mouse(mouse_event) => {
                    use crossterm::event::MouseEventKind;
                    match mouse_event.kind {
                        MouseEventKind::ScrollDown => {
                            app.preview_scroll = app.preview_scroll.saturating_add(3);

                            // Log scroll event (deduplicated by query + full_path)
                            if !app.filtered_files.is_empty() {
                                let selected_file = &app.filtered_files[app.selected_index];
                                let full_path = app.root.join(selected_file);
                                let full_path_str = full_path.to_string_lossy().to_string();
                                let key = (app.query.clone(), full_path_str.clone());

                                if !app.scrolled_files.contains(&key) {
                                    let (mtime, atime) = get_file_times(&full_path);
                                    let _ = app.db.log_scroll(
                                        &app.query,
                                        selected_file,
                                        &full_path_str,
                                        mtime,
                                        atime,
                                        &app.session_id,
                                    );
                                    app.scrolled_files.insert(key);
                                }
                            }
                        }
                        MouseEventKind::ScrollUp => {
                            app.preview_scroll = app.preview_scroll.saturating_sub(3);

                            // Log scroll event (deduplicated by query + full_path)
                            if !app.filtered_files.is_empty() {
                                let selected_file = &app.filtered_files[app.selected_index];
                                let full_path = app.root.join(selected_file);
                                let full_path_str = full_path.to_string_lossy().to_string();
                                let key = (app.query.clone(), full_path_str.clone());

                                if !app.scrolled_files.contains(&key) {
                                    let (mtime, atime) = get_file_times(&full_path);
                                    let _ = app.db.log_scroll(
                                        &app.query,
                                        selected_file,
                                        &full_path_str,
                                        mtime,
                                        atime,
                                        &app.session_id,
                                    );
                                    app.scrolled_files.insert(key);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                            return Ok(());
                        }
                        KeyCode::Esc => {
                            return Ok(());
                        }
                        KeyCode::Char(c) => {
                            app.query.push(c);
                            app.update_filtered_files();
                            app.log_impressions_debounced()?;
                        }
                        KeyCode::Backspace => {
                            app.query.pop();
                            app.update_filtered_files();
                            app.log_impressions_debounced()?;
                        }
                        KeyCode::Up => {
                            app.move_selection(-1);
                        }
                        KeyCode::Down => {
                            app.move_selection(1);
                        }
                        KeyCode::Enter => {
                            if !app.filtered_files.is_empty() {
                                let selected_file = &app.filtered_files[app.selected_index];
                                let full_path = app.root.join(selected_file);
                                let (mtime, atime) = get_file_times(&full_path);

                                // Log the click
                                app.db.log_click(
                                    &app.query,
                                    selected_file,
                                    &full_path.to_string_lossy(),
                                    mtime,
                                    atime,
                                    &app.session_id,
                                )?;

                                // Suspend TUI and launch editor
                                disable_raw_mode()?;
                                terminal.backend_mut().execute(crossterm::event::DisableMouseCapture)?;
                                terminal.backend_mut().execute(LeaveAlternateScreen)?;

                                // Launch hx editor
                                let status = std::process::Command::new("hx")
                                    .arg(&full_path)
                                    .status();

                                // Resume TUI
                                enable_raw_mode()?;
                                terminal.backend_mut().execute(EnterAlternateScreen)?;
                                terminal.backend_mut().execute(crossterm::event::EnableMouseCapture)?;
                                terminal.clear()?;

                                if let Err(e) = status {
                                    eprintln!("Failed to launch editor: {}", e);
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
