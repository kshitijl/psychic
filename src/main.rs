mod context;
mod db;
mod feature_defs;
mod features;
mod ranker;
mod walker;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use crossterm::{
    ExecutableCommand,
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use db::{Database, EventData, FileMetadata};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::Text,
    widgets::{Block, Borders, List, ListItem, Paragraph},
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

/// Output format for feature generation
#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    Csv,
    Json,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate features for machine learning from collected events
    GenerateFeatures {
        /// Output format (csv or json)
        #[arg(short, long, value_enum, default_value = "csv")]
        format: OutputFormat,
    },
    /// Retrain the ranking model using collected events
    Retrain,
}

/// A terminal-based file browser with fuzzy search and analytics
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Data directory for database and training files (default: ~/.local/share/psychic)
    #[arg(long, global = true, value_name = "DIR")]
    data_dir: Option<PathBuf>,

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

struct Subsession {
    id: u64,
    query: String,
    created_at: Instant,
    impressions_logged: bool,
}

#[derive(Debug, Clone)]
struct FileEntry {
    full_path: PathBuf,
    display_name: String,
}

impl FileEntry {
    fn from_walkdir(full_path: PathBuf, root: &PathBuf) -> Self {
        let display_name = full_path
            .strip_prefix(root)
            .unwrap_or(&full_path)
            .to_string_lossy()
            .to_string();

        FileEntry {
            full_path,
            display_name,
        }
    }

    fn from_history(full_path: PathBuf, root: &PathBuf) -> Self {
        let display_name = if full_path.strip_prefix(root).is_ok() {
            // File is in current tree - show relative path normally
            full_path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .to_string()
        } else {
            // File is from elsewhere - show .../{filename}
            let filename = full_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            format!(".../{}", filename)
        };

        FileEntry {
            full_path,
            display_name,
        }
    }
}

struct App {
    query: String,
    files: Vec<PathBuf>,
    historical_files: Vec<PathBuf>, // Previously clicked/scrolled files
    filtered_files: Vec<FileEntry>,
    file_scores: Vec<ranker::FileScore>, // Scores and features for filtered files
    selected_index: usize,
    file_list_scroll: u16, // Scroll offset for file list
    preview_scroll: u16,
    preview_cache: Option<(String, Text<'static>)>, // (file_path, rendered_text)
    scrolled_files: HashSet<(String, String)>, // (query, full_path) - track what we've logged scroll for
    root: PathBuf,
    session_id: String,
    current_subsession: Option<Subsession>,
    next_subsession_id: u64,
    db: Database,
    ranker: ranker::Ranker,
}

impl App {
    fn new(root: PathBuf, data_dir: &PathBuf) -> Result<Self> {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hash, Hasher};

        let start_time = Instant::now();
        log::debug!("App::new() started");

        // Generate random 64-bit session ID
        let mut hasher = RandomState::new().build_hasher();
        Instant::now().hash(&mut hasher);
        std::process::id().hash(&mut hasher);
        let session_id = hasher.finish().to_string();

        let db_start = Instant::now();
        let db = Database::new(data_dir)?;
        let db_path = db.db_path();
        log::debug!("Database initialization took {:?}", db_start.elapsed());

        // Try to load the ranker model if it exists (stored next to db file)
        let ranker_start = Instant::now();
        let model_path = db_path
            .parent()
            .map(|p| p.join("model.txt"))
            .unwrap_or_else(|| PathBuf::from("model.txt"));

        let ranker = ranker::Ranker::new(&model_path, db_path.clone())?;
        log::info!("Loaded ranking model from {:?}", model_path);
        log::debug!("Ranker initialization took {:?}", ranker_start.elapsed());

        // Load previously interacted files
        let historical_start = Instant::now();
        let historical_files = db
            .get_previously_interacted_files()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|p| {
                let path = PathBuf::from(&p);
                if path.exists() { Some(path) } else { None }
            })
            .collect();
        log::debug!(
            "Loading historical files took {:?}",
            historical_start.elapsed()
        );

        log::debug!("App::new() total time: {:?}", start_time.elapsed());

        Ok(App {
            query: String::new(),
            files: Vec::new(),
            historical_files,
            filtered_files: Vec::new(),
            file_scores: Vec::new(),
            selected_index: 0,
            file_list_scroll: 0,
            preview_scroll: 0,
            preview_cache: None,
            scrolled_files: HashSet::new(),
            root,
            session_id,
            current_subsession: None,
            next_subsession_id: 1,
            db,
            ranker,
        })
    }

    fn update_filtered_files(&mut self) {
        let start_time = Instant::now();
        let query_lower = self.query.to_lowercase();

        // First, filter files that match the query from walkdir
        let filter_start = Instant::now();
        let mut file_entries: Vec<FileEntry> = Vec::new();
        let mut matching_files: Vec<(String, PathBuf, Option<i64>)> = Vec::new();

        for path in &self.files {
            let entry = FileEntry::from_walkdir(path.clone(), &self.root);

            if query_lower.is_empty() || entry.display_name.to_lowercase().contains(&query_lower) {
                let (mtime, _, _) = get_file_metadata(&entry.full_path);
                matching_files.push((entry.display_name.clone(), entry.full_path.clone(), mtime));
                file_entries.push(entry);
            }
        }

        // Also include historical files that match the query (and aren't already in results)
        let existing_paths: HashSet<PathBuf> =
            file_entries.iter().map(|e| e.full_path.clone()).collect();
        for path in &self.historical_files {
            if !existing_paths.contains(path) {
                let entry = FileEntry::from_history(path.clone(), &self.root);

                if query_lower.is_empty()
                    || entry.display_name.to_lowercase().contains(&query_lower)
                {
                    let (mtime, _, _) = get_file_metadata(&entry.full_path);
                    matching_files.push((
                        entry.display_name.clone(),
                        entry.full_path.clone(),
                        mtime,
                    ));
                    file_entries.push(entry);
                }
            }
        }

        log::debug!(
            "Filtering {} files (with metadata) took {:?}",
            file_entries.len(),
            filter_start.elapsed()
        );

        // Then, rank them with the model
        let rank_start = Instant::now();
        match self.ranker.rank_files(&self.query, &matching_files, &self.session_id, &self.root) {
            Ok(scored) => {
                // Reorder file_entries based on ranking scores
                let score_order: Vec<String> =
                    scored.iter().map(|fs| fs.path.clone()).collect();
                let mut reordered_entries = Vec::new();

                for score_path in &score_order {
                    if let Some(entry) =
                        file_entries.iter().find(|e| &e.display_name == score_path)
                    {
                        reordered_entries.push(entry.clone());
                    }
                }

                self.filtered_files = reordered_entries;
                self.file_scores = scored;
            }
            Err(e) => {
                log::warn!("Ranking failed: {}, falling back to simple filtering", e);
                self.file_scores.clear();
                self.filtered_files = file_entries;
            }
        }
        log::debug!("Ranking took {:?}", rank_start.elapsed());

        log::debug!(
            "update_filtered_files() total time: {:?}",
            start_time.elapsed()
        );

        // Reset selection if out of bounds
        if self.selected_index >= self.filtered_files.len() && !self.filtered_files.is_empty() {
            self.selected_index = 0;
        }

        // Create new subsession if query changed
        let should_create_new = match &self.current_subsession {
            None => true,
            Some(subsession) => subsession.query != self.query,
        };

        if should_create_new {
            self.current_subsession = Some(Subsession {
                id: self.next_subsession_id,
                query: self.query.clone(),
                created_at: Instant::now(),
                impressions_logged: false,
            });
            self.next_subsession_id += 1;
        }
    }

    fn check_and_log_impressions(&mut self, force: bool) -> Result<()> {
        let subsession = match &mut self.current_subsession {
            Some(s) => s,
            None => return Ok(()),
        };

        // Skip if already logged
        if subsession.impressions_logged {
            return Ok(());
        }

        // Check if we should log: either forced or >200ms old
        let should_log = force || subsession.created_at.elapsed() >= Duration::from_millis(200);
        if !should_log {
            return Ok(());
        }

        // Log top 10 visible files with metadata
        let top_10: Vec<FileMetadata> = self
            .filtered_files
            .iter()
            .take(10)
            .map(|entry| {
                let (mtime, atime, file_size) = get_file_metadata(&entry.full_path);
                (
                    entry.display_name.clone(),
                    entry.full_path.to_string_lossy().to_string(),
                    mtime,
                    atime,
                    file_size,
                )
            })
            .collect();

        if !top_10.is_empty() {
            self.db
                .log_impressions(&subsession.query, &top_10, subsession.id, &self.session_id)?;
            subsession.impressions_logged = true;
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

    fn update_scroll(&mut self, visible_height: u16) {
        // Auto-scroll the file list when selection is near top or bottom
        let selected = self.selected_index as u16;
        let scroll = self.file_list_scroll;

        // If selected item is above visible area, scroll up
        if selected < scroll {
            self.file_list_scroll = selected;
        }
        // If selected item is below visible area, scroll down
        else if selected >= scroll + visible_height {
            self.file_list_scroll = selected.saturating_sub(visible_height - 1);
        }
        // If we're in the bottom 5 items and there's more to see, keep scrolling
        else if selected >= scroll + visible_height.saturating_sub(5) {
            self.file_list_scroll = selected.saturating_sub(visible_height.saturating_sub(5));
        }
    }
}

fn get_file_metadata(path: &PathBuf) -> (Option<i64>, Option<i64>, Option<i64>) {
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

        let file_size = Some(metadata.len() as i64);

        (mtime, atime, file_size)
    } else {
        (None, None, None)
    }
}

fn get_time_ago(path: &PathBuf) -> String {
    if let Ok(metadata) = std::fs::metadata(path)
        && let Ok(modified) = metadata.modified() {
        let duration = std::time::SystemTime::now()
            .duration_since(modified)
            .unwrap_or(Duration::from_secs(0));

        let formatter = timeago::Formatter::new();
        return formatter.convert(duration);
    }
    String::from("unknown")
}

fn main() -> Result<()> {
    // Initialize logger to write to ~/.local/share/psychic/app.log
    if let Ok(home) = std::env::var("HOME") {
        let log_dir = PathBuf::from(&home).join(".local").join("share").join("psychic");
        let _ = std::fs::create_dir_all(&log_dir);
        let log_file = log_dir.join("app.log");

        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_file)
        {
            env_logger::Builder::new()
                .target(env_logger::Target::Pipe(Box::new(file)))
                .filter_level(log::LevelFilter::Debug)
                .init();
        }
    }

    let cli = Cli::parse();

    // Handle subcommands
    if let Some(command) = cli.command {
        // Get data directory (global option)
        let data_dir = cli.data_dir.unwrap_or_else(|| get_default_data_dir().expect("Failed to get default data directory"));

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
                let db_path = db::Database::get_db_path(&data_dir)?;

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
        }
    }

    // Get current working directory
    let root = env::current_dir()?;

    // Get data directory for main app
    let data_dir = cli.data_dir.unwrap_or_else(|| get_default_data_dir().expect("Failed to get default data directory"));

    // Initialize app
    let mut app = App::new(root.clone(), &data_dir)?;

    log::info!(
        "Started psychic in directory {}, session {}",
        root.display(),
        app.session_id
    );

    // Gather context in background thread
    let session_id_clone = app.session_id.clone();
    let data_dir_clone = data_dir.clone();
    std::thread::spawn(move || {
        let context = context::gather_context();
        if let Ok(db) = db::Database::new(&data_dir_clone) {
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
        let mut received_files = false;
        while let Ok(path) = rx.try_recv() {
            app.files.push(path);
            received_files = true;
        }

        // Only update filtered files once after receiving all available files
        if received_files {
            app.update_filtered_files();
        }

        // Check and log impressions if >200ms old
        let _ = app.check_and_log_impressions(false);

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

            // Split top area horizontally: left for file list, middle for preview, right for features
            let top_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(35), // File list
                    Constraint::Percentage(45), // Preview
                    Constraint::Percentage(20), // Features
                ])
                .split(main_chunks[0]);

            // Update scroll position based on selection and visible height
            let visible_height = top_chunks[0].height.saturating_sub(2); // subtract border
            app.update_scroll(visible_height);

            // File list on the left
            let list_width = top_chunks[0].width.saturating_sub(2) as usize; // subtract borders
            let items: Vec<ListItem> = app
                .filtered_files
                .iter()
                .enumerate()
                .skip(app.file_list_scroll as usize)
                .take(visible_height as usize)
                .map(|(i, entry)| {
                    let time_ago = get_time_ago(&entry.full_path);
                    let rank = i + 1;

                    // Calculate space: "N. " takes 4 chars, time_ago length, we need padding between
                    let rank_prefix = format!("{:2}. ", rank);
                    let prefix_len = rank_prefix.len();
                    let time_len = time_ago.len();

                    // Available space for filename and padding
                    let available = list_width.saturating_sub(prefix_len + time_len);
                    let file_width = available.saturating_sub(2); // leave at least 2 spaces padding

                    // Right-justify time by padding filename to fill available space
                    let line = if entry.display_name.len() > file_width {
                        format!(
                            "{}{:<width$}  {}",
                            rank_prefix,
                            &entry.display_name[..file_width],
                            time_ago,
                            width = file_width
                        )
                    } else {
                        format!(
                            "{}{:<width$}  {}",
                            rank_prefix,
                            &entry.display_name,
                            time_ago,
                            width = file_width
                        )
                    };

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

            let list = List::new(items).block(Block::default().borders(Borders::ALL).title(
                format!("Files ({}/{})", app.filtered_files.len(), app.files.len()),
            ));
            f.render_widget(list, top_chunks[0]);

            // Preview on the right using bat (with smart caching)
            let preview_height = top_chunks[1].height.saturating_sub(2);
            let preview_text = if !app.filtered_files.is_empty() {
                let selected_entry = &app.filtered_files[app.selected_index];
                let full_path_str = selected_entry.full_path.to_string_lossy().to_string();

                // Check for a full, cached preview
                if let Some(cached_text) = app.preview_cache.as_ref().and_then(|(path, text)| {
                    if path == &full_path_str {
                        Some(text.clone())
                    } else {
                        None
                    }
                }) {
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
                            .arg(&selected_entry.full_path)
                            .output();

                        let text = match bat_output {
                            Ok(output) => match ansi_to_tui::IntoText::into_text(&output.stdout) {
                                Ok(text) => text,
                                Err(_) => Text::from("[Unable to parse preview]"),
                            },
                            Err(_) => {
                                // Fallback for light preview
                                match std::fs::read_to_string(&selected_entry.full_path) {
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
                            .arg(&selected_entry.full_path)
                            .output();

                        let text = match bat_output {
                            Ok(output) => match ansi_to_tui::IntoText::into_text(&output.stdout) {
                                Ok(text) => text,
                                Err(_) => Text::from("[Unable to parse preview]"),
                            },
                            Err(_) => {
                                // Fallback for full preview
                                match std::fs::read_to_string(&selected_entry.full_path) {
                                    Ok(content) => Text::from(content),
                                    Err(_) => Text::from("[Unable to preview file]"),
                                }
                            }
                        };
                        (text, true) // Cache the full preview
                    };

                    if should_cache {
                        app.preview_cache = Some((full_path_str, text_to_render.clone()));
                    }
                    text_to_render
                }
            } else {
                Text::from("")
            };

            let preview = Paragraph::new(preview_text)
                .scroll((app.preview_scroll, 0))
                .block(Block::default().borders(Borders::ALL).title("Preview"));
            f.render_widget(preview, top_chunks[1]);

            // Debug panel on the right
            let mut debug_lines = Vec::new();

            if !app.file_scores.is_empty() && app.selected_index < app.file_scores.len() {
                let file_score = &app.file_scores[app.selected_index];
                debug_lines.push(format!("Score: {:.4}", file_score.score));
                debug_lines.push(String::from(""));
                debug_lines.push(String::from("Features:"));
                debug_lines.push(String::from(""));

                // Show all features from registry
                for feature in feature_defs::FEATURE_REGISTRY.iter() {
                    if let Some(value) = file_score.features.get(feature.name()) {
                        debug_lines.push(format!("{}: {}", feature.name(), value));
                    }
                }
            } else {
                debug_lines.push(String::from("No features"));
                debug_lines.push(String::from("(no file selected)"));
            }

            debug_lines.push(String::from("")); // Separator

            // Add preview cache status
            if !app.filtered_files.is_empty() {
                let selected_entry = &app.filtered_files[app.selected_index];
                let full_path_str = selected_entry.full_path.to_string_lossy().to_string();
                let is_cached = app
                    .preview_cache
                    .as_ref()
                    .is_some_and(|(p, _)| p == &full_path_str);
                let cache_status = if is_cached { "Cached" } else { "Live" };
                debug_lines.push(format!("Preview: {}", cache_status));
            } else {
                debug_lines.push(String::from("Preview: N/A"));
            }

            let debug_text = debug_lines.join("\n");

            let debug_pane = Paragraph::new(debug_text)
                .block(Block::default().borders(Borders::ALL).title("Debug"));
            f.render_widget(debug_pane, top_chunks[2]);

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
                                // Force log impressions before scroll
                                let _ = app.check_and_log_impressions(true);

                                let selected_entry = &app.filtered_files[app.selected_index];
                                let full_path_str =
                                    selected_entry.full_path.to_string_lossy().to_string();
                                let key = (app.query.clone(), full_path_str.clone());

                                if !app.scrolled_files.contains(&key) {
                                    let (mtime, atime, file_size) =
                                        get_file_metadata(&selected_entry.full_path);
                                    let subsession_id =
                                        app.current_subsession.as_ref().map(|s| s.id).unwrap_or(1);
                                    let _ = app.db.log_scroll(EventData {
                                        query: &app.query,
                                        file_path: &selected_entry.display_name,
                                        full_path: &full_path_str,
                                        mtime,
                                        atime,
                                        file_size,
                                        subsession_id,
                                        action: "scroll",
                                        session_id: &app.session_id,
                                    });
                                    app.scrolled_files.insert(key);
                                }
                            }
                        }
                        MouseEventKind::ScrollUp => {
                            app.preview_scroll = app.preview_scroll.saturating_sub(3);

                            // Log scroll event (deduplicated by query + full_path)
                            if !app.filtered_files.is_empty() {
                                // Force log impressions before scroll
                                let _ = app.check_and_log_impressions(true);

                                let selected_entry = &app.filtered_files[app.selected_index];
                                let full_path_str =
                                    selected_entry.full_path.to_string_lossy().to_string();
                                let key = (app.query.clone(), full_path_str.clone());

                                if !app.scrolled_files.contains(&key) {
                                    let (mtime, atime, file_size) =
                                        get_file_metadata(&selected_entry.full_path);
                                    let subsession_id =
                                        app.current_subsession.as_ref().map(|s| s.id).unwrap_or(1);
                                    let _ = app.db.log_scroll(EventData {
                                        query: &app.query,
                                        file_path: &selected_entry.display_name,
                                        full_path: &full_path_str,
                                        mtime,
                                        atime,
                                        file_size,
                                        subsession_id,
                                        action: "scroll",
                                        session_id: &app.session_id,
                                    });
                                    app.scrolled_files.insert(key);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    match key.code {
                        KeyCode::Char('c')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            return Ok(());
                        }
                        KeyCode::Char('u')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            // Ctrl-U: clear entire search
                            app.query.clear();
                            app.update_filtered_files();
                        }
                        KeyCode::Esc => {
                            return Ok(());
                        }
                        KeyCode::Char(c) => {
                            app.query.push(c);
                            app.update_filtered_files();
                        }
                        KeyCode::Backspace => {
                            app.query.pop();
                            app.update_filtered_files();
                        }
                        KeyCode::Up => {
                            app.move_selection(-1);
                        }
                        KeyCode::Down => {
                            app.move_selection(1);
                        }
                        KeyCode::Enter => {
                            if !app.filtered_files.is_empty() {
                                // Force log impressions before click
                                app.check_and_log_impressions(true)?;

                                let selected_entry = &app.filtered_files[app.selected_index];
                                let (mtime, atime, file_size) =
                                    get_file_metadata(&selected_entry.full_path);

                                // Log the click
                                let subsession_id =
                                    app.current_subsession.as_ref().map(|s| s.id).unwrap_or(1);
                                app.db.log_click(EventData {
                                    query: &app.query,
                                    file_path: &selected_entry.display_name,
                                    full_path: &selected_entry.full_path.to_string_lossy(),
                                    mtime,
                                    atime,
                                    file_size,
                                    subsession_id,
                                    action: "click",
                                    session_id: &app.session_id,
                                })?;

                                // Suspend TUI and launch editor
                                disable_raw_mode()?;
                                terminal
                                    .backend_mut()
                                    .execute(crossterm::event::DisableMouseCapture)?;
                                terminal.backend_mut().execute(LeaveAlternateScreen)?;

                                // Launch hx editor
                                let status = std::process::Command::new("hx")
                                    .arg(&selected_entry.full_path)
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
