mod context;
mod db;
mod feature_defs;
mod features;
mod ranker;
mod search_worker;
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
use search_worker::{
    DisplayFileInfo, WorkerRequest, WorkerResponse, get_file_metadata, get_time_ago,
};
use std::{
    collections::{HashSet, VecDeque},
    env,
    io::{self, stdout},
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver},
    time::{Duration, Instant},
};

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
}

/// A terminal-based file browser that learns which files you want to see.
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
    visible_files: Vec<DisplayFileInfo>, // Only what's currently visible
    visible_files_offset: usize,         // What index the visible_files array starts at
    total_results: usize,                // Total number of filtered files
    total_files: usize,                  // Total number of files in index
    selected_index: usize,
    file_list_scroll: u16, // Scroll offset for file list
    preview_scroll: u16,
    preview_cache: Option<(String, Text<'static>)>, // (file_path, rendered_text)
    scrolled_files: HashSet<(String, String)>, // (query, full_path) - track what we've logged scroll for
    session_id: String,
    current_subsession: Option<Subsession>,
    next_subsession_id: u64,
    db: Database,

    num_results_to_log_as_impressions: usize,

    // For debug pane
    model_stats_cache: Option<ranker::ModelStats>, // Cached from worker, refreshed periodically
    currently_retraining: bool,
    log_receiver: Receiver<String>,
    recent_logs: VecDeque<String>,
    debug_pane_maximized: bool,

    // Search worker thread communication
    worker_tx: mpsc::Sender<WorkerRequest>,
    worker_rx: mpsc::Receiver<WorkerResponse>,
}

impl App {
    fn new(root: PathBuf, data_dir: &Path, log_receiver: Receiver<String>) -> Result<Self> {
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
        let db_path = Database::get_db_path(data_dir);
        let db = Database::new(&db_path)?;
        log::debug!("Database initialization took {:?}", db_start.elapsed());

        let (worker_tx, worker_rx) = search_worker::spawn(root.clone(), data_dir)?;

        log::debug!("App::new() total time: {:?}", start_time.elapsed());

        let app = App {
            query: String::new(),
            visible_files: Vec::new(),
            visible_files_offset: 0,
            total_results: 0,
            total_files: 0,
            selected_index: 0,
            file_list_scroll: 0,
            preview_scroll: 0,
            preview_cache: None,
            scrolled_files: HashSet::new(),
            session_id,
            current_subsession: None,
            next_subsession_id: 1,
            num_results_to_log_as_impressions: 25,
            db,
            model_stats_cache: None,
            currently_retraining: false,
            log_receiver,
            recent_logs: VecDeque::with_capacity(50),
            debug_pane_maximized: false,
            worker_tx: worker_tx.clone(),
            worker_rx,
        };

        // Send initial query to worker
        let _ = worker_tx.send(WorkerRequest::UpdateQuery(String::new()));

        Ok(app)
    }

    fn reload_model(&mut self) -> Result<()> {
        log::info!("Requesting model reload from worker");
        self.worker_tx
            .send(WorkerRequest::ReloadModel)
            .context("Failed to send ReloadModel request to worker")?;
        Ok(())
    }

    fn reload_and_rerank(&mut self) -> Result<()> {
        log::info!("Requesting clicks reload from worker");
        self.worker_tx
            .send(WorkerRequest::ReloadClicks)
            .context("Failed to send ReloadClicks request to worker")?;
        Ok(())
    }

    fn check_and_log_impressions(&mut self, force: bool) -> Result<()> {
        let subsession = match &mut self.current_subsession {
            Some(s) => s,
            None => return Ok(()),
        };

        // Skip if already logged
        if subsession.events_have_been_logged {
            return Ok(());
        }

        // Check if we should log: either forced or >200ms old
        let elapsed = jiff::Timestamp::now().duration_since(subsession.created_at);
        let threshold = jiff::SignedDuration::from_millis(200);
        let should_log = force || elapsed >= threshold;
        if !should_log {
            return Ok(());
        }

        // Log top N visible files with metadata
        let top_n: Vec<FileMetadata> = self
            .visible_files
            .iter()
            .take(self.num_results_to_log_as_impressions)
            .map(|display_info| {
                let (mtime, atime, file_size) = get_file_metadata(&display_info.full_path);

                FileMetadata {
                    relative_path: display_info.display_name.clone(),
                    full_path: display_info.full_path.to_string_lossy().to_string(),
                    mtime,
                    atime,
                    size: file_size,
                }
            })
            .collect();

        if !top_n.is_empty() {
            self.db
                .log_impressions(&subsession.query, &top_n, subsession.id, &self.session_id)?;
            subsession.events_have_been_logged = true;
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
    }

    fn update_scroll(&mut self, visible_height: u16) {
        // If all results fit on screen, don't scroll at all
        if self.total_results <= visible_height as usize {
            self.file_list_scroll = 0;
            // Still need to request visible slice if needed
            let visible_end = self.visible_files_offset + self.visible_files.len();
            let needed_end = visible_height as usize + 10;

            if needed_end > visible_end && visible_end < self.total_results {
                let count = (visible_height as usize * 2).max(60);
                self.visible_files_offset = 0;
                let _ = self.worker_tx.send(WorkerRequest::GetVisibleSlice {
                    start: 0,
                    count,
                });
            }
            return;
        }

        // Auto-scroll the file list when selection is near top or bottom
        let selected = self.selected_index as u16;
        let scroll = self.file_list_scroll;

        // If selected item is above visible area, scroll up
        if selected < scroll {
            self.file_list_scroll = selected;
        }
        // If selected item is below visible area, scroll down
        else if selected >= scroll + visible_height {
            // Smart positioning: leave some space from bottom (5 lines)
            // This makes wrap-around more comfortable
            let margin = 5u16;
            self.file_list_scroll = selected.saturating_sub(visible_height.saturating_sub(margin).min(visible_height - 1));
        }
        // If we're in the bottom 5 items and there's more to see, keep scrolling
        else if selected >= scroll + visible_height.saturating_sub(5) {
            self.file_list_scroll = selected.saturating_sub(visible_height.saturating_sub(5));
        }

        // Request more visible items if we're getting close to the edge
        let scroll_offset = self.file_list_scroll as usize;
        let visible_end = self.visible_files_offset + self.visible_files.len();
        let needed_end = scroll_offset + visible_height as usize + 10; // 10 item buffer

        // Check if we need to request a new slice
        // We need new data if: scroll_offset is outside our current window OR we're near the end
        let need_new_slice = scroll_offset < self.visible_files_offset
            || scroll_offset >= visible_end
            || (needed_end > visible_end && visible_end < self.total_results);

        if need_new_slice {
            // Request more items
            let count = (visible_height as usize * 2).max(60); // At least 60 or 2x screen height
            self.visible_files_offset = scroll_offset; // Update offset optimistically
            let _ = self.worker_tx.send(WorkerRequest::GetVisibleSlice {
                start: scroll_offset,
                count,
            });
        }
    }
}

fn main() -> Result<()> {
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
                out.finish(format_args!(
                    "[{} {} {}] {}",
                    jiff::Timestamp::now().strftime("%Y-%m-%d %H:%M:%S"),
                    record.level(),
                    record.target(),
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
        }
    }

    // Get current working directory and canonicalize once
    let root = env::current_dir()?.canonicalize()?;

    // Get data directory for main app
    let data_dir = cli
        .data_dir
        .unwrap_or_else(|| get_default_data_dir().expect("Failed to get default data directory"));

    // Create channel for retraining status
    let (retrain_tx, retrain_rx) = mpsc::channel();

    // Start background retraining in a new thread
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

    // Initialize app
    let mut app = App::new(root.clone(), &data_dir, log_rx)?;

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

    // Setup terminal
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    stdout().execute(crossterm::event::EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    // Run the app
    let result = run_app(&mut terminal, &mut app, retrain_rx);

    // Cleanup
    disable_raw_mode()?;
    stdout().execute(crossterm::event::DisableMouseCapture)?;
    stdout().execute(LeaveAlternateScreen)?;

    result
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    retrain_rx: Receiver<bool>,
) -> Result<()> {
    loop {
        // Log impressions for this subsession if >200ms old
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

            // Split top area horizontally: left for file list, middle for preview, right for debug
            let top_chunks = if app.debug_pane_maximized {
                // When debug is maximized, give it most of the space
                Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(25), // File list (smaller)
                        Constraint::Percentage(0),  // Preview (hidden)
                        Constraint::Percentage(75), // Debug (maximized)
                    ])
                    .split(main_chunks[0])
            } else {
                Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(35), // File list
                        Constraint::Percentage(45), // Preview
                        Constraint::Percentage(20), // Debug
                    ])
                    .split(main_chunks[0])
            };

            // Update scroll position based on selection and visible height
            let visible_height = top_chunks[0].height.saturating_sub(2); // subtract border
            app.update_scroll(visible_height);

            // File list on the left
            let list_width = top_chunks[0].width.saturating_sub(2) as usize; // subtract borders
            let scroll_offset = app.file_list_scroll as usize;

            // Calculate which items from visible_files to display
            // visible_files starts at visible_files_offset, we want to show from scroll_offset
            let skip_count = scroll_offset.saturating_sub(app.visible_files_offset);

            let items: Vec<ListItem> = app
                .visible_files
                .iter()
                .skip(skip_count)
                .enumerate()
                .map(|(display_idx, display_info)| {
                    let i = scroll_offset + display_idx;
                    let time_ago = get_time_ago(&display_info.full_path);
                    let rank = i + 1;

                    // Calculate space: "N. " takes 4 chars, time_ago length, we need padding between
                    let rank_prefix = format!("{:2}. ", rank);
                    let prefix_len = rank_prefix.len();
                    let time_len = time_ago.len();

                    // Available space for filename and padding
                    let available = list_width.saturating_sub(prefix_len + time_len);
                    let file_width = available.saturating_sub(2); // leave at least 2 spaces padding

                    // Right-justify time by padding filename to fill available space
                    let line = if display_info.display_name.len() > file_width {
                        format!(
                            "{}{:<width$}  {}",
                            rank_prefix,
                            &display_info.display_name[..file_width],
                            time_ago,
                            width = file_width
                        )
                    } else {
                        format!(
                            "{}{:<width$}  {}",
                            rank_prefix,
                            &display_info.display_name,
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

            let list = List::new(items).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!("Files ({}/{})", app.total_results, app.total_files)),
            );
            f.render_widget(list, top_chunks[0]);

            // Calculate index into visible_files array
            // selected_index is global, visible_files_offset is where the array starts
            let visible_idx = app.selected_index.saturating_sub(app.visible_files_offset);

            let current_file = if visible_idx < app.visible_files.len() {
                Some(&app.visible_files[visible_idx])
            } else {
                None
            };

            let current_file_path = current_file.map(|x| x.full_path.to_string_lossy().to_string());

            // Preview on the right using bat (with smart caching)
            let preview_height = top_chunks[1].height.saturating_sub(2);
            let preview_text = if !app.visible_files.is_empty() && app.total_results > 0 {
                if let Some(current_file_path) = &current_file_path {
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
                } else {
                    Text::from("[Loading preview...]")
                }
            } else {
                Text::from("")
            };

            let preview_pane_title = current_file
                .and_then(|x| x.full_path.file_name())
                .map(|x| x.to_string_lossy())
                .map(|x| x.to_string())
                .unwrap_or("No file selected".to_string());

            let preview = Paragraph::new(preview_text)
                .scroll((app.preview_scroll, 0))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(preview_pane_title),
                );
            f.render_widget(preview, top_chunks[1]);

            // Debug panel on the right
            let mut debug_lines = Vec::new();

            // Show current selection info
            if !app.visible_files.is_empty() && app.total_results > 0 {
                let visible_idx = app.selected_index.saturating_sub(app.visible_files_offset);
                if visible_idx < app.visible_files.len() {
                    let display_info = &app.visible_files[visible_idx];
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
                } else {
                    debug_lines.push(String::from("(loading...)"));
                }
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
            if !app.visible_files.is_empty() && app.total_results > 0 {
                let visible_idx = app.selected_index.saturating_sub(app.visible_files_offset);
                if visible_idx < app.visible_files.len() {
                    let display_info = &app.visible_files[visible_idx];
                    let full_path_str = display_info.full_path.to_string_lossy().to_string();
                    let is_cached = app
                        .preview_cache
                        .as_ref()
                        .is_some_and(|(p, _)| p == &full_path_str);
                    let cache_status = if is_cached { "Cached" } else { "Live" };
                    debug_lines.push(format!("Preview: {}", cache_status));
                } else {
                    debug_lines.push(String::from("Preview: N/A"));
                }
            } else {
                debug_lines.push(String::from("Preview: N/A"));
            }

            debug_lines.push(String::from("")); // Separator

            // Add recent logs
            debug_lines.push(String::from("Recent Logs:"));
            // Show more log lines when debug is maximized
            let log_count = if app.debug_pane_maximized { 30 } else { 10 };
            let log_start = app.recent_logs.len().saturating_sub(log_count);
            for log_line in app.recent_logs.iter().skip(log_start) {
                // Truncate long lines to fit
                let max_len = if app.debug_pane_maximized { 120 } else { 60 };
                if log_line.len() > max_len {
                    debug_lines.push(format!("  {}...", &log_line[..(max_len - 3)]));
                } else {
                    debug_lines.push(format!("  {}", log_line));
                }
            }

            let debug_text = debug_lines.join("\n");

            let debug_title = if app.debug_pane_maximized {
                "Debug (Ctrl-O to minimize)"
            } else {
                "Debug (Ctrl-O to maximize)"
            };
            let debug_pane = Paragraph::new(debug_text)
                .block(Block::default().borders(Borders::ALL).title(debug_title));
            f.render_widget(debug_pane, top_chunks[2]);

            // Search input at the bottom
            let input = Paragraph::new(app.query.as_str())
                .block(Block::default().borders(Borders::ALL).title("Search"));
            f.render_widget(input, main_chunks[1]);
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
                    query,
                    total_results,
                    total_files,
                    visible_slice,
                    model_stats,
                } => {
                    // Only apply if query still matches
                    if query == app.query {
                        app.visible_files = visible_slice;
                        app.visible_files_offset = 0; // Query results always start at 0
                        app.total_results = total_results;
                        app.total_files = total_files;
                        app.model_stats_cache = model_stats;

                        // Reset selection if needed
                        if app.selected_index >= total_results {
                            app.selected_index = 0;
                        }

                        // Create subsession
                        app.current_subsession = Some(Subsession {
                            id: app.next_subsession_id,
                            query: query.clone(),
                            created_at: jiff::Timestamp::now(),
                            events_have_been_logged: false,
                        });
                        app.next_subsession_id += 1;
                    }
                }
                WorkerResponse::VisibleSlice(slice) => {
                    // Note: offset is set when we request the slice in update_scroll()
                    app.visible_files = slice;
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

                            // Log scroll event (deduplicated by query + full_path)
                            if !app.visible_files.is_empty() && app.total_results > 0 {
                                // Force log impressions before scroll
                                let _ = app.check_and_log_impressions(true);

                                let visible_idx = app.selected_index.saturating_sub(app.visible_files_offset);
                                if visible_idx < app.visible_files.len() {
                                    let display_info = &app.visible_files[visible_idx];
                                    let full_path_str =
                                        display_info.full_path.to_string_lossy().to_string();
                                    let key = (app.query.clone(), full_path_str.clone());

                                    if !app.scrolled_files.contains(&key) {
                                        let (mtime, atime, file_size) =
                                            get_file_metadata(&display_info.full_path);
                                        let subsession_id = app
                                            .current_subsession
                                            .as_ref()
                                            .map(|s| s.id)
                                            .unwrap_or(1);
                                        let _ = app.db.log_event(EventData {
                                            query: &app.query,
                                            file_path: &display_info.display_name,
                                            full_path: &full_path_str,
                                            mtime,
                                            atime,
                                            file_size,
                                            subsession_id,
                                            action: db::UserInteraction::Scroll,
                                            session_id: &app.session_id,
                                        });
                                        app.scrolled_files.insert(key);
                                    }
                                }
                            }
                        }
                        MouseEventKind::ScrollUp => {
                            app.preview_scroll = app.preview_scroll.saturating_sub(3);

                            // Log scroll event (deduplicated by query + full_path)
                            if !app.visible_files.is_empty() && app.total_results > 0 {
                                // Force log impressions before scroll
                                let _ = app.check_and_log_impressions(true);

                                let visible_idx = app.selected_index.saturating_sub(app.visible_files_offset);
                                if visible_idx < app.visible_files.len() {
                                    let display_info = &app.visible_files[visible_idx];
                                    let full_path_str =
                                        display_info.full_path.to_string_lossy().to_string();
                                    let key = (app.query.clone(), full_path_str.clone());

                                    if !app.scrolled_files.contains(&key) {
                                        let (mtime, atime, file_size) =
                                            get_file_metadata(&display_info.full_path);
                                        let subsession_id = app
                                            .current_subsession
                                            .as_ref()
                                            .map(|s| s.id)
                                            .unwrap_or(1);
                                        let _ = app.db.log_event(EventData {
                                            query: &app.query,
                                            file_path: &display_info.display_name,
                                            full_path: &full_path_str,
                                            mtime,
                                            atime,
                                            file_size,
                                            subsession_id,
                                            action: db::UserInteraction::Scroll,
                                            session_id: &app.session_id,
                                        });
                                        app.scrolled_files.insert(key);
                                    }
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
                            let _ = app
                                .worker_tx
                                .send(WorkerRequest::UpdateQuery(app.query.clone()));
                        }
                        KeyCode::Char('o')
                            if key.modifiers.contains(event::KeyModifiers::CONTROL) =>
                        {
                            // Ctrl-O: toggle debug pane maximization
                            app.debug_pane_maximized = !app.debug_pane_maximized;
                        }
                        KeyCode::Esc => {
                            return Ok(());
                        }
                        KeyCode::Char(c) => {
                            app.query.push(c);
                            let _ = app
                                .worker_tx
                                .send(WorkerRequest::UpdateQuery(app.query.clone()));
                        }
                        KeyCode::Backspace => {
                            app.query.pop();
                            let _ = app
                                .worker_tx
                                .send(WorkerRequest::UpdateQuery(app.query.clone()));
                        }
                        KeyCode::Up => {
                            app.move_selection(-1);
                        }
                        KeyCode::Down => {
                            app.move_selection(1);
                        }
                        KeyCode::Enter => {
                            if !app.visible_files.is_empty() && app.total_results > 0 {
                                // Force log impressions before click
                                app.check_and_log_impressions(true)?;

                                let visible_idx = app.selected_index.saturating_sub(app.visible_files_offset);
                                if visible_idx < app.visible_files.len() {
                                    let display_info = &app.visible_files[visible_idx];
                                    let (mtime, atime, file_size) =
                                        get_file_metadata(&display_info.full_path);

                                    // Log the click
                                    let subsession_id =
                                        app.current_subsession.as_ref().map(|s| s.id).unwrap_or(1);
                                    app.db.log_event(EventData {
                                        query: &app.query,
                                        file_path: &display_info.display_name,
                                        full_path: &display_info.full_path.to_string_lossy(),
                                        mtime,
                                        atime,
                                        file_size,
                                        subsession_id,
                                        action: db::UserInteraction::Click,
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
                                        .arg(&display_info.full_path)
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
                                    if let Err(e) = app.reload_model() {
                                        log::error!("Failed to reload model: {}", e);
                                    }

                                    // Reload click data and rerank files after editing
                                    if let Err(e) = app.reload_and_rerank() {
                                        log::error!("Failed to reload and rerank: {}", e);
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
