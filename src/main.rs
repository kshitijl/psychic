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
    collections::{HashMap, HashSet, VecDeque},
    env,
    io::{self, stdout},
    path::{Path, PathBuf},
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
    created_at: jiff::Timestamp, // When subsession was created
    impressions_logged: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FileId(usize);

#[derive(Debug, Clone)]
struct FileInfo {
    full_path: PathBuf,
    display_name: String,
    mtime: Option<i64>,
    is_from_walker: bool,
}

impl FileInfo {
    fn from_walkdir(full_path: PathBuf, mtime: Option<i64>, root: &PathBuf) -> Self {
        let display_name = full_path
            .strip_prefix(root)
            .unwrap_or(&full_path)
            .to_string_lossy()
            .to_string();

        FileInfo {
            full_path,
            display_name,
            mtime,
            is_from_walker: true,
        }
    }

    fn from_history(full_path: PathBuf, mtime: Option<i64>, root: &PathBuf) -> Self {
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

        FileInfo {
            full_path,
            display_name,
            mtime,
            is_from_walker: false,
        }
    }
}

// Worker thread communication types
#[derive(Debug, Clone)]
struct DisplayFileInfo {
    display_name: String,
    full_path: PathBuf,
    score: f64,
    features: Vec<f64>,
}

enum WorkerRequest {
    UpdateQuery(String),
    GetVisibleSlice { start: usize, count: usize },
    ReloadModel,
    ReloadClicks,
}

enum WorkerResponse {
    QueryUpdated {
        query: String,
        total_results: usize,
        total_files: usize,
        visible_slice: Vec<DisplayFileInfo>,
        model_stats: Option<ranker::ModelStats>,
    },
    VisibleSlice(Vec<DisplayFileInfo>),
}

// Unsafe Send wrapper for Ranker (LightGBM is thread-safe for read-only ops)
struct SendRanker(ranker::Ranker);
unsafe impl Send for SendRanker {}

// Worker thread state - owns all file data
struct WorkerState {
    file_registry: Vec<FileInfo>,
    path_to_id: HashMap<PathBuf, FileId>,
    filtered_files: Vec<FileId>,
    file_scores: Vec<ranker::FileScore>,
    current_query: String,
    root: PathBuf,
    ranker: SendRanker,
    model_path: PathBuf,
    db_path: PathBuf,
}

impl WorkerState {
    fn new(
        root: PathBuf,
        ranker: ranker::Ranker,
        db: &Database,
        model_path: PathBuf,
        db_path: PathBuf,
    ) -> Result<Self> {
        // Load historical files into registry
        let mut file_registry = Vec::new();
        let mut path_to_id: HashMap<PathBuf, FileId> = HashMap::new();

        let historical_paths: Vec<PathBuf> = db
            .get_previously_interacted_files()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|p| {
                let path = PathBuf::from(&p);
                if path.exists() { Some(path) } else { None }
            })
            .collect();

        for path in historical_paths {
            // Canonicalize historical paths once at startup
            let canonical_path = path.canonicalize().unwrap_or(path);
            let (mtime, _, _) = get_file_metadata(&canonical_path);
            let file_info = FileInfo::from_history(canonical_path.clone(), mtime, &root);
            let file_id = FileId(file_registry.len());
            path_to_id.insert(canonical_path, file_id);
            file_registry.push(file_info);
        }

        log::debug!(
            "WorkerState loaded {} historical files",
            file_registry.len()
        );

        Ok(WorkerState {
            file_registry,
            path_to_id,
            filtered_files: Vec::new(),
            file_scores: Vec::new(),
            current_query: String::new(),
            root,
            ranker: SendRanker(ranker),
            model_path,
            db_path,
        })
    }

    fn add_file(&mut self, path: PathBuf, mtime: Option<i64>) {
        if !self.path_to_id.contains_key(&path) {
            let file_info = FileInfo::from_walkdir(path.clone(), mtime, &self.root);
            let file_id = FileId(self.file_registry.len());
            self.path_to_id.insert(path, file_id);
            self.file_registry.push(file_info);
        }
    }

    fn filter_and_rank(&mut self, query: &str) -> Result<()> {
        self.current_query = query.to_string();
        let query_lower = query.to_lowercase();

        // Filter files that match the query
        let matching_file_ids: Vec<FileId> = (0..self.file_registry.len())
            .map(FileId)
            .filter(|&file_id| {
                let file_info = &self.file_registry[file_id.0];
                query_lower.is_empty()
                    || file_info.display_name.to_lowercase().contains(&query_lower)
            })
            .collect();

        // Convert to FileCandidate structs
        let file_candidates: Vec<ranker::FileCandidate> = matching_file_ids
            .iter()
            .map(|&file_id| {
                let file_info = &self.file_registry[file_id.0];
                ranker::FileCandidate {
                    file_id: file_id.0,
                    relative_path: file_info.display_name.clone(),
                    full_path: file_info.full_path.clone(),
                    mtime: file_info.mtime,
                    is_from_walker: file_info.is_from_walker,
                }
            })
            .collect();

        // Rank them with the model
        let current_timestamp = jiff::Timestamp::now().as_second();
        match self
            .ranker
            .0
            .rank_files(query, &file_candidates, current_timestamp, &self.root)
        {
            Ok(scored) => {
                self.filtered_files = scored.iter().map(|fs| FileId(fs.file_id)).collect();
                self.file_scores = scored;
            }
            Err(e) => {
                log::warn!("Ranking failed: {}, falling back to simple filtering", e);
                self.file_scores.clear();
                self.filtered_files = matching_file_ids;
            }
        }

        Ok(())
    }

    fn get_slice(&self, start: usize, count: usize) -> Vec<DisplayFileInfo> {
        self.filtered_files
            .iter()
            .skip(start)
            .take(count)
            .map(|&file_id| {
                let file_info = &self.file_registry[file_id.0];
                let file_score = self.file_scores.iter().find(|fs| fs.file_id == file_id.0);

                let score = file_score.map(|fs| fs.score).unwrap_or(0.0);
                let features = file_score.map(|fs| fs.features.clone()).unwrap_or_default();

                DisplayFileInfo {
                    display_name: file_info.display_name.clone(),
                    full_path: file_info.full_path.clone(),
                    score,
                    features,
                }
            })
            .collect()
    }

    fn reload_model(&mut self) -> Result<()> {
        log::info!("Worker: Reloading model from disk");
        self.ranker.0 = ranker::Ranker::new(&self.model_path, self.db_path.clone())?;
        log::info!("Worker: Model reloaded successfully");
        Ok(())
    }

    fn reload_clicks(&mut self) -> Result<()> {
        log::info!("Worker: Reloading click data");
        self.ranker.0.clicks = ranker::Ranker::load_clicks(&self.db_path)?;
        log::info!("Worker: Click data reloaded successfully");
        Ok(())
    }
}

// Worker thread main loop
fn worker_thread_loop(
    task_rx: mpsc::Receiver<WorkerRequest>,
    result_tx: mpsc::Sender<WorkerResponse>,
    walker_rx: mpsc::Receiver<(PathBuf, Option<i64>)>,
    mut state: WorkerState,
) {
    use std::sync::mpsc::RecvTimeoutError;
    use std::time::Instant;

    let mut last_update = Instant::now();
    let mut files_changed = false;

    loop {
        // Process walker updates (non-blocking)
        while let Ok((path, mtime)) = walker_rx.try_recv() {
            state.add_file(path, mtime);
            files_changed = true;
        }

        // If files changed and enough time passed, send update
        if files_changed && last_update.elapsed() > Duration::from_millis(100) {
            if let Err(e) = state.filter_and_rank(&state.current_query.clone()) {
                log::error!("Auto-refresh filter/rank failed: {}", e);
            } else {
                let visible_slice = state.get_slice(0, 60);
                let _ = result_tx.send(WorkerResponse::QueryUpdated {
                    query: state.current_query.clone(),
                    total_results: state.filtered_files.len(),
                    total_files: state.file_registry.len(),
                    visible_slice,
                    model_stats: state.ranker.0.stats.clone(),
                });
                last_update = Instant::now();
                files_changed = false;
            }
        }

        // Wait for worker requests with timeout
        match task_rx.recv_timeout(Duration::from_millis(5)) {
            Ok(WorkerRequest::UpdateQuery(query)) => {
                // Debounce: drain all pending queries and keep the latest
                let query = drain_latest_query(&task_rx, query);
                state.current_query = query.clone();

                // Filter and rank
                if let Err(e) = state.filter_and_rank(&query) {
                    log::error!("Filter/rank failed: {}", e);
                    continue;
                }

                // Send back results with enough items to fill screen (assume ~50 lines)
                let visible_slice = state.get_slice(0, 60);
                let _ = result_tx.send(WorkerResponse::QueryUpdated {
                    query,
                    total_results: state.filtered_files.len(),
                    total_files: state.file_registry.len(),
                    visible_slice,
                    model_stats: state.ranker.0.stats.clone(),
                });
                last_update = Instant::now();
                files_changed = false;
            }
            Ok(WorkerRequest::GetVisibleSlice { start, count }) => {
                let slice = state.get_slice(start, count);
                let _ = result_tx.send(WorkerResponse::VisibleSlice(slice));
            }
            Ok(WorkerRequest::ReloadModel) => {
                if let Err(e) = state.reload_model() {
                    log::error!("Failed to reload model: {}", e);
                } else {
                    // Re-filter and rank with new model
                    let query = state.current_query.clone();
                    if let Err(e) = state.filter_and_rank(&query) {
                        log::error!("Filter/rank failed after model reload: {}", e);
                    } else {
                        let visible_slice = state.get_slice(0, 60);
                        let _ = result_tx.send(WorkerResponse::QueryUpdated {
                            query,
                            total_results: state.filtered_files.len(),
                            total_files: state.file_registry.len(),
                            visible_slice,
                            model_stats: state.ranker.0.stats.clone(),
                        });
                    }
                }
            }
            Ok(WorkerRequest::ReloadClicks) => {
                if let Err(e) = state.reload_clicks() {
                    log::error!("Failed to reload clicks: {}", e);
                } else {
                    // Re-filter and rank with new clicks
                    let query = state.current_query.clone();
                    if let Err(e) = state.filter_and_rank(&query) {
                        log::error!("Filter/rank failed after clicks reload: {}", e);
                    } else {
                        let visible_slice = state.get_slice(0, 60);
                        let _ = result_tx.send(WorkerResponse::QueryUpdated {
                            query,
                            total_results: state.filtered_files.len(),
                            total_files: state.file_registry.len(),
                            visible_slice,
                            model_stats: state.ranker.0.stats.clone(),
                        });
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                // No work to do, loop again
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                log::debug!("Worker thread channel disconnected");
                break;
            }
        }
    }
}

// Helper to drain all pending queries and return the latest one
fn drain_latest_query(rx: &mpsc::Receiver<WorkerRequest>, initial: String) -> String {
    let mut latest = initial;
    while let Ok(WorkerRequest::UpdateQuery(query)) = rx.try_recv() {
        latest = query;
    }
    latest
}

struct App {
    query: String,
    visible_files: Vec<DisplayFileInfo>, // Only what's currently visible
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
    model_stats_cache: Option<ranker::ModelStats>, // Cached from worker, refreshed periodically
    retraining: bool,
    log_receiver: Receiver<String>,
    recent_logs: VecDeque<String>,
    debug_maximized: bool,
    // Worker thread communication
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

        // Create channels for worker thread
        let (worker_tx, worker_task_rx) = mpsc::channel::<WorkerRequest>();
        let (worker_result_tx, worker_rx) = mpsc::channel::<WorkerResponse>();
        let (walker_tx, walker_rx) = mpsc::channel::<(PathBuf, Option<i64>)>();

        // Initialize worker state
        let worker_state = WorkerState::new(
            root.clone(),
            ranker,
            &db,
            model_path.clone(),
            db_path.clone(),
        )?;

        // Spawn worker thread
        std::thread::spawn(move || {
            worker_thread_loop(worker_task_rx, worker_result_tx, walker_rx, worker_state);
        });

        // Spawn walker thread
        let root_clone = root.clone();
        std::thread::spawn(move || {
            start_file_walker(root_clone, walker_tx);
        });

        log::debug!("App::new() total time: {:?}", start_time.elapsed());

        let app = App {
            query: String::new(),
            visible_files: Vec::new(),
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
            db,
            model_stats_cache: None,
            retraining: false,
            log_receiver,
            recent_logs: VecDeque::with_capacity(50),
            debug_maximized: false,
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
        if subsession.impressions_logged {
            return Ok(());
        }

        // Check if we should log: either forced or >200ms old
        let elapsed = jiff::Timestamp::now().duration_since(subsession.created_at);
        let threshold = jiff::SignedDuration::from_millis(200);
        let should_log = force || elapsed >= threshold;
        if !should_log {
            return Ok(());
        }

        // Log top 10 visible files with metadata
        let top_10: Vec<FileMetadata> = self
            .visible_files
            .iter()
            .take(10)
            .map(|display_info| {
                let (mtime, atime, file_size) = get_file_metadata(&display_info.full_path);
                (
                    display_info.display_name.clone(),
                    display_info.full_path.to_string_lossy().to_string(),
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

        // Request more visible items if we're getting close to the edge
        let scroll_offset = self.file_list_scroll as usize;
        let current_visible_end = scroll_offset + self.visible_files.len();
        let needed_end = scroll_offset + visible_height as usize + 10; // 10 item buffer

        if needed_end > current_visible_end && current_visible_end < self.total_results {
            // Request more items
            let count = (visible_height as usize * 2).max(60); // At least 60 or 2x screen height
            let _ = self.worker_tx.send(WorkerRequest::GetVisibleSlice {
                start: scroll_offset,
                count,
            });
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
        && let Ok(modified) = metadata.modified()
    {
        let duration = std::time::SystemTime::now()
            .duration_since(modified)
            .unwrap_or(Duration::from_secs(0));

        let formatter = timeago::Formatter::new();
        return formatter.convert(duration);
    }
    String::from("unknown")
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
        // Walker files are now handled by the worker thread

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

            // Split top area horizontally: left for file list, middle for preview, right for debug
            let top_chunks = if app.debug_maximized {
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
            let items: Vec<ListItem> = app
                .visible_files
                .iter()
                .enumerate()
                .map(|(visible_idx, display_info)| {
                    let i = scroll_offset + visible_idx;
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

            let scroll_offset = app.file_list_scroll as usize;
            let visible_idx = app.selected_index.saturating_sub(scroll_offset);

            let current_file = if visible_idx < app.visible_files.len() {
                Some(&app.visible_files[visible_idx])
            } else {
                None
            };

            let current_file_path = current_file.map(|x| x.full_path.to_string_lossy().to_string());

            // Preview on the right using bat (with smart caching)
            let preview_height = top_chunks[1].height.saturating_sub(2);
            let preview_text = if !app.visible_files.is_empty() && app.total_results > 0 {
                if current_file_path == None {
                    Text::from("[Loading preview...]")
                } else {
                    let current_file_path = current_file_path.as_ref().unwrap();
                    let full_path = PathBuf::from(&current_file_path);

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
                let scroll_offset = app.file_list_scroll as usize;
                let visible_idx = app.selected_index.saturating_sub(scroll_offset);
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
            if app.retraining {
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
                let scroll_offset = app.file_list_scroll as usize;
                let visible_idx = app.selected_index.saturating_sub(scroll_offset);
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
            let log_count = if app.debug_maximized { 30 } else { 10 };
            let log_start = app.recent_logs.len().saturating_sub(log_count);
            for log_line in app.recent_logs.iter().skip(log_start) {
                // Truncate long lines to fit
                let max_len = if app.debug_maximized { 120 } else { 60 };
                if log_line.len() > max_len {
                    debug_lines.push(format!("  {}...", &log_line[..(max_len - 3)]));
                } else {
                    debug_lines.push(format!("  {}", log_line));
                }
            }

            let debug_text = debug_lines.join("\n");

            let debug_title = if app.debug_maximized {
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
            app.retraining = retraining_status;
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
                            impressions_logged: false,
                        });
                        app.next_subsession_id += 1;
                    }
                }
                WorkerResponse::VisibleSlice(slice) => {
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

                                let scroll_offset = app.file_list_scroll as usize;
                                let visible_idx = app.selected_index.saturating_sub(scroll_offset);
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
                                        let _ = app.db.log_scroll(EventData {
                                            query: &app.query,
                                            file_path: &display_info.display_name,
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
                        }
                        MouseEventKind::ScrollUp => {
                            app.preview_scroll = app.preview_scroll.saturating_sub(3);

                            // Log scroll event (deduplicated by query + full_path)
                            if !app.visible_files.is_empty() && app.total_results > 0 {
                                // Force log impressions before scroll
                                let _ = app.check_and_log_impressions(true);

                                let scroll_offset = app.file_list_scroll as usize;
                                let visible_idx = app.selected_index.saturating_sub(scroll_offset);
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
                                        let _ = app.db.log_scroll(EventData {
                                            query: &app.query,
                                            file_path: &display_info.display_name,
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
                            app.debug_maximized = !app.debug_maximized;
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

                                let scroll_offset = app.file_list_scroll as usize;
                                let visible_idx = app.selected_index.saturating_sub(scroll_offset);
                                if visible_idx < app.visible_files.len() {
                                    let display_info = &app.visible_files[visible_idx];
                                    let (mtime, atime, file_size) =
                                        get_file_metadata(&display_info.full_path);

                                    // Log the click
                                    let subsession_id =
                                        app.current_subsession.as_ref().map(|s| s.id).unwrap_or(1);
                                    app.db.log_click(EventData {
                                        query: &app.query,
                                        file_path: &display_info.display_name,
                                        full_path: &display_info.full_path.to_string_lossy(),
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
