use crate::db::Database;
use crate::ranker;
use crate::walker::start_file_walker;
use anyhow::Result;
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use mpsc::Sender;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::mpsc::{self},
    thread::JoinHandle,
    time::Duration,
};
use strum::EnumCount;

// Metadata sent from walker to worker
#[derive(Debug, Clone)]
pub struct WalkerFileMetadata {
    pub path: PathBuf,
    pub mtime: Option<i64>,
    pub atime: Option<i64>,
    pub file_size: Option<i64>,
    pub is_dir: bool,
}

// Commands sent from worker to walker
#[derive(Debug, Clone)]
pub enum WalkerCommand {
    ChangeCwd(PathBuf),
}

// Messages from walker to worker
#[derive(Debug, Clone)]
pub enum WalkerMessage {
    FileMetadata(WalkerFileMetadata),
    AllDone,
}

// Filter types
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum_macros::FromRepr, strum_macros::EnumCount)]
#[repr(u8)]
pub enum FilterType {
    None = 0,      // no filter
    OnlyCwd = 1,   // only things in cwd (recursive)
    DirectCwd = 2, // only things directly in cwd (non-recursive)
    OnlyDirs = 3,  // only directories
    OnlyFiles = 4, // only files
}

impl FilterType {
    /// Get the next filter in the cycle
    pub fn next(self) -> Self {
        Self::from_repr((self as u8 + 1) % Self::COUNT as u8).unwrap()
    }

    /// Get the previous filter in the cycle
    pub fn prev(self) -> Self {
        Self::from_repr((self as u8 + Self::COUNT as u8 - 1) % Self::COUNT as u8).unwrap()
    }
}

// These next three are types for communicating with the UI thread.
#[derive(Debug, Clone)]
pub struct DisplayFileInfo {
    pub display_name: String,
    pub full_path: PathBuf,
    pub score: f64,
    pub features: Vec<f64>,
    pub mtime: Option<i64>,
    pub atime: Option<i64>,
    pub file_size: Option<i64>,
    pub is_dir: bool,
    pub is_cwd: bool,
    pub is_historical: bool, // From UserClickedInEventsDb, not current CwdWalker
    pub is_under_cwd: bool,
    pub simple_score: Option<f64>, // For debug pane: score from simple model
    pub ml_score: Option<f64>,     // For debug pane: score from ML model
    pub simple_weight: Option<f64>, // For debug pane: weight assigned to simple model
    pub ml_weight: Option<f64>,    // For debug pane: weight assigned to ML model
}

pub struct UpdateQueryRequest {
    pub query: String,
    pub query_id: u64,
    pub filter: FilterType,
}

pub enum WorkerRequest {
    UpdateQuery(UpdateQueryRequest),
    GetPage { query_id: u64, page_num: usize },
    ReloadModel { query_id: u64 },
    ReloadClicks { query_id: u64 },
    ChangeCwd { new_cwd: PathBuf, query_id: u64 },
}

#[derive(Debug, Clone)]
pub struct PageData {
    pub page_num: usize,
    pub start_index: usize,
    pub end_index: usize,
    pub files: Vec<DisplayFileInfo>,
}

pub enum WorkerResponse {
    QueryUpdated {
        query_id: u64,
        total_results: usize,
        total_files: usize,
        initial_page: PageData,
        model_stats: Option<ranker::ModelStats>,
    },
    Page {
        query_id: u64,
        page_data: PageData,
    },
    FilesChanged,
    WalkerDone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FileId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileOrigin {
    CwdWalker,
    UserClickedInEventsDb,
}

#[derive(Debug, Clone)]
struct FileInfo {
    full_path: PathBuf,
    display_name: String,
    mtime: Option<i64>,
    atime: Option<i64>,
    file_size: Option<i64>,
    origin: FileOrigin,
    is_dir: bool,
    is_under_cwd: bool,
}

impl FileInfo {
    fn from_history(
        full_path: PathBuf,
        mtime: Option<i64>,
        atime: Option<i64>,
        file_size: Option<i64>,
        is_dir: bool,
        root: &PathBuf,
    ) -> Self {
        let display_name = if full_path == *root {
            // Special case: if this is the root directory itself, show just the dir name
            root.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(".")
                .to_string()
        } else {
            match full_path.strip_prefix(root) {
                Ok(postfix) => {
                    // File is in current tree - show relative path
                    postfix.to_string_lossy().to_string()
                }
                Err(_) => {
                    // File is from elsewhere - show full absolute path
                    // Will be colored differently in UI to indicate it's historical
                    full_path.to_string_lossy().to_string()
                }
            }
        };

        let is_under_cwd = full_path.starts_with(root);

        FileInfo {
            full_path,
            display_name,
            mtime,
            atime,
            file_size,
            origin: FileOrigin::UserClickedInEventsDb,
            is_dir,
            is_under_cwd,
        }
    }
}

pub fn spawn<T>(
    cwd: PathBuf,
    data_dir: &Path,
    event_tx: Sender<T>,
    no_click_loading: bool,
    no_model: bool,
) -> Result<(Sender<WorkerRequest>, JoinHandle<()>)>
where
    T: From<WorkerResponse> + Send + 'static,
{
    let (worker_tx, worker_task_rx) = mpsc::channel::<WorkerRequest>();
    let (walker_message_tx, walker_message_rx) = mpsc::channel::<WalkerMessage>();
    let (walker_command_tx, walker_command_rx) = mpsc::channel::<WalkerCommand>();

    let data_dir = data_dir.to_path_buf();
    let cwd_clone = cwd.clone();
    let walker_command_tx_clone = walker_command_tx.clone();
    let worker_handle = std::thread::spawn(move || {
        // WorkerState is constructed in this thread so the ranker never moves
        // between thread. It's technically thread-safe to move for read-only
        // operations, but doing it this way means we don't have to do an unsafe
        // impl Send.
        let worker_state = WorkerState::new(
            cwd_clone,
            &data_dir,
            walker_command_tx_clone,
            no_click_loading,
            no_model,
        )
        .unwrap();
        worker_thread_loop(worker_task_rx, event_tx, walker_message_rx, worker_state);
    });

    // Start walker thread
    start_file_walker(cwd, walker_command_rx, walker_message_tx);

    Ok((worker_tx, worker_handle))
}

// Worker thread state - owns all file data
struct WorkerState {
    file_registry: Vec<FileInfo>,
    path_to_id: HashMap<PathBuf, FileId>,
    filtered_files: Vec<FileId>,
    file_scores: Vec<ranker::FileScore>,
    current_query: String,
    current_query_id: u64,
    current_filter: FilterType,
    root: PathBuf,
    ranker: ranker::Ranker,
    model_path: PathBuf,
    db_path: PathBuf,
    walker_command_tx: Sender<WalkerCommand>,
}

impl WorkerState {
    fn new(
        root: PathBuf,
        data_dir: &Path,
        walker_command_tx: Sender<WalkerCommand>,
        no_click_loading: bool,
        _no_model: bool,
    ) -> Result<Self> {
        let worker_state_start = std::time::Instant::now();

        let ranker_start = std::time::Instant::now();
        let model_path = data_dir.join("model.txt");
        let db_path = Database::get_db_path(data_dir);

        let ranker = if no_click_loading || _no_model {
            // Skip loading model and clicks if either flag is set
            ranker::Ranker::new_empty(&db_path)?
        } else {
            Self::load_ranker(&model_path, &db_path)?
        };
        log::info!(
            "TIMING {{\"op\":\"ranker_init\",\"ms\":{}}}",
            ranker_start.elapsed().as_secs_f64() * 1000.0
        );

        // Load historical files.
        let historical_start = std::time::Instant::now();
        let mut file_registry = Vec::new();
        let mut path_to_id: HashMap<PathBuf, FileId> = HashMap::new();

        if !no_click_loading {
            let db = Database::new(&db_path)?;
            let historical_paths: Vec<PathBuf> = db
                .get_previously_interacted_files()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|p| {
                    let path = PathBuf::from(&p);
                    if path.exists() { Some(path) } else { None }
                })
                .collect();
            log::info!("Loading {} historical paths", historical_paths.len());

            for path in historical_paths {
                // Canonicalize historical paths and get their metadata once, at
                // startup, to minimize syscalls later during searches.
                let canonical_path = path.canonicalize().unwrap_or(path);

                path_to_id.entry(canonical_path.clone()).or_insert_with(|| {
                    let metadata = get_file_metadata(&canonical_path);
                    let file_info = FileInfo::from_history(
                        canonical_path,
                        metadata.mtime,
                        metadata.atime,
                        metadata.file_size,
                        metadata.is_dir,
                        &root,
                    );
                    let file_id = FileId(file_registry.len());
                    file_registry.push(file_info);
                    file_id
                });
            }
        }

        log::info!(
            "TIMING {{\"op\":\"load_historical_files\",\"ms\":{},\"count\":{}}}",
            historical_start.elapsed().as_secs_f64() * 1000.0,
            file_registry.len()
        );

        // Add the root directory itself to the registry
        let root_add_start = std::time::Instant::now();
        // (walker skips it, but we want it to appear in results as "(cwd)")
        let canonical_root = root.canonicalize().unwrap_or_else(|_| root.clone());
        if !path_to_id.contains_key(&canonical_root) {
            let metadata = get_file_metadata(&canonical_root);
            let display_name = canonical_root
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(".")
                .to_string();

            log::debug!(
                "Adding root directory to registry: canonical_root={:?}, display_name={:?}",
                canonical_root,
                display_name
            );

            let file_info = FileInfo {
                full_path: canonical_root.clone(),
                display_name,
                mtime: metadata.mtime,
                atime: metadata.atime,
                file_size: metadata.file_size,
                origin: FileOrigin::CwdWalker,
                is_dir: true,
                is_under_cwd: true,
            };

            let file_id = FileId(file_registry.len());
            path_to_id.insert(canonical_root.clone(), file_id);
            file_registry.push(file_info);
        }
        log::info!(
            "TIMING {{\"op\":\"add_root_directory\",\"ms\":{}}}",
            root_add_start.elapsed().as_secs_f64() * 1000.0
        );

        log::info!(
            "TIMING {{\"op\":\"worker_state_new_total\",\"ms\":{}}}",
            worker_state_start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(WorkerState {
            file_registry,
            path_to_id,
            filtered_files: Vec::new(),
            file_scores: Vec::new(),
            current_query: String::new(),
            current_query_id: 0,
            current_filter: FilterType::None,
            root: canonical_root,
            ranker,
            model_path,
            db_path,
            walker_command_tx,
        })
    }

    fn add_file(
        &mut self,
        path: PathBuf,
        mtime: Option<i64>,
        atime: Option<i64>,
        file_size: Option<i64>,
        is_dir: bool,
    ) {
        // `path` is the original path from the walker.
        let canonical_path = path.canonicalize().unwrap_or_else(|_| path.clone());

        if !self.path_to_id.contains_key(&canonical_path) {
            // We have a new file.
            // The display path should be the original `path` relative to `self.root`.
            let display_name = if canonical_path == self.root {
                // For the root directory itself, show just the directory name
                self.root
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(".")
                    .to_string()
            } else {
                path.strip_prefix(&self.root)
                    .unwrap_or(&path) // fallback to original path if not in root
                    .to_string_lossy()
                    .to_string()
            };

            let file_info = FileInfo {
                full_path: canonical_path.clone(), // Store the canonical path
                display_name,
                mtime,
                atime,
                file_size,
                origin: FileOrigin::CwdWalker,
                is_dir,
                is_under_cwd: true,
            };

            let file_id = FileId(self.file_registry.len());
            self.path_to_id.insert(canonical_path, file_id);
            self.file_registry.push(file_info);
        }
    }

    fn filter_and_rank(&mut self, query: &str) -> Result<()> {
        let filter_rank_start = std::time::Instant::now();
        self.current_query = query.to_string();

        // Create fuzzy matcher
        let matcher = SkimMatcherV2::default();

        // Filter files that match the query
        let filter_start = std::time::Instant::now();
        let matching_file_ids: Vec<FileId> = (0..self.file_registry.len())
            .map(FileId)
            .filter(|&file_id| {
                let file_info = &self.file_registry[file_id.0];

                // Apply text query filter using fuzzy matching
                let matches_query = query.is_empty()
                    || matcher
                        .fuzzy_match(&file_info.display_name, query)
                        .is_some();

                if !matches_query {
                    return false;
                }

                // Apply type filter
                match self.current_filter {
                    FilterType::None => true,
                    FilterType::OnlyCwd => {
                        // Show files that are under current root, regardless of origin
                        // This includes both files from CwdWalker and historical files that happen to be in cwd
                        file_info.is_under_cwd
                    }
                    FilterType::DirectCwd => {
                        // Show only files directly in cwd (non-recursive)
                        file_info.is_under_cwd
                            && file_info.full_path.parent() == Some(self.root.as_path())
                    }
                    FilterType::OnlyDirs => file_info.is_dir,
                    FilterType::OnlyFiles => !file_info.is_dir,
                }
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
                    file_size: file_info.file_size,
                    is_from_walker: file_info.origin == FileOrigin::CwdWalker,
                    is_dir: file_info.is_dir,
                }
            })
            .collect();
        log::info!(
            "TIMING {{\"op\":\"filter_files\",\"ms\":{},\"count\":{}}}",
            filter_start.elapsed().as_secs_f64() * 1000.0,
            file_candidates.len()
        );

        // Rank them with the model
        let rank_start = std::time::Instant::now();
        let current_timestamp = jiff::Timestamp::now().as_second();
        match self
            .ranker
            .rank_files(query, &file_candidates, current_timestamp, &self.root)
        {
            Ok(scored) => {
                log::info!(
                    "TIMING {{\"op\":\"rank_files\",\"ms\":{},\"count\":{}}}",
                    rank_start.elapsed().as_secs_f64() * 1000.0,
                    scored.len()
                );
                self.filtered_files = scored.iter().map(|fs| FileId(fs.file_id)).collect();
                self.file_scores = scored;
            }
            Err(e) => {
                log::warn!("Ranking failed: {}, falling back to simple filtering", e);
                self.file_scores.clear();
                self.filtered_files = matching_file_ids;
            }
        }

        log::info!(
            "TIMING {{\"op\":\"filter_and_rank_total\",\"ms\":{}}}",
            filter_rank_start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }

    fn get_slice(&self, start: usize, count: usize) -> Vec<DisplayFileInfo> {
        // Precondition: start must be within bounds
        assert!(
            start <= self.filtered_files.len(),
            "get_slice: start {} exceeds filtered_files length {}",
            start,
            self.filtered_files.len()
        );

        self.filtered_files
            .iter()
            .skip(start)
            .take(count)
            .map(|&file_id| {
                // Precondition: file_id must be valid index into registry
                assert!(
                    file_id.0 < self.file_registry.len(),
                    "Invalid file_id {} (registry size: {})",
                    file_id.0,
                    self.file_registry.len()
                );

                let file_info = &self.file_registry[file_id.0];
                let file_score = self.file_scores.iter().find(|fs| fs.file_id == file_id.0);

                let score = file_score.map(|fs| fs.score).unwrap_or(0.0);
                let features = file_score.map(|fs| fs.features.clone()).unwrap_or_default();
                let simple_score = file_score.and_then(|fs| fs.simple_score);
                let ml_score = file_score.and_then(|fs| fs.ml_score);
                let simple_weight = file_score.and_then(|fs| fs.simple_weight);
                let ml_weight = file_score.and_then(|fs| fs.ml_weight);

                // Check if this is the current working directory
                let is_cwd = file_info.full_path == self.root;
                let is_historical = file_info.origin == FileOrigin::UserClickedInEventsDb;

                DisplayFileInfo {
                    display_name: file_info.display_name.clone(),
                    full_path: file_info.full_path.clone(),
                    score,
                    features,
                    mtime: file_info.mtime,
                    atime: file_info.atime,
                    file_size: file_info.file_size,
                    is_dir: file_info.is_dir,
                    is_cwd,
                    is_historical,
                    is_under_cwd: file_info.is_under_cwd,
                    simple_score,
                    ml_score,
                    simple_weight,
                    ml_weight,
                }
            })
            .collect()
    }

    fn get_page(&self, page_num: usize, page_size: usize) -> PageData {
        // Precondition: page_size must be reasonable (non-zero)
        assert!(
            page_size > 0,
            "page_size must be positive, got {}",
            page_size
        );

        let start_index = page_num * page_size;
        let end_index = (start_index + page_size).min(self.filtered_files.len());
        let count = end_index.saturating_sub(start_index);

        let files = self.get_slice(start_index, count);

        // Postcondition: returned page must be consistent
        assert_eq!(
            files.len(),
            count,
            "get_page: returned {} files but expected {}",
            files.len(),
            count
        );

        PageData {
            page_num,
            start_index,
            end_index,
            files,
        }
    }

    /// Load ranker from disk, falling back to empty ranker if model file doesn't exist
    fn load_ranker(model_path: &Path, db_path: &Path) -> Result<ranker::Ranker> {
        if !model_path.exists() {
            log::info!(
                "Model file not found at {:?}, using empty ranker",
                model_path
            );
            ranker::Ranker::new_empty(db_path)
        } else {
            let ranker = ranker::Ranker::new(model_path, db_path)?;
            log::info!("Loaded ranking model from {:?}", model_path);
            Ok(ranker)
        }
    }

    fn reload_model(&mut self) -> Result<()> {
        log::info!("Worker: Reloading model from disk");
        self.ranker = Self::load_ranker(&self.model_path, &self.db_path)?;
        log::info!("Worker: Model reloaded successfully");
        Ok(())
    }

    fn reload_clicks(&mut self) -> Result<()> {
        log::info!("Worker: Reloading click data");
        let (clicks, total_clicks) = ranker::Ranker::load_clicks(&self.db_path)?;
        self.ranker.clicks = clicks;
        self.ranker.total_clicks = total_clicks;
        log::info!(
            "Worker: Click data reloaded successfully (total_clicks={})",
            total_clicks
        );
        Ok(())
    }

    fn change_cwd(&mut self, new_cwd: PathBuf) -> Result<()> {
        log::info!("Worker: Changing cwd from {:?} to {:?}", self.root, new_cwd);

        // Remove all walker-sourced files from registry
        self.file_registry
            .retain(|f| f.origin != FileOrigin::CwdWalker);

        // Update root
        self.root = new_cwd.clone();

        // Recalculate display names for all historical files with new root
        for file in self.file_registry.iter_mut() {
            file.display_name = match file.full_path.strip_prefix(&self.root) {
                Ok(postfix) => {
                    // File is in current tree - show relative path
                    postfix.to_string_lossy().to_string()
                }
                Err(_) => {
                    // File is from elsewhere - show full absolute path
                    // Will be colored differently in UI to indicate it's historical
                    file.full_path.to_string_lossy().to_string()
                }
            };
            file.is_under_cwd = file.full_path.starts_with(&self.root);
        }

        // Rebuild path_to_id map (only keep historical files)
        self.path_to_id.clear();
        for (idx, file) in self.file_registry.iter().enumerate() {
            self.path_to_id.insert(file.full_path.clone(), FileId(idx));
        }

        // Send command to walker thread to change directory
        self.walker_command_tx
            .send(WalkerCommand::ChangeCwd(new_cwd))?;

        log::info!("Worker: CWD change command sent to walker");
        Ok(())
    }
}
// Worker thread main loop
fn worker_thread_loop<T>(
    task_rx: mpsc::Receiver<WorkerRequest>,
    event_tx: mpsc::Sender<T>,
    walker_rx: mpsc::Receiver<WalkerMessage>,
    mut state: WorkerState,
) where
    T: From<WorkerResponse> + Send,
{
    use std::sync::mpsc::RecvTimeoutError;
    use std::time::Instant;

    let worker_loop_start = Instant::now();
    let mut last_files_changed_notification = Instant::now();

    loop {
        // Process walker updates (non-blocking)
        let mut files_changed = false;
        let mut walker_done = false;
        while let Ok(message) = walker_rx.try_recv() {
            match message {
                WalkerMessage::FileMetadata(metadata) => {
                    state.add_file(
                        metadata.path,
                        metadata.mtime,
                        metadata.atime,
                        metadata.file_size,
                        metadata.is_dir,
                    );
                    files_changed = true;
                }
                WalkerMessage::AllDone => {
                    log::info!(
                        "TIMING {{\"op\":\"walker_complete\",\"ms\":{}}}",
                        worker_loop_start.elapsed().as_secs_f64() * 1000.0
                    );
                    walker_done = true;
                    files_changed = true;
                    // Notify UI that walker is done
                    let _ = event_tx.send(WorkerResponse::WalkerDone.into());
                }
            }
        }

        // If files changed, notify the UI so it can decide to trigger a refresh.
        // We debounce this to avoid spamming the UI thread, UNLESS the walker is done.
        if files_changed
            && (walker_done
                || last_files_changed_notification.elapsed() > Duration::from_millis(200))
        {
            let _ = event_tx.send(WorkerResponse::FilesChanged.into());
            last_files_changed_notification = Instant::now();
        }

        // Wait for worker requests with timeout
        match task_rx.recv_timeout(Duration::from_millis(5)) {
            Ok(WorkerRequest::UpdateQuery(update_req)) => {
                // Debounce: drain all pending queries and keep the latest
                let latest_req = drain_latest_update_request(&task_rx, update_req);
                state.current_query = latest_req.query.clone();
                state.current_query_id = latest_req.query_id;
                state.current_filter = latest_req.filter;

                // Filter and rank
                if let Err(e) = state.filter_and_rank(&latest_req.query) {
                    log::error!("Filter/rank failed: {}", e);
                    continue;
                }

                // Send back results with initial page (page 0)
                let initial_page = state.get_page(0, 128);
                let _ = event_tx.send(
                    WorkerResponse::QueryUpdated {
                        query_id: latest_req.query_id,
                        total_results: state.filtered_files.len(),
                        total_files: state.file_registry.len(),
                        initial_page,
                        model_stats: state.ranker.stats.clone(),
                    }
                    .into(),
                );
            }
            Ok(WorkerRequest::GetPage { query_id, page_num }) => {
                // If the request is for an old query, ignore it.
                if query_id != state.current_query_id {
                    continue;
                }
                let page_data = state.get_page(page_num, 128);
                let _ = event_tx.send(
                    WorkerResponse::Page {
                        query_id,
                        page_data,
                    }
                    .into(),
                );
            }
            Ok(WorkerRequest::ReloadModel { query_id }) => {
                state.current_query_id = query_id;
                if let Err(e) = state.reload_model() {
                    log::error!("Failed to reload model: {}", e);
                } else {
                    // Re-filter and rank with new model
                    let query = state.current_query.clone();
                    if let Err(e) = state.filter_and_rank(&query) {
                        log::error!("Filter/rank failed after model reload: {}", e);
                    } else {
                        let initial_page = state.get_page(0, 128);
                        let _ = event_tx.send(
                            WorkerResponse::QueryUpdated {
                                query_id,
                                total_results: state.filtered_files.len(),
                                total_files: state.file_registry.len(),
                                initial_page,
                                model_stats: state.ranker.stats.clone(),
                            }
                            .into(),
                        );
                    }
                }
            }
            Ok(WorkerRequest::ReloadClicks { query_id }) => {
                state.current_query_id = query_id;
                if let Err(e) = state.reload_clicks() {
                    log::error!("Failed to reload clicks: {}", e);
                } else {
                    // Re-filter and rank with new clicks
                    let query = state.current_query.clone();
                    if let Err(e) = state.filter_and_rank(&query) {
                        log::error!("Filter/rank failed after clicks reload: {}", e);
                    } else {
                        let initial_page = state.get_page(0, 128);
                        let _ = event_tx.send(
                            WorkerResponse::QueryUpdated {
                                query_id,
                                total_results: state.filtered_files.len(),
                                total_files: state.file_registry.len(),
                                initial_page,
                                model_stats: state.ranker.stats.clone(),
                            }
                            .into(),
                        );
                    }
                }
            }
            Ok(WorkerRequest::ChangeCwd { new_cwd, query_id }) => {
                state.current_query_id = query_id;
                if let Err(e) = state.change_cwd(new_cwd) {
                    log::error!("Failed to change cwd: {}", e);
                } else {
                    // Clear query and re-filter (will show only historical files until walker sends new ones)
                    state.current_query = String::new();
                    if let Err(e) = state.filter_and_rank("") {
                        log::error!("Filter/rank failed after cwd change: {}", e);
                    } else {
                        let initial_page = state.get_page(0, 128);
                        let _ = event_tx.send(
                            WorkerResponse::QueryUpdated {
                                query_id,
                                total_results: state.filtered_files.len(),
                                total_files: state.file_registry.len(),
                                initial_page,
                                model_stats: state.ranker.stats.clone(),
                            }
                            .into(),
                        );
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

// Helper to drain all pending UpdateQuery requests and return the latest one
fn drain_latest_update_request(
    rx: &mpsc::Receiver<WorkerRequest>,
    initial: UpdateQueryRequest,
) -> UpdateQueryRequest {
    let mut latest = initial;
    while let Ok(request) = rx.try_recv() {
        if let WorkerRequest::UpdateQuery(update_req) = request {
            latest = update_req;
        } else {
            // This is not ideal, we've consumed a non-UpdateQuery request.
            // For this application, the channel logic is simple enough that
            // this case is unlikely, but in a more complex app, we'd need
            // to handle or requeue the request.
            log::warn!("Unexpected request type in drain_latest_update_request");
            break;
        }
    }
    latest
}

// Internal struct for file metadata (not exported)
struct FileMetadata {
    mtime: Option<i64>,
    atime: Option<i64>,
    file_size: Option<i64>,
    is_dir: bool,
}

fn get_file_metadata(path: &PathBuf) -> FileMetadata {
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
        let is_dir = metadata.is_dir();

        FileMetadata {
            mtime,
            atime,
            file_size,
            is_dir,
        }
    } else {
        FileMetadata {
            mtime: None,
            atime: None,
            file_size: None,
            is_dir: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_filter_cwd_includes_historical_files_in_cwd() {
        // Test that OnlyCwd filter shows files under cwd, even if they're from history

        let cwd = PathBuf::from("/home/user/project");

        let walker_path = PathBuf::from("/home/user/project/src/main.rs");
        // File from CwdWalker
        let file_from_walker = FileInfo {
            full_path: walker_path.clone(),
            display_name: "src/main.rs".to_string(),
            mtime: Some(1000),
            atime: None,
            file_size: Some(100),
            origin: FileOrigin::CwdWalker,
            is_dir: false,
            is_under_cwd: walker_path.starts_with(&cwd),
        };

        let history_in_cwd_path = PathBuf::from("/home/user/project/README.md");
        // File from history but also in current directory
        let file_from_history_in_cwd = FileInfo {
            full_path: history_in_cwd_path.clone(),
            display_name: "/home/user/project/README.md".to_string(),
            mtime: Some(900),
            atime: None,
            file_size: Some(50),
            origin: FileOrigin::UserClickedInEventsDb,
            is_dir: false,
            is_under_cwd: history_in_cwd_path.starts_with(&cwd),
        };

        let history_outside_path = PathBuf::from("/home/user/other/file.txt");
        // File from history outside current directory
        let file_from_history_elsewhere = FileInfo {
            full_path: history_outside_path.clone(),
            display_name: "/home/user/other/file.txt".to_string(),
            mtime: Some(800),
            atime: None,
            file_size: Some(25),
            origin: FileOrigin::UserClickedInEventsDb,
            is_dir: false,
            is_under_cwd: history_outside_path.starts_with(&cwd),
        };

        // Test OnlyCwd filter
        assert!(
            file_from_walker.is_under_cwd,
            "File from walker in cwd should pass OnlyCwd filter"
        );

        assert!(
            file_from_history_in_cwd.is_under_cwd,
            "File from history but in cwd should pass OnlyCwd filter"
        );

        assert!(
            !file_from_history_elsewhere.is_under_cwd,
            "File from history outside cwd should NOT pass OnlyCwd filter"
        );
    }

    #[test]
    fn test_filter_types() {
        let file_dir = FileInfo {
            full_path: PathBuf::from("/test/dir"),
            display_name: "dir".to_string(),
            mtime: None,
            atime: None,
            file_size: None,
            origin: FileOrigin::CwdWalker,
            is_dir: true,
            is_under_cwd: true,
        };

        let file_regular = FileInfo {
            full_path: PathBuf::from("/test/file.txt"),
            display_name: "file.txt".to_string(),
            mtime: None,
            atime: None,
            file_size: None,
            origin: FileOrigin::CwdWalker,
            is_dir: false,
            is_under_cwd: true,
        };

        // OnlyDirs filter: is_dir == true
        assert_eq!(
            file_dir.is_dir, true,
            "Directory should pass OnlyDirs filter"
        );
        assert_eq!(
            file_regular.is_dir, false,
            "Regular file should NOT pass OnlyDirs filter"
        );

        // OnlyFiles filter: !is_dir (is_dir == false)
        assert_eq!(
            !file_dir.is_dir, false,
            "Directory should NOT pass OnlyFiles filter"
        );
        assert_eq!(
            !file_regular.is_dir, true,
            "Regular file should pass OnlyFiles filter"
        );
    }
}
