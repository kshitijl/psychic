use crate::db::Database;
use crate::ranker;
use crate::walker::start_file_walker;
use anyhow::Result;
use mpsc::{Receiver, Sender};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::mpsc::{self},
    thread::JoinHandle,
    time::Duration,
};

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
}

pub struct UpdateQueryRequest {
    pub query: String,
    pub query_id: u64,
}

pub enum WorkerRequest {
    UpdateQuery(UpdateQueryRequest),
    GetPage {
        query_id: u64,
        page_num: usize,
    },
    ReloadModel { query_id: u64 },
    ReloadClicks { query_id: u64 },
    ChangeCwd {
        new_cwd: PathBuf,
        query_id: u64,
    },
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
}

impl FileInfo {
    fn from_walkdir(
        full_path: PathBuf,
        mtime: Option<i64>,
        atime: Option<i64>,
        file_size: Option<i64>,
        is_dir: bool,
        root: &PathBuf,
    ) -> Self {
        let display_name = full_path
            .strip_prefix(root)
            .unwrap_or(&full_path)
            .to_string_lossy()
            .to_string();

        FileInfo {
            full_path,
            display_name,
            mtime,
            atime,
            file_size,
            origin: FileOrigin::CwdWalker,
            is_dir,
        }
    }

    fn from_history(
        full_path: PathBuf,
        mtime: Option<i64>,
        atime: Option<i64>,
        file_size: Option<i64>,
        is_dir: bool,
        root: &PathBuf,
    ) -> Self {
        let display_name = match full_path.strip_prefix(root) {
            Ok(postfix) => {
                // File is in current tree - show relative path
                postfix.to_string_lossy().to_string()
            }
            Err(_) => {
                // File is from elsewhere - show .../{filename} as a shorthand
                // for "it's not from this dir"
                let filename = full_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");
                format!(".../{}", filename)
            }
        };

        FileInfo {
            full_path,
            display_name,
            mtime,
            atime,
            file_size,
            origin: FileOrigin::UserClickedInEventsDb,
            is_dir,
        }
    }
}

pub fn spawn(
    cwd: PathBuf,
    data_dir: &Path,
) -> Result<(Sender<WorkerRequest>, Receiver<WorkerResponse>, JoinHandle<()>)> {
    let (worker_tx, worker_task_rx) = mpsc::channel::<WorkerRequest>();
    let (worker_result_tx, worker_rx) = mpsc::channel::<WorkerResponse>();
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
        let worker_state = WorkerState::new(cwd_clone, &data_dir, walker_command_tx_clone).unwrap();
        worker_thread_loop(worker_task_rx, worker_result_tx, walker_message_rx, worker_state);
    });

    // Start walker thread
    start_file_walker(cwd, walker_command_rx, walker_message_tx);

    Ok((worker_tx, worker_rx, worker_handle))
}

// Worker thread state - owns all file data
struct WorkerState {
    file_registry: Vec<FileInfo>,
    path_to_id: HashMap<PathBuf, FileId>,
    filtered_files: Vec<FileId>,
    file_scores: Vec<ranker::FileScore>,
    current_query: String,
    current_query_id: u64,
    root: PathBuf,
    ranker: ranker::Ranker,
    model_path: PathBuf,
    db_path: PathBuf,
    walker_command_tx: Sender<WalkerCommand>,
}

impl WorkerState {
    fn new(root: PathBuf, data_dir: &Path, walker_command_tx: Sender<WalkerCommand>) -> Result<Self> {
        let ranker_start = jiff::Timestamp::now();
        let model_path = data_dir.join("model.txt");

        let db_path = Database::get_db_path(data_dir);

        let ranker = ranker::Ranker::new(&model_path, &db_path).unwrap();
        log::info!("Loaded ranking model from {:?}", model_path);
        log::debug!(
            "Ranker initialization took {:?}",
            jiff::Timestamp::now() - ranker_start
        );

        // Load historical files.
        let mut file_registry = Vec::new();
        let mut path_to_id: HashMap<PathBuf, FileId> = HashMap::new();

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

        for path in historical_paths {
            // Canonicalize historical paths and get their metadata once, at
            // startup, to minimize syscalls later during searches.
            let canonical_path = path.canonicalize().unwrap_or(path);
            let metadata = get_file_metadata(&canonical_path);
            let file_info = FileInfo::from_history(
                canonical_path.clone(),
                metadata.mtime,
                metadata.atime,
                metadata.file_size,
                metadata.is_dir,
                &root,
            );
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
            current_query_id: 0,
            root,
            ranker,
            model_path,
            db_path,
            walker_command_tx,
        })
    }

    fn add_file(&mut self, path: PathBuf, mtime: Option<i64>, atime: Option<i64>, file_size: Option<i64>, is_dir: bool) {
        if !self.path_to_id.contains_key(&path) {
            let file_info = FileInfo::from_walkdir(path.clone(), mtime, atime, file_size, is_dir, &self.root);
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

                // Do not include the root directory in results
                if file_info.full_path == self.root {
                    return false;
                }

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
                    is_from_walker: file_info.origin == FileOrigin::CwdWalker,
                    is_dir: file_info.is_dir,
                }
            })
            .collect();

        // Rank them with the model
        let current_timestamp = jiff::Timestamp::now().as_second();
        match self
            .ranker
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

                DisplayFileInfo {
                    display_name: file_info.display_name.clone(),
                    full_path: file_info.full_path.clone(),
                    score,
                    features,
                    mtime: file_info.mtime,
                    atime: file_info.atime,
                    file_size: file_info.file_size,
                    is_dir: file_info.is_dir,
                }
            })
            .collect()
    }

    fn get_page(&self, page_num: usize, page_size: usize) -> PageData {
        // Precondition: page_size must be reasonable (non-zero)
        assert!(page_size > 0, "page_size must be positive, got {}", page_size);

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

    fn reload_model(&mut self) -> Result<()> {
        log::info!("Worker: Reloading model from disk");
        self.ranker = ranker::Ranker::new(&self.model_path, &self.db_path)?;
        log::info!("Worker: Model reloaded successfully");
        Ok(())
    }

    fn reload_clicks(&mut self) -> Result<()> {
        log::info!("Worker: Reloading click data");
        self.ranker.clicks = ranker::Ranker::load_clicks(&self.db_path)?;
        log::info!("Worker: Click data reloaded successfully");
        Ok(())
    }

    fn change_cwd(&mut self, new_cwd: PathBuf) -> Result<()> {
        log::info!("Worker: Changing cwd from {:?} to {:?}", self.root, new_cwd);

        // Remove all walker-sourced files from registry
        self.file_registry.retain(|f| f.origin != FileOrigin::CwdWalker);

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
                    // File is from elsewhere - show .../{filename} as a shorthand
                    let filename = file.full_path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");
                    format!(".../{}", filename)
                }
            };
        }

        // Rebuild path_to_id map (only keep historical files)
        self.path_to_id.clear();
        for (idx, file) in self.file_registry.iter().enumerate() {
            self.path_to_id.insert(file.full_path.clone(), FileId(idx));
        }

        // Send command to walker thread to change directory
        self.walker_command_tx.send(WalkerCommand::ChangeCwd(new_cwd))?;

        log::info!("Worker: CWD change command sent to walker");
        Ok(())
    }
}
// Worker thread main loop
fn worker_thread_loop(
    task_rx: mpsc::Receiver<WorkerRequest>,
    result_tx: mpsc::Sender<WorkerResponse>,
    walker_rx: mpsc::Receiver<WalkerMessage>,
    mut state: WorkerState,
) {
    use std::sync::mpsc::RecvTimeoutError;
    use std::time::Instant;

    let mut last_files_changed_notification = Instant::now();

    loop {
        // Process walker updates (non-blocking)
        let mut files_changed = false;
        let mut walker_done = false;
        while let Ok(message) = walker_rx.try_recv() {
            match message {
                WalkerMessage::FileMetadata(metadata) => {
                    state.add_file(metadata.path, metadata.mtime, metadata.atime, metadata.file_size, metadata.is_dir);
                    files_changed = true;
                }
                WalkerMessage::AllDone => {
                    walker_done = true;
                    files_changed = true;
                }
            }
        }

        // If files changed, notify the UI so it can decide to trigger a refresh.
        // We debounce this to avoid spamming the UI thread, UNLESS the walker is done.
        if files_changed && (walker_done || last_files_changed_notification.elapsed() > Duration::from_millis(200)) {
            let _ = result_tx.send(WorkerResponse::FilesChanged);
            last_files_changed_notification = Instant::now();
        }

        // Wait for worker requests with timeout
        match task_rx.recv_timeout(Duration::from_millis(5)) {
            Ok(WorkerRequest::UpdateQuery(update_req)) => {
                // Debounce: drain all pending queries and keep the latest
                let latest_req = drain_latest_update_request(&task_rx, update_req);
                state.current_query = latest_req.query.clone();
                state.current_query_id = latest_req.query_id;

                // Filter and rank
                if let Err(e) = state.filter_and_rank(&latest_req.query) {
                    log::error!("Filter/rank failed: {}", e);
                    continue;
                }

                // Send back results with initial page (page 0)
                let initial_page = state.get_page(0, 128);
                let _ = result_tx.send(WorkerResponse::QueryUpdated {
                    query_id: latest_req.query_id,
                    total_results: state.filtered_files.len(),
                    total_files: state.file_registry.len(),
                    initial_page,
                    model_stats: state.ranker.stats.clone(),
                });
            }
            Ok(WorkerRequest::GetPage { query_id, page_num }) => {
                // If the request is for an old query, ignore it.
                if query_id != state.current_query_id {
                    continue;
                }
                let page_data = state.get_page(page_num, 128);
                let _ = result_tx.send(WorkerResponse::Page { query_id, page_data });
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
                        let _ = result_tx.send(WorkerResponse::QueryUpdated {
                            query_id,
                            total_results: state.filtered_files.len(),
                            total_files: state.file_registry.len(),
                            initial_page,
                            model_stats: state.ranker.stats.clone(),
                        });
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
                        let _ = result_tx.send(WorkerResponse::QueryUpdated {
                            query_id,
                            total_results: state.filtered_files.len(),
                            total_files: state.file_registry.len(),
                            initial_page,
                            model_stats: state.ranker.stats.clone(),
                        });
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
                        let _ = result_tx.send(WorkerResponse::QueryUpdated {
                            query_id,
                            total_results: state.filtered_files.len(),
                            total_files: state.file_registry.len(),
                            initial_page,
                            model_stats: state.ranker.stats.clone(),
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

