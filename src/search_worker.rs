use crate::db::Database;
use crate::ranker;
use crate::walker::start_file_walker;
use anyhow::Result;
use mpsc::{Receiver, Sender};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::mpsc::{self},
    time::Duration,
};

// These next three are types for communicating with the UI thread.
#[derive(Debug, Clone)]
pub struct DisplayFileInfo {
    pub display_name: String,
    pub full_path: PathBuf,
    pub score: f64,
    pub features: Vec<f64>,
}

pub enum WorkerRequest {
    UpdateQuery(String),
    GetVisibleSlice { start: usize, count: usize },
    ReloadModel,
    ReloadClicks,
}

pub enum WorkerResponse {
    QueryUpdated {
        query: String,
        total_results: usize,
        total_files: usize,
        visible_slice: Vec<DisplayFileInfo>,
        model_stats: Option<ranker::ModelStats>,
    },
    VisibleSlice(Vec<DisplayFileInfo>),
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
    origin: FileOrigin,
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
            origin: FileOrigin::CwdWalker,
        }
    }

    fn from_history(full_path: PathBuf, mtime: Option<i64>, root: &PathBuf) -> Self {
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
            origin: FileOrigin::UserClickedInEventsDb,
        }
    }
}

pub fn spawn(
    cwd: PathBuf,
    data_dir: &Path,
) -> Result<(Sender<WorkerRequest>, Receiver<WorkerResponse>)> {
    let (worker_tx, worker_task_rx) = mpsc::channel::<WorkerRequest>();
    let (worker_result_tx, worker_rx) = mpsc::channel::<WorkerResponse>();
    let (walker_tx, walker_rx) = mpsc::channel::<(PathBuf, Option<i64>)>();

    let data_dir = data_dir.to_path_buf();
    let cwd_clone = cwd.clone();
    std::thread::spawn(move || {
        // WorkerState is constructed in this thread so the ranker never moves
        // between thread. It's technically thread-safe to move for read-only
        // operations, but doing it this way means we don't have to do an unsafe
        // impl Send.
        let worker_state = WorkerState::new(cwd_clone, &data_dir).unwrap();
        worker_thread_loop(worker_task_rx, worker_result_tx, walker_rx, worker_state);
    });

    let root_clone = cwd.clone();
    std::thread::spawn(move || {
        start_file_walker(root_clone, walker_tx);
    });

    Ok((worker_tx, worker_rx))
}

// Worker thread state - owns all file data
struct WorkerState {
    file_registry: Vec<FileInfo>,
    path_to_id: HashMap<PathBuf, FileId>,
    filtered_files: Vec<FileId>,
    file_scores: Vec<ranker::FileScore>,
    current_query: String,
    root: PathBuf,
    ranker: ranker::Ranker,
    model_path: PathBuf,
    db_path: PathBuf,
}

impl WorkerState {
    fn new(root: PathBuf, data_dir: &Path) -> Result<Self> {
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
            ranker,
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
                    is_from_walker: file_info.origin == FileOrigin::CwdWalker,
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
                    model_stats: state.ranker.stats.clone(),
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
                    model_stats: state.ranker.stats.clone(),
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
                            model_stats: state.ranker.stats.clone(),
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

// Helper to drain all pending queries and return the latest one
fn drain_latest_query(rx: &mpsc::Receiver<WorkerRequest>, initial: String) -> String {
    let mut latest = initial;
    while let Ok(WorkerRequest::UpdateQuery(query)) = rx.try_recv() {
        latest = query;
    }
    latest
}

pub fn get_file_metadata(path: &PathBuf) -> (Option<i64>, Option<i64>, Option<i64>) {
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

pub fn get_time_ago(path: &PathBuf) -> String {
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
