use crate::db::Database;
use crate::metadata_ext::MetadataExt;
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
    /// Walk somewhere else. `hidden` is the set of hidden directories that
    /// apply to `path`, already narrowed by the worker, so the walker can skip
    /// those subtrees instead of walking and then discarding them.
    ChangeCwd { path: PathBuf, hidden: Vec<PathBuf> },
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
    pub fuzzy_score: i64,          // For debug pane: fuzzy match score from skim matcher
}

pub struct UpdateQueryRequest {
    pub query: String,
    pub query_id: u64,
    pub filter: FilterType,
}

pub enum WorkerRequest {
    UpdateQuery(UpdateQueryRequest),
    GetPage {
        query_id: u64,
        page_num: usize,
    },
    ReloadModel {
        query_id: u64,
    },
    ReloadClicks {
        query_id: u64,
    },
    ChangeCwd {
        new_cwd: PathBuf,
        query_id: u64,
    },
    /// Drop a path from the results because it is no longer on disk.
    Evict {
        path: PathBuf,
        query_id: u64,
    },
    /// Stop showing a directory and everything under it, from now on.
    Hide {
        path: PathBuf,
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
        /// How long filtering and ranking took, in milliseconds. The UI shows this
        /// next to the round trip so the two can be told apart: this is the work,
        /// the difference is the channel hop and scheduling.
        rank_ms: f64,
    },
    Page {
        query_id: u64,
        page_data: PageData,
    },
    FilesChanged,
    /// The filesystem walk finished. Carries the same elapsed time that goes into
    /// the `walker_complete` TIMING line, so the debug pane and the log report one
    /// number rather than two measured a channel hop apart.
    WalkerDone {
        walk_ms: f64,
    },
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
    /// Set when the path was found missing from disk at the moment the user
    /// acted on it. Evicted entries stay in the registry so `FileId` indices
    /// (held by `filtered_files` and the UI's page cache) remain valid, but
    /// `filter_and_rank` never emits them again.
    evicted: bool,
    /// Set when this path sits under a hidden prefix that does not contain the
    /// current root - see `WorkerState::recompute_hidden`. Recomputed whenever
    /// the root or the hidden set changes, so filtering stays a bool test.
    hidden: bool,
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
            evicted: false,
            // Set by recompute_hidden once the whole registry is built.
            hidden: false,
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

    // Read once here and hand to both threads: the walker needs it before
    // WorkerState exists, and a second connection on the startup path would
    // buy nothing. The table only grows when the user hides something, so it
    // is a handful of rows.
    let hidden_prefixes = Database::new(&Database::get_db_path(&data_dir))
        .and_then(|db| db.get_hidden_prefixes())
        .unwrap_or_else(|e| {
            log::error!("Failed to load hidden directories: {}", e);
            Vec::new()
        });
    log::info!("Loaded {} hidden directories", hidden_prefixes.len());

    let canonical_cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());
    let walker_hidden = active_hidden_for(&hidden_prefixes, &canonical_cwd);

    let cwd_clone = cwd.clone();
    let hidden_for_worker = hidden_prefixes;
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
            hidden_for_worker,
        )
        .unwrap();
        worker_thread_loop(worker_task_rx, event_tx, walker_message_rx, worker_state);
    });

    // Start walker thread
    // The canonical form, because hidden prefixes are stored canonical and the
    // walker compares entry paths against them directly. Passing the raw cwd
    // would silently stop the skipping from matching if the two ever diverged.
    start_file_walker(
        canonical_cwd,
        walker_hidden,
        walker_command_rx,
        walker_message_tx,
    );

    Ok((worker_tx, worker_handle))
}

/// Whether `path` sits under one of the currently active hidden prefixes.
///
/// `starts_with` compares whole path components, so hiding `/a/b` does not hide
/// `/a/bcd`.
fn is_hidden_by(active_hidden: &[PathBuf], path: &Path) -> bool {
    active_hidden.iter().any(|prefix| path.starts_with(prefix))
}

/// The hidden directories that actually suppress anything when the search root
/// is `root`.
///
/// A hidden directory is suppressed everywhere *except* from inside it: if the
/// root is at or under a hidden prefix, the user has deliberately navigated in
/// there, and hiding the contents of the directory they are standing in would
/// leave them staring at an empty screen with no explanation. Everywhere else -
/// including from the parent - it stays out of the way, which is the point.
///
/// The alternative rule, "show it whenever it happens to be under the current
/// root", does not work: running psychic from a parent directory puts the
/// walker inside the hidden tree and the hiding stops meaning anything.
fn active_hidden_for(hidden: &[PathBuf], root: &Path) -> Vec<PathBuf> {
    hidden
        .iter()
        .filter(|prefix| !root.starts_with(prefix))
        .cloned()
        .collect()
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
    /// How long the last filter-and-rank took, reported back with its results.
    last_rank_ms: f64,
    /// Every directory the user has hidden, as stored.
    hidden_prefixes: Vec<PathBuf>,
    /// The subset of `hidden_prefixes` that suppresses anything right now:
    /// those that do not contain `root`. Derived, never stored.
    active_hidden: Vec<PathBuf>,
}

impl WorkerState {
    fn new(
        root: PathBuf,
        data_dir: &Path,
        walker_command_tx: Sender<WalkerCommand>,
        no_click_loading: bool,
        _no_model: bool,
        hidden_prefixes: Vec<PathBuf>,
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
                evicted: false,
                hidden: false,
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

        let mut state = WorkerState {
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
            last_rank_ms: 0.0,
            hidden_prefixes,
            active_hidden: Vec::new(),
        };
        state.recompute_hidden();

        Ok(state)
    }

    /// Work out which hidden directories apply right now, and mark the registry.
    ///
    /// Called whenever the root or the hidden set changes - see
    /// [`active_hidden_for`] for the rule - so that the query path only ever
    /// tests a bool.
    fn recompute_hidden(&mut self) {
        self.active_hidden = active_hidden_for(&self.hidden_prefixes, &self.root);

        for file_info in self.file_registry.iter_mut() {
            file_info.hidden = is_hidden_by(&self.active_hidden, &file_info.full_path);
        }
    }

    /// Hide `path` and everything under it, from now on and in future sessions.
    ///
    /// Returns whether anything changed.
    fn hide(&mut self, path: PathBuf) -> Result<bool> {
        assert!(
            !self.root.starts_with(&path),
            "Refused in input.rs: hiding an ancestor of the root would hide nothing now and \
             everything later"
        );

        if self.hidden_prefixes.contains(&path) {
            return Ok(false);
        }

        Database::new(&self.db_path)?.hide_prefix(&path)?;
        log::info!("Worker: hiding {:?}", path);
        self.hidden_prefixes.push(path);
        self.recompute_hidden();

        Ok(true)
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

        if let Some(&file_id) = self.path_to_id.get(&canonical_path) {
            // Already registered. If it had been evicted as missing, the walker
            // has just seen it on disk again, so it is real: put it back.
            let file_info = &mut self.file_registry[file_id.0];
            if file_info.evicted {
                log::info!("Worker: un-evicting rediscovered path {:?}", canonical_path);
                file_info.evicted = false;
            }
        } else {
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
                evicted: false,
                hidden: is_hidden_by(&self.active_hidden, &canonical_path),
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

        // Filter files that match the query and capture fuzzy scores
        let filter_start = std::time::Instant::now();
        let matching_files: Vec<(FileId, i64)> = (0..self.file_registry.len())
            .map(FileId)
            .filter_map(|file_id| {
                let file_info = &self.file_registry[file_id.0];

                // Gone from disk, or under a directory the user hid. Both are
                // checked first so a suppressed entry costs a bool test rather
                // than a fuzzy match.
                if file_info.evicted || file_info.hidden {
                    return None;
                }

                // Apply text query filter using fuzzy matching
                let fuzzy_score = if query.is_empty() {
                    // Empty query matches everything with max score
                    i64::MAX
                } else {
                    matcher.fuzzy_match(&file_info.display_name, query)?
                };

                // Apply type filter
                let passes_type_filter = match self.current_filter {
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
                };

                if passes_type_filter {
                    Some((file_id, fuzzy_score))
                } else {
                    None
                }
            })
            .collect();

        // Convert to FileCandidate structs
        let file_candidates: Vec<ranker::FileCandidate> = matching_files
            .iter()
            .map(|&(file_id, fuzzy_score)| {
                let file_info = &self.file_registry[file_id.0];
                ranker::FileCandidate {
                    file_id: file_id.0,
                    relative_path: file_info.display_name.clone(),
                    full_path: file_info.full_path.clone(),
                    mtime: file_info.mtime,
                    file_size: file_info.file_size,
                    is_from_walker: file_info.origin == FileOrigin::CwdWalker,
                    is_dir: file_info.is_dir,
                    fuzzy_score,
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
                self.filtered_files = matching_files.iter().map(|&(file_id, _)| file_id).collect();
            }
        }

        self.last_rank_ms = filter_rank_start.elapsed().as_secs_f64() * 1000.0;
        log::info!(
            "TIMING {{\"op\":\"filter_and_rank_total\",\"ms\":{}}}",
            self.last_rank_ms
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
                let fuzzy_score = file_score.map(|fs| fs.fuzzy_score).unwrap_or(0);

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
                    fuzzy_score,
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

    /// Load ranker from disk, falling back to an empty ranker if the model is
    /// missing or unreadable.
    ///
    /// An unusable model must never stop psychic from starting: ranking degrades
    /// to the simple model, which is what a fresh install runs on anyway, and the
    /// retrain kicked off at launch replaces the bad file. Training writes the
    /// model atomically so a half-written file should not arise, but a model
    /// truncated by a killed older build, a full disk, or an interrupted copy
    /// would otherwise brick startup until the user knew to delete it.
    fn load_ranker(model_path: &Path, db_path: &Path) -> Result<ranker::Ranker> {
        if !model_path.exists() {
            log::info!(
                "Model file not found at {:?}, using empty ranker",
                model_path
            );
            return ranker::Ranker::new_empty(db_path);
        }

        match ranker::Ranker::new(model_path, db_path) {
            Ok(ranker) => {
                log::info!("Loaded ranking model from {:?}", model_path);
                Ok(ranker)
            }
            Err(e) => {
                log::error!(
                    "Failed to load model at {:?} ({}); falling back to the simple \
                     model until the next retrain finishes",
                    model_path,
                    e
                );
                ranker::Ranker::new_empty(db_path)
            }
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

    /// Stop showing `path`, which the UI found missing from disk.
    ///
    /// The registry entry is marked rather than removed: `FileId` is an index
    /// into `file_registry`, and those indices are held by `filtered_files` and
    /// by the pages already sent to the UI.
    ///
    /// Returns whether a registered path was actually evicted.
    fn evict(&mut self, path: &Path) -> bool {
        // The UI acts on `full_path`, which is canonical, so no canonicalizing
        // here - and the path is gone anyway, so canonicalize would fail.
        let Some(&file_id) = self.path_to_id.get(path) else {
            log::warn!("Worker: asked to evict unregistered path {:?}", path);
            return false;
        };

        let file_info = &mut self.file_registry[file_id.0];
        if file_info.evicted {
            return false;
        }
        file_info.evicted = true;
        log::info!("Worker: evicted missing path {:?}", path);
        true
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

        // The root moved, so which hidden directories apply moved with it: one
        // that contains the new root is now exempt, and one that no longer
        // contains it starts applying again.
        self.recompute_hidden();

        // Send command to walker thread to change directory
        self.walker_command_tx.send(WalkerCommand::ChangeCwd {
            path: new_cwd,
            hidden: self.active_hidden.clone(),
        })?;

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
                    // Measured from process start, not from when this loop began,
                    // so it lines up with first_render / first_query_complete /
                    // startup_complete and with the debug pane.
                    let walk_ms = crate::PROCESS_START.elapsed().as_secs_f64() * 1000.0;
                    log::info!("TIMING {{\"op\":\"walker_complete\",\"ms\":{}}}", walk_ms);
                    walker_done = true;
                    files_changed = true;
                    // Notify UI that walker is done
                    let _ = event_tx.send(WorkerResponse::WalkerDone { walk_ms }.into());
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
                        rank_ms: state.last_rank_ms,
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
                                rank_ms: state.last_rank_ms,
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
                                rank_ms: state.last_rank_ms,
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
                                rank_ms: state.last_rank_ms,
                            }
                            .into(),
                        );
                    }
                }
            }
            Ok(WorkerRequest::Evict { path, query_id }) => {
                state.current_query_id = query_id;
                // Re-run the current query so the missing row disappears
                // immediately. Nothing to do if the path was not registered.
                if state.evict(&path) {
                    let query = state.current_query.clone();
                    if let Err(e) = state.filter_and_rank(&query) {
                        log::error!("Filter/rank failed after eviction: {}", e);
                    } else {
                        let initial_page = state.get_page(0, 128);
                        let _ = event_tx.send(
                            WorkerResponse::QueryUpdated {
                                query_id,
                                total_results: state.filtered_files.len(),
                                total_files: state.file_registry.len(),
                                initial_page,
                                model_stats: state.ranker.stats.clone(),
                                rank_ms: state.last_rank_ms,
                            }
                            .into(),
                        );
                    }
                }
            }
            Ok(WorkerRequest::Hide { path, query_id }) => {
                state.current_query_id = query_id;
                match state.hide(path) {
                    Ok(false) => {}
                    Ok(true) => {
                        // Re-run the current query so the rows go at once.
                        let query = state.current_query.clone();
                        if let Err(e) = state.filter_and_rank(&query) {
                            log::error!("Filter/rank failed after hiding: {}", e);
                        } else {
                            let initial_page = state.get_page(0, 128);
                            let _ = event_tx.send(
                                WorkerResponse::QueryUpdated {
                                    query_id,
                                    total_results: state.filtered_files.len(),
                                    total_files: state.file_registry.len(),
                                    initial_page,
                                    model_stats: state.ranker.stats.clone(),
                                    rank_ms: state.last_rank_ms,
                                }
                                .into(),
                            );
                        }
                    }
                    Err(e) => log::error!("Failed to hide directory: {}", e),
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
        let mtime = metadata.mtime_as_secs();
        let atime = metadata.atime_as_secs();
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
            evicted: false,
            hidden: false,
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
            evicted: false,
            hidden: false,
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
            evicted: false,
            hidden: false,
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
            evicted: false,
            hidden: false,
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
            evicted: false,
            hidden: false,
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

#[cfg(test)]
mod load_ranker_tests {
    use super::*;

    /// A scratch directory with a real (empty) events database in it.
    struct TempDataDir {
        path: PathBuf,
    }

    impl TempDataDir {
        fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("psychic-test-{}-{}", name, std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("Failed to create temp data dir");

            // load_ranker reads click history, so the schema has to exist.
            crate::db::Database::new(&path.join("events.db")).expect("Failed to create test db");

            Self { path }
        }

        fn db_path(&self) -> PathBuf {
            self.path.join("events.db")
        }

        fn model_path(&self) -> PathBuf {
            self.path.join("model.txt")
        }
    }

    impl Drop for TempDataDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn test_missing_model_falls_back_to_simple_ranking() {
        let dir = TempDataDir::new("missing-model");

        let ranker = WorkerState::load_ranker(&dir.model_path(), &dir.db_path())
            .expect("A missing model must not stop psychic from starting");

        assert!(
            !ranker.has_model(),
            "Nothing to load, so ranking runs on the simple model"
        );
    }

    #[test]
    fn test_truncated_model_falls_back_instead_of_failing_startup() {
        let dir = TempDataDir::new("truncated-model");
        // What a write interrupted partway through leaves behind.
        std::fs::write(&dir.model_path(), "").expect("Failed to write empty model");

        let ranker = WorkerState::load_ranker(&dir.model_path(), &dir.db_path())
            .expect("An empty model file must not stop psychic from starting");

        assert!(!ranker.has_model(), "The unusable model was not loaded");
    }

    #[test]
    fn test_corrupt_model_falls_back_instead_of_failing_startup() {
        let dir = TempDataDir::new("corrupt-model");
        std::fs::write(&dir.model_path(), "this is not a LightGBM model")
            .expect("Failed to write corrupt model");

        let ranker = WorkerState::load_ranker(&dir.model_path(), &dir.db_path())
            .expect("A corrupt model must not stop psychic from starting");

        assert!(!ranker.has_model(), "The unusable model was not loaded");
    }
}

#[cfg(test)]
mod eviction_tests {
    use super::*;

    /// A scratch directory with a real (empty) events database in it.
    pub(super) struct TempDataDir {
        pub(super) path: PathBuf,
    }

    impl TempDataDir {
        pub(super) fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("psychic-test-{}-{}", name, std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("Failed to create temp data dir");
            crate::db::Database::new(&path.join("events.db")).expect("Failed to create test db");
            Self { path }
        }
    }

    impl Drop for TempDataDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// A worker holding three walker-discovered files under `/test`.
    ///
    /// The paths do not exist on disk, which is what we want: `add_file`
    /// canonicalizes, and canonicalizing a missing path leaves it unchanged, so
    /// the registry holds exactly the paths written here.
    pub(super) fn worker_with_three_files(
        name: &str,
    ) -> (WorkerState, TempDataDir, mpsc::Receiver<WalkerCommand>) {
        let dir = TempDataDir::new(name);
        let (walker_command_tx, walker_command_rx) = mpsc::channel::<WalkerCommand>();

        let mut state = WorkerState::new(
            PathBuf::from("/test"),
            &dir.path,
            walker_command_tx,
            true, // no_click_loading: keep the registry to just what we add
            true, // no_model
            Vec::new(),
        )
        .expect("Worker state should build against an empty data dir");

        // `new` registers the root itself; drop it so the test sees only files.
        state.file_registry.clear();
        state.path_to_id.clear();

        for name in ["alpha.txt", "beta.txt", "gamma.txt"] {
            state.add_file(
                PathBuf::from("/test").join(name),
                Some(1000),
                Some(1000),
                Some(10),
                false,
            );
        }

        (state, dir, walker_command_rx)
    }

    /// A worker rooted at `root` holding one file inside `/test/old` and one
    /// outside it, with `/test/old` already hidden.
    pub(super) fn worker_across_hidden_boundary(
        root: &str,
        name: &str,
    ) -> (WorkerState, TempDataDir, mpsc::Receiver<WalkerCommand>) {
        let dir = TempDataDir::new(name);
        let (walker_command_tx, walker_command_rx) = mpsc::channel::<WalkerCommand>();

        let mut state = WorkerState::new(
            PathBuf::from(root),
            &dir.path,
            walker_command_tx,
            true,
            true,
            vec![PathBuf::from("/test/old")],
        )
        .expect("Worker state should build against an empty data dir");

        state.file_registry.clear();
        state.path_to_id.clear();

        for path in ["/test/old/notes.txt", "/test/current/notes.txt"] {
            state.add_file(PathBuf::from(path), Some(1000), Some(1000), Some(10), false);
        }

        (state, dir, walker_command_rx)
    }

    pub(super) fn results(state: &WorkerState) -> Vec<String> {
        state
            .filtered_files
            .iter()
            .map(|id| state.file_registry[id.0].display_name.clone())
            .collect()
    }

    #[test]
    fn test_evicted_file_disappears_from_results() {
        let (mut state, _dir, _rx) = worker_with_three_files("evict-basic");

        state.filter_and_rank("").expect("Filter should succeed");
        let mut before = results(&state);
        before.sort();
        assert_eq!(
            before,
            vec!["alpha.txt", "beta.txt", "gamma.txt"],
            "All three files show before anything is evicted"
        );

        assert!(
            state.evict(&PathBuf::from("/test/beta.txt")),
            "Evicting a registered path reports that it did something"
        );

        state.filter_and_rank("").expect("Filter should succeed");
        let mut after = results(&state);
        after.sort();
        assert_eq!(
            after,
            vec!["alpha.txt", "gamma.txt"],
            "The evicted file is gone and the others are untouched"
        );
    }

    #[test]
    fn test_evicted_file_stays_gone_for_a_matching_query() {
        let (mut state, _dir, _rx) = worker_with_three_files("evict-query");

        state.evict(&PathBuf::from("/test/beta.txt"));

        state
            .filter_and_rank("beta")
            .expect("Filter should succeed");
        assert_eq!(
            results(&state),
            Vec::<String>::new(),
            "A query that matches only the evicted file finds nothing"
        );
    }

    #[test]
    fn test_evicting_unknown_path_is_a_no_op() {
        let (mut state, _dir, _rx) = worker_with_three_files("evict-unknown");

        assert!(
            !state.evict(&PathBuf::from("/test/never-registered.txt")),
            "An unregistered path reports that nothing was evicted"
        );
        assert!(
            state.evict(&PathBuf::from("/test/beta.txt")),
            "First eviction of a real path succeeds"
        );
        assert!(
            !state.evict(&PathBuf::from("/test/beta.txt")),
            "Evicting the same path twice reports no further change"
        );

        state.filter_and_rank("").expect("Filter should succeed");
        assert_eq!(
            results(&state).len(),
            2,
            "Only the one real eviction took effect"
        );
    }

    #[test]
    fn test_walker_rediscovery_un_evicts() {
        // A path can come back: deleted and recreated, or a directory the user
        // navigated away from and back to. The walker seeing it on disk is
        // proof it exists, so the eviction should not outlive that.
        let (mut state, _dir, _rx) = worker_with_three_files("evict-rediscover");

        state.evict(&PathBuf::from("/test/beta.txt"));
        state.filter_and_rank("").expect("Filter should succeed");
        assert_eq!(results(&state).len(), 2, "Evicted file is hidden");

        state.add_file(
            PathBuf::from("/test/beta.txt"),
            Some(2000),
            Some(2000),
            Some(20),
            false,
        );

        state.filter_and_rank("").expect("Filter should succeed");
        let mut after = results(&state);
        after.sort();
        assert_eq!(
            after,
            vec!["alpha.txt", "beta.txt", "gamma.txt"],
            "Rediscovered file is back, and was not registered a second time"
        );
    }
}

#[cfg(test)]
mod hiding_tests {
    use super::eviction_tests::*;
    use super::*;

    #[test]
    fn test_active_hidden_excludes_prefixes_containing_the_root() {
        let hidden = vec![PathBuf::from("/test/old"), PathBuf::from("/other")];

        assert_eq!(
            active_hidden_for(&hidden, &PathBuf::from("/test")),
            vec![PathBuf::from("/test/old"), PathBuf::from("/other")],
            "From the parent, both still apply"
        );
        assert_eq!(
            active_hidden_for(&hidden, &PathBuf::from("/test/old")),
            vec![PathBuf::from("/other")],
            "Standing in a hidden directory exempts it, and only it"
        );
        assert_eq!(
            active_hidden_for(&hidden, &PathBuf::from("/test/old/deeper")),
            vec![PathBuf::from("/other")],
            "Standing below a hidden directory exempts it too"
        );
    }

    #[test]
    fn test_hidden_prefix_matches_whole_components_only() {
        let hidden = vec![PathBuf::from("/test/old")];

        assert!(
            is_hidden_by(&hidden, &PathBuf::from("/test/old/notes.txt")),
            "A file inside the hidden directory is hidden"
        );
        assert!(
            is_hidden_by(&hidden, &PathBuf::from("/test/old")),
            "The hidden directory itself is hidden"
        );
        assert!(
            !is_hidden_by(&hidden, &PathBuf::from("/test/older/notes.txt")),
            "A sibling sharing a name prefix is not hidden"
        );
        assert!(
            !is_hidden_by(&hidden, &PathBuf::from("/test/notes.txt")),
            "A file outside the hidden directory is not hidden"
        );
    }

    #[test]
    fn test_hidden_files_do_not_show_from_outside() {
        let (mut state, _dir, _rx) = worker_across_hidden_boundary("/test", "hide-outside");

        state
            .filter_and_rank("notes")
            .expect("Filter should succeed");
        assert_eq!(
            results(&state),
            vec!["current/notes.txt"],
            "From the parent, only the file outside the hidden directory shows"
        );
    }

    #[test]
    fn test_hidden_files_show_from_inside() {
        // Navigating into a hidden directory has to work, or the user is left
        // staring at an empty screen with no explanation.
        let (mut state, _dir, _rx) = worker_across_hidden_boundary("/test/old", "hide-inside");

        state
            .filter_and_rank("notes")
            .expect("Filter should succeed");
        let mut found = results(&state);
        found.sort();
        assert_eq!(
            found,
            vec!["/test/current/notes.txt", "notes.txt"],
            "Inside the hidden directory everything shows normally"
        );
    }

    #[test]
    fn test_hiding_takes_effect_without_a_restart() {
        let (mut state, _dir, _rx) = worker_with_three_files("hide-live");

        state.filter_and_rank("").expect("Filter should succeed");
        assert_eq!(results(&state).len(), 3, "Everything shows to begin with");

        // Root is /test, so hide a subdirectory of it rather than an ancestor.
        state.add_file(
            PathBuf::from("/test/sub/buried.txt"),
            Some(1000),
            Some(1000),
            Some(10),
            false,
        );
        state.filter_and_rank("").expect("Filter should succeed");
        assert_eq!(results(&state).len(), 4, "The new file shows before hiding");

        assert!(
            state
                .hide(PathBuf::from("/test/sub"))
                .expect("Hiding should persist"),
            "Hiding a directory reports that it changed something"
        );

        state.filter_and_rank("").expect("Filter should succeed");
        let mut after = results(&state);
        after.sort();
        assert_eq!(
            after,
            vec!["alpha.txt", "beta.txt", "gamma.txt"],
            "The file under the hidden directory is gone, the rest untouched"
        );
    }

    #[test]
    fn test_hiding_the_same_directory_twice_changes_nothing() {
        let (mut state, _dir, _rx) = worker_with_three_files("hide-twice");

        assert!(
            state.hide(PathBuf::from("/test/sub")).unwrap(),
            "First hide takes effect"
        );
        assert!(
            !state.hide(PathBuf::from("/test/sub")).unwrap(),
            "Hiding again reports no change"
        );
    }

    #[test]
    fn test_hiding_survives_a_restart() {
        let (mut state, dir, _rx) = worker_with_three_files("hide-persist");
        state.hide(PathBuf::from("/test/sub")).unwrap();

        let reloaded = crate::db::Database::new(&crate::db::Database::get_db_path(&dir.path))
            .unwrap()
            .get_hidden_prefixes()
            .unwrap();

        assert_eq!(
            reloaded,
            vec![PathBuf::from("/test/sub")],
            "The hidden directory is in the database for the next session"
        );
    }

    #[test]
    fn test_unhiding_removes_it() {
        let (mut state, dir, _rx) = worker_with_three_files("hide-undo");
        state.hide(PathBuf::from("/test/sub")).unwrap();

        let db = crate::db::Database::new(&crate::db::Database::get_db_path(&dir.path)).unwrap();
        assert!(
            db.unhide_prefix(&PathBuf::from("/test/sub")).unwrap(),
            "Unhiding a hidden directory reports that it did something"
        );
        assert!(
            !db.unhide_prefix(&PathBuf::from("/test/sub")).unwrap(),
            "Unhiding it again reports no change"
        );
        assert!(
            db.get_hidden_prefixes().unwrap().is_empty(),
            "Nothing is hidden any more"
        );
    }

    #[test]
    fn test_hidden_files_never_reach_a_page() {
        // Impressions are logged from the UI's page cache, which is filled from
        // these pages. A hidden file that reached a page would be logged as
        // seen when the user never saw it, so this is the property that keeps
        // hiding out of the training data.
        let (mut state, _dir, _rx) = worker_across_hidden_boundary("/test", "hide-pages");

        state.filter_and_rank("").expect("Filter should succeed");
        let page = state.get_page(0, 128);

        let paths: Vec<PathBuf> = page.files.iter().map(|f| f.full_path.clone()).collect();
        assert_eq!(
            paths,
            vec![PathBuf::from("/test/current/notes.txt")],
            "The hidden file is absent from the page the UI logs impressions from"
        );
    }
}
