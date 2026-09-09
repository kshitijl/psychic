use crate::db::Database;
use crate::feature_defs::feature_names;
use crate::metadata_ext::MetadataExt;
use crate::ranker;
use crate::walker::start_file_walker;
use anyhow::Result;
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use mpsc::Sender;
use serde_json::json;
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
    /// Every direct child of the root has been sent.
    ///
    /// Published straight away rather than on the next debounce tick: these are
    /// what the user is looking at, they arrive within a few milliseconds, and
    /// for a directory with little under it they are the entire answer.
    ChildrenDone,
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
    /// Pick up the newly retrained model and the latest click history.
    Reload {
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
    /// (held by `file_scores` and the UI's page cache) remain valid, but
    /// `filter_and_rank` never emits them again.
    evicted: bool,
    /// Set when this path sits under a hidden prefix that does not contain the
    /// current root - see `WorkerState::recompute_hidden`. Recomputed whenever
    /// the root or the hidden set changes, so filtering stays a bool test.
    hidden: bool,
}

/// Every path the user has interacted with that is still on disk, as registry
/// entries.
///
/// One `stat` per path. It used to be three: `exists()`, then `canonicalize()`,
/// then `metadata()`. The first and third ask the same question - a `stat` that
/// succeeds *is* the existence check - and the second answers one nothing asks.
/// Every writer of the events table stores a path that is already canonical,
/// because it comes from a registry entry the walker canonicalised when it
/// found it. Checked against the real database: of the stored paths still on
/// disk, none differed from their canonical form by anything except a trailing
/// slash, and `Path` compares and hashes by component, so a trailing slash was
/// never going to produce a second registry entry anyway.
fn load_historical_files(db_path: &Path, root: &Path) -> Vec<FileInfo> {
    // Timed in here rather than around the call, which would measure the
    // ranker load running beside it as well.
    let start = std::time::Instant::now();

    let db = match Database::new(db_path) {
        Ok(db) => db,
        Err(e) => {
            log::error!("Failed to open the database for historical files: {}", e);
            return Vec::new();
        }
    };

    let paths = db.get_previously_interacted_files().unwrap_or_default();
    let candidates = paths.len();

    let files: Vec<FileInfo> = paths
        .into_iter()
        .filter_map(|path| {
            let path = PathBuf::from(path);
            // The one syscall. `Err` here means the path is gone, which is the
            // answer `exists()` used to be asked for separately.
            let metadata = std::fs::metadata(&path).ok()?;

            Some(FileInfo::from_history(
                path,
                metadata.mtime_as_secs(),
                metadata.atime_as_secs(),
                Some(metadata.len() as i64),
                metadata.is_dir(),
                root,
            ))
        })
        .collect();

    log::info!(
        "TIMING {{\"op\":\"load_historical_files\",\"ms\":{},\"count\":{},\"candidates\":{}}}",
        start.elapsed().as_secs_f64() * 1000.0,
        files.len(),
        candidates
    );

    files
}

/// How a path should read in the list, given where we are standing.
///
/// The root gets its own directory name. Stripping the root from itself leaves
/// an empty string, and the renderer decorates that into a nameless "/ (cwd)"
/// row - which is what the current directory looked like after navigating into
/// somewhere the user had visited before.
fn display_name_for(path: &Path, root: &Path) -> String {
    if path == root {
        return root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(".")
            .to_string();
    }

    match path.strip_prefix(root) {
        // Under the current tree: show where it is relative to here.
        Ok(relative) => relative.to_string_lossy().to_string(),
        // From somewhere else entirely: show the whole path. The UI colours
        // these differently to say so.
        Err(_) => path.to_string_lossy().to_string(),
    }
}

impl FileInfo {
    fn from_history(
        full_path: PathBuf,
        mtime: Option<i64>,
        atime: Option<i64>,
        file_size: Option<i64>,
        is_dir: bool,
        root: &Path,
    ) -> Self {
        let display_name = display_name_for(&full_path, root);
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

/// What the worker needs at startup, beyond where to look and where to log.
///
/// A struct rather than three loose bools at the call site, which is a
/// transposition waiting to happen.
pub struct WorkerOptions {
    /// Every directory the user has hidden.
    ///
    /// Passed in rather than read here: the caller already has a connection
    /// open on this thread, and the walker needs these before `WorkerState`
    /// exists, so reading them here meant a second connection to the same
    /// database from the same thread.
    pub hidden_prefixes: Vec<PathBuf>,
    /// Skip click history and previously interacted files.
    pub no_click_loading: bool,
    /// Skip the ranking model.
    pub no_model: bool,
    /// Whether the walker consults `.gitignore` and friends.
    pub respect_gitignore: bool,
}

pub fn spawn<T>(
    cwd: PathBuf,
    data_dir: &Path,
    event_tx: Sender<T>,
    options: WorkerOptions,
) -> Result<(Sender<WorkerRequest>, JoinHandle<()>)>
where
    T: From<WorkerResponse> + Send + 'static,
{
    let (worker_tx, worker_task_rx) = mpsc::channel::<WorkerRequest>();
    let (walker_message_tx, walker_message_rx) = mpsc::channel::<WalkerMessage>();
    let (walker_command_tx, walker_command_rx) = mpsc::channel::<WalkerCommand>();

    let data_dir = data_dir.to_path_buf();

    let hidden_prefixes = options.hidden_prefixes;
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
            options.no_click_loading,
            options.no_model,
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
        options.respect_gitignore,
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
    file_scores: Vec<ranker::FileScore>,
    current_query: String,
    current_query_id: u64,
    current_filter: FilterType,
    root: PathBuf,
    ranker: ranker::Ranker,
    model_path: PathBuf,
    /// The worker's own connection, opened once. A `Connection` is `Send` but
    /// not `Sync`, so it cannot be shared with the other threads that need one -
    /// but this thread should not keep opening a new one either.
    db: Database,
    walker_command_tx: Sender<WalkerCommand>,
    /// How long the last filter-and-rank took, reported back with its results.
    last_rank_ms: f64,
    /// Where the last query's time went, held until its results have been sent.
    last_query_timings: Option<QueryTimings>,
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

        let model_path = data_dir.join("model.txt");
        let db_path = Database::get_db_path(data_dir);

        // The two halves of start-up share nothing: one reads the model file
        // and the click history, the other reads the path list and stats each
        // path. So do them at once. The ranker stays on *this* thread because a
        // LightGBM `Booster` holds raw pointers and is not `Send`; what crosses
        // the boundary is a `Vec<FileInfo>`, which is.
        let db = Database::new(&db_path)?;

        let (ranker, historical) = std::thread::scope(|scope| {
            let loader = (!no_click_loading).then(|| {
                let db_path = db_path.clone();
                let root = root.clone();
                scope.spawn(move || load_historical_files(&db_path, &root))
            });

            let ranker_start = std::time::Instant::now();
            let ranker = if no_click_loading || _no_model {
                // Skip loading model and clicks if either flag is set
                ranker::Ranker::new_empty(&db)
            } else {
                Self::load_ranker(&model_path, &db)
            };
            log::info!(
                "TIMING {{\"op\":\"ranker_init\",\"ms\":{}}}",
                ranker_start.elapsed().as_secs_f64() * 1000.0
            );

            let historical = loader
                .map(|handle| handle.join().unwrap_or_default())
                .unwrap_or_default();

            (ranker, historical)
        });
        let ranker = ranker?;

        let mut file_registry: Vec<FileInfo> = Vec::with_capacity(historical.len() + 1);
        let mut path_to_id: HashMap<PathBuf, FileId> = HashMap::new();
        for file in historical {
            let path = file.full_path.clone();
            path_to_id.entry(path).or_insert_with(|| {
                let file_id = FileId(file_registry.len());
                file_registry.push(file);
                file_id
            });
        }

        let canonical_root = root.canonicalize().unwrap_or_else(|_| root.clone());

        log::info!(
            "TIMING {{\"op\":\"worker_state_new_total\",\"ms\":{}}}",
            worker_state_start.elapsed().as_secs_f64() * 1000.0
        );

        let mut state = WorkerState {
            file_registry,
            path_to_id,
            file_scores: Vec::new(),
            current_query: String::new(),
            current_query_id: 0,
            current_filter: FilterType::None,
            root: canonical_root,
            ranker,
            model_path,
            db,
            walker_command_tx,
            last_rank_ms: 0.0,
            last_query_timings: None,
            hidden_prefixes,
            active_hidden: Vec::new(),
        };
        // The walker never reports the directory it is walking, so this is the
        // only thing that puts the current directory in the registry.
        state.ensure_root_row();
        state.recompute_hidden();

        Ok(state)
    }

    /// Make sure the current directory has a row of its own.
    ///
    /// The walker skips the directory it is walking, so nothing else adds it.
    /// It has to run again after every `change_cwd`, not only at startup: the
    /// row `new` created is dropped along with the walked files, and if the new
    /// directory happens to be one the user has visited before - which, with
    /// the zsh hook logging every `cd`, is nearly always - it is sitting in the
    /// registry as a *historical* entry instead, showing an absolute path where
    /// it should show its own name.
    fn ensure_root_row(&mut self) {
        let root = self.root.clone();
        let display_name = display_name_for(&root, &root);

        if let Some(&file_id) = self.path_to_id.get(&root) {
            let existing = &mut self.file_registry[file_id.0];
            existing.display_name = display_name;
            // It is where we are standing now, not somewhere we once went.
            existing.origin = FileOrigin::CwdWalker;
            existing.is_dir = true;
            existing.is_under_cwd = true;
            existing.evicted = false;
            return;
        }

        let metadata = std::fs::metadata(&root).ok();
        let file_id = FileId(self.file_registry.len());
        self.file_registry.push(FileInfo {
            full_path: root.clone(),
            display_name,
            mtime: metadata.as_ref().and_then(|m| m.mtime_as_secs()),
            atime: metadata.as_ref().and_then(|m| m.atime_as_secs()),
            file_size: metadata.as_ref().map(|m| m.len() as i64),
            origin: FileOrigin::CwdWalker,
            is_dir: true,
            is_under_cwd: true,
            evicted: false,
            hidden: false,
        });
        self.path_to_id.insert(root, file_id);
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

        self.db.hide_prefix(&path)?;
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
            let display_name = display_name_for(&path, &self.root);

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
                    relative_path: &file_info.display_name,
                    full_path: &file_info.full_path,
                    mtime: file_info.mtime,
                    file_size: file_info.file_size,
                    is_from_walker: file_info.origin == FileOrigin::CwdWalker,
                    is_dir: file_info.is_dir,
                    fuzzy_score,
                }
            })
            .collect();
        let filter_ms = filter_start.elapsed().as_secs_f64() * 1000.0;
        let count = file_candidates.len();

        // Rank them with the model
        let current_timestamp = jiff::Timestamp::now().as_second();
        let rank_timings =
            match self
                .ranker
                .rank_files(query, &file_candidates, current_timestamp, &self.root)
            {
                Ok(ranking) => {
                    self.file_scores = ranking.scores;
                    Some(ranking.timings)
                }
                Err(e) => {
                    log::warn!("Ranking failed: {}, falling back to simple filtering", e);
                    // The filter's own order, with nothing scored. This used to
                    // leave `file_scores` empty beside a populated list of ids,
                    // which is the one case where the two disagreed.
                    self.file_scores = matching_files
                        .iter()
                        .map(|&(file_id, fuzzy_score)| ranker::FileScore {
                            file_id: file_id.0,
                            score: 0.0,
                            features: Vec::new(),
                            simple_score: None,
                            ml_score: None,
                            simple_weight: None,
                            ml_weight: None,
                            fuzzy_score,
                        })
                        .collect();
                    None
                }
            };

        self.last_rank_ms = filter_rank_start.elapsed().as_secs_f64() * 1000.0;
        // Logged once the results have been sent - see `log_query_timings`.
        self.last_query_timings = Some(QueryTimings {
            filter_ms,
            rank: rank_timings,
            total_ms: self.last_rank_ms,
            count,
        });
        Ok(())
    }

    /// Write the last query's timings out, as one line.
    ///
    /// This used to be ~23 lines per query - eight op lines plus one per
    /// feature - and fern flushes per record, so every one of them was a write
    /// syscall on the worker thread standing between the user's keystroke and
    /// their results. Now it is one line, after the send.
    fn log_query_timings(&mut self) {
        if let Some(timings) = self.last_query_timings.take() {
            log::info!("TIMING {}", timings.to_json());
        }
    }

    fn get_slice(&self, start: usize, count: usize) -> Vec<DisplayFileInfo> {
        // Precondition: start must be within bounds
        assert!(
            start <= self.file_scores.len(),
            "get_slice: start {} exceeds result count {}",
            start,
            self.file_scores.len()
        );

        // Each row's score sits at the row's own position. This used to search
        // `file_scores` for a matching `file_id` per row, which made drawing a
        // 128-row page O(results * 128) - the whole result set walked, 128
        // times, on every keystroke.
        self.file_scores
            .iter()
            .skip(start)
            .take(count)
            .map(|file_score| {
                let file_id = file_score.file_id;
                // Precondition: file_id must be valid index into registry
                assert!(
                    file_id < self.file_registry.len(),
                    "Invalid file_id {} (registry size: {})",
                    file_id,
                    self.file_registry.len()
                );

                let file_info = &self.file_registry[file_id];
                let score = file_score.score;
                let features = file_score.features.clone();
                let simple_score = file_score.simple_score;
                let ml_score = file_score.ml_score;
                let simple_weight = file_score.simple_weight;
                let ml_weight = file_score.ml_weight;
                let fuzzy_score = file_score.fuzzy_score;

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

    /// One page of results, in the size the UI caches by.
    ///
    /// The size is `app::PAGE_SIZE` rather than an argument: the UI keys its
    /// page cache by `index / PAGE_SIZE`, so a worker that paginated by anything
    /// else would hand back pages that land in the wrong slot. Every caller
    /// passed the same literal 128 anyway.
    fn get_page(&self, page_num: usize) -> PageData {
        let page_size = crate::app::PAGE_SIZE;

        let start_index = page_num * page_size;
        let end_index = (start_index + page_size).min(self.file_scores.len());
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
    fn load_ranker(model_path: &Path, db: &Database) -> Result<ranker::Ranker> {
        if !model_path.exists() {
            log::info!(
                "Model file not found at {:?}, using empty ranker",
                model_path
            );
            return ranker::Ranker::new_empty(db);
        }

        match ranker::Ranker::new(model_path, db) {
            Ok(ranker) => {
                log::info!("Loaded ranking model from {:?}", model_path);
                Ok(ranker)
            }
            Err(e) => {
                // Includes a model from a build with a different feature set,
                // which is what the first launch after an upgrade loads. The
                // simple model is the right answer to all of these: ranking
                // degrades to what a fresh install runs on, rather than failing
                // per query and handing back the filter's own order unscored.
                log::error!(
                    "Failed to load model at {:?} ({}); falling back to the simple \
                     model until the next retrain finishes",
                    model_path,
                    e
                );
                ranker::Ranker::new_empty(db)
            }
        }
    }

    fn reload_model(&mut self) -> Result<()> {
        log::info!("Worker: Reloading model from disk");
        self.ranker = Self::load_ranker(&self.model_path, &self.db)?;
        log::info!("Worker: Model reloaded successfully");
        Ok(())
    }

    /// Stop showing `path`, which the UI found missing from disk.
    ///
    /// The registry entry is marked rather than removed: `FileId` is an index
    /// into `file_registry`, and those indices are held by `file_scores` and
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

        // Canonical, like the root `new` starts with, so that comparisons
        // against the registry's canonical paths mean what they say.
        self.root = new_cwd.canonicalize().unwrap_or(new_cwd.clone());

        // Recalculate display names for all historical files with new root
        for file in self.file_registry.iter_mut() {
            file.display_name = display_name_for(&file.full_path, &self.root);
            file.is_under_cwd = file.full_path.starts_with(&self.root);
        }

        // Rebuild path_to_id map (only keep historical files)
        self.path_to_id.clear();
        for (idx, file) in self.file_registry.iter().enumerate() {
            self.path_to_id.insert(file.full_path.clone(), FileId(idx));
        }

        // The row for where we are now went with the walked files above.
        self.ensure_root_row();

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
/// Where one query's time went. Held on the worker until its results are sent,
/// then written out as a single `TIMING` line by `log_query_timings`.
struct QueryTimings {
    filter_ms: f64,
    /// `None` when ranking failed and the results are the filter's own order.
    rank: Option<ranker::RankTimings>,
    total_ms: f64,
    count: usize,
}

impl QueryTimings {
    /// The one line a query writes, as JSON.
    ///
    /// Kept separate from logging so the shape can be pinned by a test;
    /// `analyze_perf.rs` parses exactly this.
    fn to_json(&self) -> String {
        let per_feature: serde_json::Map<String, serde_json::Value> = match &self.rank {
            Some(rank) => feature_names()
                .iter()
                .zip(&rank.per_feature_ms)
                .map(|(name, ms)| (name.to_string(), json!(round_us(*ms))))
                .collect(),
            None => serde_json::Map::new(),
        };

        // A query whose ranking failed still reports the filter and the total;
        // the stages that never ran report nothing rather than a misleading zero.
        let stage = |pick: fn(&ranker::RankTimings) -> f64| {
            self.rank.as_ref().map(|rank| round_us(pick(rank)))
        };

        json!({
            "op": "query",
            "filter_ms": round_us(self.filter_ms),
            "simple_ms": stage(|r| r.simple_ms),
            "features_ms": stage(|r| r.features_ms),
            "predict_ms": stage(|r| r.predict_ms),
            "blend_ms": stage(|r| r.blend_ms),
            "total_ms": round_us(self.total_ms),
            "count": self.count,
            "per_feature": per_feature,
        })
        .to_string()
    }
}

/// Round a millisecond figure to microseconds.
///
/// Full f64 precision here is noise - `Instant` does not resolve it and nobody
/// reads it - and it makes the line several times longer than the numbers in it.
fn round_us(ms: f64) -> f64 {
    (ms * 1000.0).round() / 1000.0
}

/// Send a finished query's results, then log where its time went.
///
/// The log write is a syscall on the worker thread, so it goes after the
/// response rather than in front of it.
fn send_query_updated<T>(state: &mut WorkerState, event_tx: &mpsc::Sender<T>, query_id: u64)
where
    T: From<WorkerResponse> + Send,
{
    let initial_page = state.get_page(0);
    let _ = event_tx.send(
        WorkerResponse::QueryUpdated {
            query_id,
            total_results: state.file_scores.len(),
            total_files: state.file_registry.len(),
            initial_page,
            model_stats: state.ranker.stats.clone(),
            rank_ms: state.last_rank_ms,
        }
        .into(),
    );
    state.log_query_timings();
}

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
        // Set by the milestones worth showing at once, rather than whenever the
        // debounce below next comes round.
        let mut publish_now = false;
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
                WalkerMessage::ChildrenDone => {
                    files_changed = true;
                    publish_now = true;
                }
                WalkerMessage::AllDone => {
                    // Measured from process start, not from when this loop began,
                    // so it lines up with first_render / first_query_complete /
                    // startup_complete and with the debug pane.
                    let walk_ms = crate::PROCESS_START.elapsed().as_secs_f64() * 1000.0;
                    log::info!("TIMING {{\"op\":\"walker_complete\",\"ms\":{}}}", walk_ms);
                    files_changed = true;
                    publish_now = true;
                    // Notify UI that walker is done
                    let _ = event_tx.send(WorkerResponse::WalkerDone { walk_ms }.into());
                }
            }
        }

        // If files changed, notify the UI so it can decide to trigger a refresh.
        // Debounced to avoid spamming the UI thread, unless this is one of the
        // milestones that should reach the screen the moment it happens.
        if files_changed
            && (publish_now
                || last_files_changed_notification.elapsed() > Duration::from_millis(200))
        {
            let _ = event_tx.send(WorkerResponse::FilesChanged.into());
            last_files_changed_notification = Instant::now();
        }

        // Wait for worker requests with timeout
        let pending = match task_rx.recv_timeout(Duration::from_millis(5)) {
            Ok(first) => drain_requests(&task_rx, first),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                log::debug!("Worker thread channel disconnected");
                break;
            }
        };

        for request in pending {
            match request {
                WorkerRequest::UpdateQuery(latest_req) => {
                    state.current_query = latest_req.query.clone();
                    state.current_query_id = latest_req.query_id;
                    state.current_filter = latest_req.filter;

                    // Filter and rank
                    if let Err(e) = state.filter_and_rank(&latest_req.query) {
                        log::error!("Filter/rank failed: {}", e);
                        continue;
                    }

                    send_query_updated(&mut state, &event_tx, latest_req.query_id);
                }
                WorkerRequest::GetPage { query_id, page_num } => {
                    // If the request is for an old query, ignore it.
                    if query_id != state.current_query_id {
                        continue;
                    }
                    let page_data = state.get_page(page_num);
                    let _ = event_tx.send(
                        WorkerResponse::Page {
                            query_id,
                            page_data,
                        }
                        .into(),
                    );
                }
                WorkerRequest::Reload { query_id } => {
                    state.current_query_id = query_id;
                    // `reload_model` reloads the clicks too: it goes through
                    // `load_ranker`, and both `Ranker::new` and
                    // `Ranker::new_empty` call `Ranker::load_clicks`.
                    if let Err(e) = state.reload_model() {
                        log::error!("Failed to reload model: {}", e);
                    } else {
                        // Re-filter and rank with new model
                        let query = state.current_query.clone();
                        if let Err(e) = state.filter_and_rank(&query) {
                            log::error!("Filter/rank failed after model reload: {}", e);
                        } else {
                            send_query_updated(&mut state, &event_tx, query_id);
                        }
                    }
                }
                WorkerRequest::ChangeCwd { new_cwd, query_id } => {
                    state.current_query_id = query_id;
                    // Entering a directory refilters from scratch and redraws
                    // the whole list, so it is a safe moment to pick up a
                    // newly retrained model: nothing reorders unprompted.
                    if let Err(e) = state.reload_model() {
                        log::error!("Failed to reload model on cwd change: {}", e);
                    }
                    if let Err(e) = state.change_cwd(new_cwd) {
                        log::error!("Failed to change cwd: {}", e);
                    } else {
                        // Clear query and re-filter (will show only historical files until walker sends new ones)
                        state.current_query = String::new();
                        if let Err(e) = state.filter_and_rank("") {
                            log::error!("Filter/rank failed after cwd change: {}", e);
                        } else {
                            send_query_updated(&mut state, &event_tx, query_id);
                        }
                    }
                }
                WorkerRequest::Evict { path, query_id } => {
                    state.current_query_id = query_id;
                    // Re-run the current query so the missing row disappears
                    // immediately. Nothing to do if the path was not registered.
                    if state.evict(&path) {
                        let query = state.current_query.clone();
                        if let Err(e) = state.filter_and_rank(&query) {
                            log::error!("Filter/rank failed after eviction: {}", e);
                        } else {
                            send_query_updated(&mut state, &event_tx, query_id);
                        }
                    }
                }
                WorkerRequest::Hide { path, query_id } => {
                    state.current_query_id = query_id;
                    match state.hide(path) {
                        Ok(false) => {}
                        Ok(true) => {
                            // Re-run the current query so the rows go at once.
                            let query = state.current_query.clone();
                            if let Err(e) = state.filter_and_rank(&query) {
                                log::error!("Filter/rank failed after hiding: {}", e);
                            } else {
                                send_query_updated(&mut state, &event_tx, query_id);
                            }
                        }
                        Err(e) => log::error!("Failed to hide directory: {}", e),
                    }
                }
            }
        }
    }
}

/// Take everything already waiting, collapsing runs of query updates.
///
/// Only *consecutive* `UpdateQuery`s are collapsed. A query the user has
/// already typed past need not be run, but every other request is a distinct
/// instruction and has to survive: this used to keep the latest `UpdateQuery`
/// and throw away whatever it found in between, so typing and then pressing
/// Enter while the worker was busy ate the `ChangeCwd`. The UI moved to the new
/// directory and the worker went on serving the old one.
fn drain_requests(rx: &mpsc::Receiver<WorkerRequest>, first: WorkerRequest) -> Vec<WorkerRequest> {
    let mut queue = vec![first];

    while let Ok(request) = rx.try_recv() {
        let supersedes_the_last = matches!(
            (queue.last(), &request),
            (
                Some(WorkerRequest::UpdateQuery(_)),
                WorkerRequest::UpdateQuery(_)
            )
        );
        if supersedes_the_last {
            queue.pop();
        }
        queue.push(request);
    }

    queue
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn update(query: &str) -> WorkerRequest {
        WorkerRequest::UpdateQuery(UpdateQueryRequest {
            query: query.to_string(),
            query_id: 0,
            filter: FilterType::None,
        })
    }

    #[test]
    fn test_typing_does_not_swallow_what_follows_it() {
        // The user types, then hits Enter on a directory, then types again,
        // all while the worker is busy. Every keystroke but the last is dead,
        // but the navigation in the middle is not: this used to keep only the
        // newest query and drop whatever it found on the way, so the UI moved
        // to the new directory and the worker went on serving the old one.
        let (tx, rx) = mpsc::channel::<WorkerRequest>();
        tx.send(update("a")).unwrap();
        tx.send(update("ab")).unwrap();
        tx.send(WorkerRequest::ChangeCwd {
            new_cwd: PathBuf::from("/elsewhere"),
            query_id: 7,
        })
        .unwrap();
        tx.send(update("z")).unwrap();

        let first = rx.recv().unwrap();
        let queue = drain_requests(&rx, first);

        let described: Vec<String> = queue
            .iter()
            .map(|request| match request {
                WorkerRequest::UpdateQuery(r) => format!("query {:?}", r.query),
                WorkerRequest::ChangeCwd { new_cwd, .. } => format!("cd {}", new_cwd.display()),
                _ => "other".to_string(),
            })
            .collect();

        assert_eq!(
            described,
            vec!["query \"ab\"", "cd /elsewhere", "query \"z\""],
            "the superseded query goes, everything else stays, in order"
        );
    }

    #[test]
    fn test_a_lone_request_is_returned_as_is() {
        let (_tx, rx) = mpsc::channel::<WorkerRequest>();

        let queue = drain_requests(&rx, update("hello"));

        assert_eq!(queue.len(), 1);
    }

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
        assert!(file_dir.is_dir, "Directory should pass OnlyDirs filter");
        assert!(
            !file_regular.is_dir,
            "Regular file should NOT pass OnlyDirs filter"
        );

        // OnlyFiles filter: !is_dir (is_dir == false)
        assert!(
            !(!file_dir.is_dir),
            "Directory should NOT pass OnlyFiles filter"
        );
        assert!(
            !file_regular.is_dir,
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

        let db = crate::db::Database::new(&dir.db_path()).expect("open test db");
        let ranker = WorkerState::load_ranker(&dir.model_path(), &db)
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
        std::fs::write(dir.model_path(), "").expect("Failed to write empty model");

        let db = crate::db::Database::new(&dir.db_path()).expect("open test db");
        let ranker = WorkerState::load_ranker(&dir.model_path(), &db)
            .expect("An empty model file must not stop psychic from starting");

        assert!(!ranker.has_model(), "The unusable model was not loaded");
    }

    #[test]
    fn test_corrupt_model_falls_back_instead_of_failing_startup() {
        let dir = TempDataDir::new("corrupt-model");
        std::fs::write(dir.model_path(), "this is not a LightGBM model")
            .expect("Failed to write corrupt model");

        let db = crate::db::Database::new(&dir.db_path()).expect("open test db");
        let ranker = WorkerState::load_ranker(&dir.model_path(), &db)
            .expect("A corrupt model must not stop psychic from starting");

        assert!(!ranker.has_model(), "The unusable model was not loaded");
    }
}

#[cfg(test)]
pub(super) mod fresh_install_tests_support {
    use std::path::PathBuf;

    /// An empty directory, as a new user has.
    pub(crate) struct FreshDataDir {
        pub(crate) path: PathBuf,
    }

    impl FreshDataDir {
        pub(crate) fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("psychic-fresh-{}-{}", name, std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create data dir");
            Self { path }
        }

        pub(crate) fn db_path(&self) -> PathBuf {
            crate::db::Database::get_db_path(&self.path)
        }

        pub(crate) fn model_path(&self) -> PathBuf {
            self.path.join("model.txt")
        }
    }

    impl Drop for FreshDataDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod page_tests {
    //! A page of results is a slice of `file_scores`, by position.
    //!
    //! It used to search `file_scores` for a row whose `file_id` matched, once
    //! per row, against a parallel `Vec<FileId>` that held the same order. These
    //! pin that the two really were the same order, so indexing is safe.

    use super::eviction_tests::worker_with_three_files;

    #[test]
    fn test_each_row_carries_its_own_score() {
        let (mut state, _dir, _rx) = worker_with_three_files("page-scores");
        state.filter_and_rank("").expect("ranking");

        let page = state.get_page(0);
        assert_eq!(page.files.len(), state.file_scores.len());

        for (row, scored) in page.files.iter().zip(&state.file_scores) {
            assert_eq!(
                row.full_path, state.file_registry[scored.file_id].full_path,
                "row {} is not the file its score belongs to",
                row.display_name
            );
            assert_eq!(row.score, scored.score, "and carries that file's score");
            assert_eq!(row.fuzzy_score, scored.fuzzy_score);
        }
    }

    #[test]
    fn test_the_page_is_in_ranked_order() {
        let (mut state, _dir, _rx) = worker_with_three_files("page-order");
        state.filter_and_rank("").expect("ranking");

        let scores: Vec<f64> = state.get_page(0).files.iter().map(|f| f.score).collect();
        let mut descending = scores.clone();
        descending.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(scores, descending, "best first, as ranked");
    }

    #[test]
    fn test_the_last_page_stops_at_the_last_result() {
        let (mut state, _dir, _rx) = worker_with_three_files("page-short");
        state.filter_and_rank("").expect("ranking");

        // Three results in a 128-row page: the page ends where they do.
        let page = state.get_page(0);
        assert_eq!(page.start_index, 0);
        assert_eq!(page.end_index, 3);
        assert_eq!(page.files.len(), 3);
    }

    #[test]
    #[should_panic(expected = "exceeds result count")]
    fn test_asking_for_a_page_past_the_end_is_a_bug_and_says_so() {
        // The UI only asks for pages it has been told exist, so this is a
        // precondition rather than a case to handle: a page number out of range
        // means the UI and the worker disagree about how many results there
        // are, and quietly returning nothing would hide that.
        let (mut state, _dir, _rx) = worker_with_three_files("page-past-end");
        state.filter_and_rank("").expect("ranking");

        state.get_page(9);
    }
}

#[cfg(test)]
mod query_timing_tests {
    //! One line per query, in a shape `analyze_perf.rs` can read back.

    use super::*;
    use crate::feature_defs::FEATURE_REGISTRY;

    fn timings() -> QueryTimings {
        QueryTimings {
            filter_ms: 1.5,
            rank: Some(ranker::RankTimings {
                simple_ms: 0.25,
                features_ms: 0.75,
                predict_ms: 0.5,
                blend_ms: 0.125,
                // Registry order: give the sixth feature all the time, so the
                // line shows which feature each number belongs to.
                per_feature_ms: (0..FEATURE_REGISTRY.len())
                    .map(|i| if i == 5 { 0.5 } else { 0.0 })
                    .collect(),
            }),
            total_ms: 3.25,
            count: 171,
        }
    }

    #[test]
    fn test_a_query_writes_one_line_with_its_whole_breakdown() {
        assert_eq!(
            timings().to_json(),
            r#"{"blend_ms":0.125,"count":171,"features_ms":0.75,"filter_ms":1.5,"op":"query","per_feature":{"clicks_for_this_query":0.0,"clicks_last_24h":0.0,"clicks_last_30_days":0.0,"clicks_last_7_days":0.0,"clicks_last_hour":0.0,"clicks_last_week_parent_dir":0.0,"engagements_in_episode_with_query":0.0,"filename_starts_with_query":0.0,"fuzzy_score":0.0,"is_dir":0.0,"is_hidden":0.0,"is_under_cwd":0.0,"log_file_size":0.5,"modified_age":0.0,"modified_last_24h":0.0},"predict_ms":0.5,"simple_ms":0.25,"total_ms":3.25}"#
        );
    }

    #[test]
    fn test_a_failed_ranking_reports_the_filter_and_nothing_it_did_not_do() {
        let mut timings = timings();
        timings.rank = None;

        assert_eq!(
            timings.to_json(),
            r#"{"blend_ms":null,"count":171,"features_ms":null,"filter_ms":1.5,"op":"query","per_feature":{},"predict_ms":null,"simple_ms":null,"total_ms":3.25}"#
        );
    }

    #[test]
    fn test_the_numbers_are_rounded_to_microseconds() {
        // Instant does not resolve past this and nobody reads it; full f64
        // precision made the line several times longer than the numbers in it.
        let mut timings = timings();
        timings.filter_ms = 1.234_567_891_23;
        timings.total_ms = 0.000_499;

        let json = timings.to_json();
        assert!(
            json.contains(r#""filter_ms":1.235"#),
            "filter_ms should round to microseconds: {}",
            json
        );
        assert!(
            json.contains(r#""total_ms":0.0"#),
            "half a nanosecond rounds away: {}",
            json
        );
    }
}

#[cfg(test)]
mod reload_tests {
    //! `WorkerRequest::Reload` is one request because one reload does both
    //! jobs: `reload_model` goes through `load_ranker`, and `Ranker::new` and
    //! `Ranker::new_empty` both read the click history. A separate
    //! `ReloadClicks` request re-read the same rows and reranked a second
    //! time for nothing.

    use super::fresh_install_tests_support::*;
    use super::*;
    use crate::db::Database;

    fn worker_state(dir: &FreshDataDir) -> WorkerState {
        let (walker_tx, _walker_rx) = mpsc::channel::<WalkerCommand>();
        WorkerState::new(
            PathBuf::from("/test"),
            &dir.path,
            walker_tx,
            false,
            false,
            Vec::new(),
        )
        .expect("worker must start")
    }

    #[test]
    fn test_reloading_the_model_also_picks_up_clicks_logged_since_startup() {
        let dir = FreshDataDir::new("reload-clicks");
        let mut state = worker_state(&dir);

        assert!(
            state.ranker.clicks.clicks_by_file.is_empty(),
            "nothing has been clicked before the worker starts"
        );

        // What the UI writes when the user opens a file, after the worker has
        // already loaded its click history.
        let db = Database::new(&dir.db_path()).expect("open the same database");
        db.log_event(crate::db::EventData {
            query: "al",
            file_path: "alpha.rs",
            full_path: "/test/alpha.rs",
            mtime: Some(1_700_000_000),
            atime: None,
            file_size: Some(100),
            subsession_id: 1,
            action: crate::db::UserInteraction::Click,
            session_id: "session-1",
            episode_queries: None,
        })
        .expect("log the click");

        state
            .reload_model()
            .expect("the reload a file click or a cwd change triggers");

        assert_eq!(
            state
                .ranker
                .clicks
                .clicks_by_file
                .keys()
                .collect::<Vec<_>>(),
            vec!["/test/alpha.rs"],
            "reloading the model reloads the clicks with it, so no second \
             request is needed"
        );
    }
}

#[cfg(test)]
mod fresh_install_tests {
    //! What happens the first time psychic is run on a machine.
    //!
    //! The pieces of this were each covered and the whole was not, which is how
    //! a fresh install came to spend a long time logging `Background retraining
    //! failed: no such table: events` on every launch without anyone noticing.

    use super::fresh_install_tests_support::*;
    use super::*;
    use crate::db::Database;

    fn names(db: &Database, kind: &str) -> Vec<String> {
        db.connection()
            .prepare("SELECT name FROM sqlite_master WHERE type = ?1 AND name NOT LIKE 'sqlite_%'")
            .unwrap()
            .query_map([kind], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    }

    #[test]
    fn test_an_empty_data_directory_gets_the_whole_schema() {
        let dir = FreshDataDir::new("schema");

        let db = Database::new(&dir.db_path()).expect("a new user's first open");

        let mut tables = names(&db, "table");
        tables.sort();
        assert_eq!(tables, vec!["events", "hidden_prefixes", "sessions"]);

        let mut indexes = names(&db, "index");
        indexes.sort();
        assert_eq!(
            indexes,
            vec!["idx_events_action", "idx_events_engagement"],
            "a new database is created with the indexes, not left to a migration"
        );
    }

    #[test]
    fn test_the_first_launch_ranks_on_the_simple_model() {
        let dir = FreshDataDir::new("cold-start");
        let (walker_tx, _walker_rx) = mpsc::channel::<WalkerCommand>();

        // No model file, no click history, no events: exactly what
        // `WorkerState` is handed the first time psychic is run.
        let mut state = WorkerState::new(
            PathBuf::from("/test"),
            &dir.path,
            walker_tx,
            false,
            false,
            Vec::new(),
        )
        .expect("a fresh install must start");

        assert!(
            !state.ranker.has_model(),
            "there is no model to load on a first launch"
        );

        for name in ["alpha.rs", "beta.rs"] {
            state.add_file(
                PathBuf::from("/test").join(name),
                Some(1_700_000_000),
                Some(1_700_000_000),
                Some(100),
                false,
            );
        }

        state.filter_and_rank("").expect("first query");

        assert_eq!(
            state.file_scores.len(),
            3,
            "two files and the directory the user is standing in"
        );
        let scored = state.file_scores.first().expect("something was scored");
        assert!(scored.simple_score.is_some(), "the simple model ran");
        assert_eq!(scored.ml_score, None, "and nothing else did");
        assert_eq!(scored.simple_weight, Some(1.0));
        assert_eq!(scored.ml_weight, Some(0.0));
    }

    #[test]
    fn test_clicks_reach_the_database_and_come_back_as_training_rows() {
        let dir = FreshDataDir::new("clicks");
        let db = Database::new(&dir.db_path()).unwrap();

        // What the UI writes: the rows it showed, then the one that was taken.
        let seen = [
            crate::db::FileMetadata {
                relative_path: "alpha.rs".to_string(),
                full_path: "/test/alpha.rs".to_string(),
                mtime: Some(1_700_000_000),
                atime: None,
                size: Some(100),
            },
            crate::db::FileMetadata {
                relative_path: "beta.rs".to_string(),
                full_path: "/test/beta.rs".to_string(),
                mtime: Some(1_700_000_000),
                atime: None,
                size: Some(100),
            },
        ];
        db.log_impressions("al", &seen, 1, "session-1").unwrap();
        db.log_event(crate::db::EventData {
            query: "al",
            file_path: "alpha.rs",
            full_path: "/test/alpha.rs",
            mtime: Some(1_700_000_000),
            atime: None,
            file_size: Some(100),
            subsession_id: 1,
            action: crate::db::UserInteraction::Click,
            session_id: "session-1",
            episode_queries: None,
        })
        .unwrap();

        assert_eq!(
            db.get_previously_interacted_files().unwrap(),
            vec!["/test/alpha.rs".to_string()],
            "the click is what makes a path findable next time"
        );
        assert_eq!(
            db.engagements_since(0).unwrap().len(),
            1,
            "and what the ranker loads"
        );

        let summary = crate::features::generate_features(
            &dir.db_path(),
            &dir.path.join("features.csv"),
            &dir.path.join("feature_schema.json"),
            crate::features::OutputFormat::Csv,
        )
        .expect("feature generation on a nearly empty database");

        assert_eq!(summary.rows, 2, "one row per impression");
        assert_eq!(summary.positives, 1, "the one that was clicked");
    }

    #[test]
    fn test_training_is_skipped_while_nothing_has_been_clicked() {
        let dir = FreshDataDir::new("no-positives");
        let db = Database::new(&dir.db_path()).unwrap();

        // Impressions but no click: the state every install starts in, and for
        // as long as the user is only looking.
        db.log_impressions(
            "a",
            &[crate::db::FileMetadata {
                relative_path: "alpha.rs".to_string(),
                full_path: "/test/alpha.rs".to_string(),
                mtime: None,
                atime: None,
                size: None,
            }],
            1,
            "session-1",
        )
        .unwrap();

        let summary = crate::features::generate_features(
            &dir.db_path(),
            &dir.path.join("features.csv"),
            &dir.path.join("feature_schema.json"),
            crate::features::OutputFormat::Csv,
        )
        .unwrap();

        assert_eq!(summary.positives, 0);

        // `retrain_model` returns Ok without running train.py at all. Handing
        // it to Python instead produced a traceback and an ERROR in the log on
        // every first launch, for a state that is entirely normal.
        crate::ranker::retrain_model(&dir.path, Some(dir.path.join("training.log")))
            .expect("a fresh install must not report a failure");

        assert!(
            !dir.model_path().exists(),
            "and no model is written from nothing"
        );
    }
}

#[cfg(test)]
mod trained_model_tests {
    //! The rest of the fresh-install story: use psychic enough that there is
    //! something to learn from, and check the model that comes out is actually
    //! the one used.
    //!
    //! This runs `train.py` through `uv`, so it takes about seven seconds and
    //! wants a network the first time, when uv has to populate its cache. It
    //! runs by default anyway. `uv` is already required to build and use
    //! psychic at all (see the README), and a test that has to be asked for is
    //! a test nobody runs: this suite already carries two of those, which look
    //! like coverage and are not.

    use super::fresh_install_tests_support::*;
    use super::*;
    use crate::db::Database;

    /// Enough sessions that every episode has something to learn from.
    ///
    /// Also more than the blend's crossover at 30 engagements, so that the
    /// trained model is expected to outweigh the simple one afterwards.
    const SESSIONS: usize = 40;

    #[test]
    fn test_a_used_install_trains_and_then_ranks_with_the_model() {
        let dir = FreshDataDir::new("trained");
        let db = Database::new(&dir.db_path()).unwrap();

        // Use it: each query shows three files and the user takes one.
        for i in 0..SESSIONS {
            let chosen = format!("/test/file{}.rs", i % 7);
            let shown: Vec<crate::db::FileMetadata> = (0..3)
                .map(|n| crate::db::FileMetadata {
                    relative_path: format!("file{}.rs", (i + n) % 7),
                    full_path: format!("/test/file{}.rs", (i + n) % 7),
                    mtime: Some(1_700_000_000),
                    atime: None,
                    size: Some(100 + n as i64),
                })
                .collect();
            db.log_impressions("fi", &shown, i as u64, "session-1")
                .unwrap();
            db.log_event(crate::db::EventData {
                query: "fi",
                file_path: &chosen,
                full_path: &chosen,
                mtime: Some(1_700_000_000),
                atime: None,
                file_size: Some(100),
                subsession_id: i as u64,
                action: crate::db::UserInteraction::Click,
                session_id: "session-1",
                episode_queries: None,
            })
            .unwrap();
        }

        crate::ranker::retrain_model(&dir.path, Some(dir.path.join("training.log")))
            .expect("training should succeed once there is something to learn from");

        assert!(
            dir.model_path().exists(),
            "a model file is what the next launch loads: {}",
            std::fs::read_to_string(dir.path.join("training.log")).unwrap_or_default()
        );

        // Now the next launch: the model is picked up, and used.
        let ranker = WorkerState::load_ranker(&dir.model_path(), &db)
            .expect("the model psychic just wrote must load");
        assert!(
            ranker.has_model(),
            "and be used rather than fallen back from"
        );

        let mut ranker = ranker;
        // Candidates borrow from the registry in production; here they borrow
        // from names that outlive the ranking call.
        let names: Vec<(String, PathBuf)> = (0..3)
            .map(|n| {
                (
                    format!("file{}.rs", n),
                    PathBuf::from(format!("/test/file{}.rs", n)),
                )
            })
            .collect();
        let candidates: Vec<ranker::FileCandidate> = names
            .iter()
            .enumerate()
            .map(|(n, (relative_path, full_path))| ranker::FileCandidate {
                file_id: n,
                relative_path,
                full_path,
                mtime: Some(1_700_000_000),
                file_size: Some(100),
                is_from_walker: true,
                is_dir: false,
                fuzzy_score: 50,
            })
            .collect();

        let scored = ranker
            .rank_files(
                "fi",
                &candidates,
                jiff::Timestamp::now().as_second(),
                Path::new("/test"),
            )
            .expect("ranking with a model")
            .scores;

        let top = scored.first().expect("something was ranked");
        let simple = top
            .simple_weight
            .expect("the simple model still has a weight");
        let ml = top.ml_weight.expect("and so does the model");

        assert!(top.ml_score.is_some(), "the trained model produced a score");
        assert!(
            ml > simple,
            "past the crossover the model should lead: ml {:.3} vs simple {:.3}",
            ml,
            simple
        );
        assert!(
            simple > 0.0,
            "but the simple model still contributes: {:.3}",
            simple
        );
        assert!(
            (simple + ml - 1.0).abs() < 1e-9,
            "the two weights are a blend"
        );
    }
}

#[cfg(test)]
mod cwd_row_tests {
    use super::eviction_tests::worker_with_three_files;
    use super::*;

    /// The row for the directory we are standing in, if there is one.
    fn cwd_row(state: &WorkerState) -> Option<&FileInfo> {
        let root = state.root.clone();
        state.file_registry.iter().find(|f| f.full_path == root)
    }

    #[test]
    fn test_the_current_directory_has_a_row_at_startup() {
        let (state, _dir, _rx) = worker_with_three_files("cwd-row-start");

        // `worker_with_three_files` clears the registry after `new`, so rebuild
        // the row the way `new` would to check the name it produces.
        let mut state = state;
        state.ensure_root_row();

        assert_eq!(
            cwd_row(&state).map(|f| f.display_name.as_str()),
            Some("test"),
            "the current directory is shown by its own name"
        );
    }

    #[test]
    fn test_navigating_somewhere_new_still_leaves_a_row() {
        let (mut state, _dir, _rx) = worker_with_three_files("cwd-row-fresh");

        state
            .change_cwd(PathBuf::from("/test/somewhere"))
            .expect("change_cwd");

        let row = cwd_row(&state).expect("the directory we are in needs a row");
        assert_eq!(row.display_name, "somewhere");
        assert!(row.is_dir);
        assert!(row.is_under_cwd);
    }

    #[test]
    fn test_navigating_somewhere_already_visited_does_not_leave_it_nameless() {
        let (mut state, _dir, _rx) = worker_with_three_files("cwd-row-historical");

        // With the zsh hook logging every `cd`, the directory you navigate into
        // is nearly always in the registry already, as history.
        let visited = PathBuf::from("/test/visited");
        let file_id = FileId(state.file_registry.len());
        state.file_registry.push(FileInfo::from_history(
            visited.clone(),
            None,
            None,
            None,
            true,
            &PathBuf::from("/test"),
        ));
        state.path_to_id.insert(visited.clone(), file_id);

        state.change_cwd(visited).expect("change_cwd");

        let row = cwd_row(&state).expect("the directory we are in needs a row");
        assert_eq!(
            row.display_name, "visited",
            "stripping the root from itself gave an empty name, which the \
             renderer drew as a nameless \"/ (cwd)\" row"
        );
        assert_eq!(
            row.origin,
            FileOrigin::CwdWalker,
            "and it is where we are standing now, not somewhere we once went"
        );
    }

    #[test]
    fn test_the_row_is_not_duplicated_by_navigating_back_and_forth() {
        let (mut state, _dir, _rx) = worker_with_three_files("cwd-row-twice");

        for dir in ["/test/a", "/test/b", "/test/a"] {
            state.change_cwd(PathBuf::from(dir)).expect("change_cwd");
        }

        let rows = state
            .file_registry
            .iter()
            .filter(|f| f.full_path == Path::new("/test/a"))
            .count();
        assert_eq!(
            rows, 1,
            "one row per directory, however often it is visited"
        );
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
            .file_scores
            .iter()
            .map(|fs| state.file_registry[fs.file_id].display_name.clone())
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
    fn test_hiding_a_directory_hides_what_is_already_below_it() {
        // Hiding a/b while a/b/c is on screen has to take a/b/c with it, not
        // just the row that was selected.
        let (mut state, _dir, _rx) = worker_with_three_files("hide-descendants");

        state.add_file(PathBuf::from("/test/b"), Some(1000), Some(1000), None, true);
        for path in ["/test/b/c", "/test/b/c/deep.txt"] {
            state.add_file(PathBuf::from(path), Some(1000), Some(1000), Some(10), false);
        }

        state.filter_and_rank("").expect("Filter should succeed");
        assert_eq!(
            results(&state).len(),
            6,
            "Three files, the directory, and two things under it"
        );

        state
            .hide(PathBuf::from("/test/b"))
            .expect("Hiding should persist");

        state.filter_and_rank("").expect("Filter should succeed");
        let mut after = results(&state);
        after.sort();
        assert_eq!(
            after,
            vec!["alpha.txt", "beta.txt", "gamma.txt"],
            "The hidden directory and everything under it are gone together"
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
        let page = state.get_page(0);

        let paths: Vec<PathBuf> = page.files.iter().map(|f| f.full_path.clone()).collect();
        assert_eq!(
            paths,
            vec![PathBuf::from("/test/current/notes.txt")],
            "The hidden file is absent from the page the UI logs impressions from"
        );
    }
}
