use anyhow::{Context, Result};
use std::{
    collections::{HashMap, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool, mpsc},
    thread::JoinHandle,
    time::Instant,
};

use crate::analytics::Analytics;
use crate::cli::{OnCwdVisitAction, OnDirClickAction};
use crate::db::{EventData, FileMetadata};
use crate::preview::{PreviewRequest, PreviewState};
use crate::search_worker::{self, DisplayFileInfo, WorkerRequest};
use crate::{history, ranker, ui_state};

/// A page of DisplayFileInfo for caching
#[derive(Debug, Clone)]
pub struct Page {
    pub start_index: usize,
    pub end_index: usize,
    pub files: Vec<DisplayFileInfo>,
}

// Page-based caching constants
pub const PAGE_SIZE: usize = 128;
pub const PREFETCH_MARGIN: usize = 32;

/// The handful of latencies worth watching, in milliseconds.
///
/// Startup numbers are measured once from process start; the search numbers are
/// replaced on every query, so the pane shows how fast *this* search was rather
/// than a historical average. `None` means it has not happened yet.
#[derive(Debug, Clone, Default)]
pub struct Timings {
    /// Process start to the first frame drawn.
    pub first_paint_ms: Option<f64>,
    /// Process start to the first search results being ready to show.
    pub first_results_ms: Option<f64>,
    /// Process start to the filesystem walk finishing.
    pub walk_complete_ms: Option<f64>,
    /// Round trip for the most recent query: keystroke to results in hand.
    pub last_search_ms: Option<f64>,
    /// The worker's share of that: filtering and ranking, without the channel hop.
    pub last_rank_ms: Option<f64>,
}

pub struct AppOptions {
    pub on_dir_click: OnDirClickAction,
    pub on_cwd_visit: OnCwdVisitAction,
    pub initial_filter: search_worker::FilterType,
    pub no_preview: bool,
    pub no_click_loading: bool,
    pub no_model: bool,
    pub no_click_logging: bool,
    /// False when `--no-ignore` was given: show what git would hide.
    pub respect_gitignore: bool,
    pub editor: String,
}

pub struct AppBootstrap {
    pub session_id: String,
    pub event_tx: mpsc::Sender<crate::AppEvent>,
    pub input: crate::tty_input::TtyInput,
    pub preview_tx: mpsc::Sender<PreviewRequest>,
}

pub struct App {
    pub query: String,
    pub page_cache: HashMap<usize, Page>, // Page-based cache
    pub total_results: usize,             // Total number of filtered files
    pub total_files: usize,               // Total number of files in index
    pub selected_index: usize,
    pub file_list_scroll: usize, // Scroll offset for file list
    pub preview: PreviewState,
    pub cwd: PathBuf, // Current working directory
    pub history: history::History,
    pub history_selected: usize, // Selected item in history mode UI

    // For marquee path bar
    pub path_bar_scroll: u16,
    pub path_bar_scroll_direction: i8,
    pub last_path_bar_update: Instant,
    /// Width of the path bar as the last frame laid it out. Only the renderer
    /// knows it, and the marquee cannot advance without it.
    pub path_bar_width: u16,
    /// Rows of file list the last frame drew. Like `path_bar_width`, only the
    /// renderer knows it - and impressions are the rows that were on screen,
    /// which cannot be known without it.
    pub visible_list_height: u16,

    // For debug pane
    pub model_stats_cache: Option<ranker::ModelStats>, // Cached from worker, refreshed periodically
    pub currently_retraining: bool,
    /// The last few log lines, for the debug pane. Filled by the event loop
    /// from a receiver it owns: see the note on shutdown in `main`, and note
    /// that `App` deliberately does *not* hold the receiving end of the
    /// logging channel. Threads log while they wind down, and `App` is dropped
    /// during shutdown, so owning the sink here would kill it too early.
    pub recent_logs: VecDeque<String>,

    // Filter state
    pub current_filter: search_worker::FilterType,

    /// One line of feedback shown in the search bar, e.g. when a selected file
    /// turned out to be gone. Cleared on the next keypress, so it is visible
    /// exactly until the user does something else.
    pub status_message: Option<String>,

    // UI state machine
    pub ui_state: ui_state::UiState,

    // Configuration options
    pub options: AppOptions,

    // Analytics tracking (subsessions, impressions, scrolls)
    pub analytics: Analytics,

    // Search worker thread communication
    pub worker_tx: mpsc::Sender<WorkerRequest>,
    pub worker_handle: Option<JoinHandle<()>>,

    /// The input thread, which has to be stopped while a child process owns
    /// the terminal.
    pub input: crate::tty_input::TtyInput,

    // Tick thread control (for pausing when launching child processes)
    pub tick_paused: Arc<AtomicBool>,
    /// Whether anything on screen is animating - today, whether the path bar
    /// has more path than room. The tick thread reads it and stays quiet when
    /// there is nothing to drive, so an idle psychic stops redrawing itself
    /// five times a second.
    pub something_animates: Arc<AtomicBool>,

    // Startup tracking
    pub walker_done: bool,
    pub startup_complete_logged: bool,

    // Debug pane: latencies, and database stats loaded on demand
    pub timings: Timings,
    pub db_stats: Option<crate::db::DbStats>,
    /// Set once the background load is under way, so it is not started twice.
    db_stats_requested: bool,
    data_dir: PathBuf,
    event_tx: mpsc::Sender<crate::AppEvent>,
    /// When the query we are waiting on was sent, to time the round trip.
    query_sent_at: Option<(u64, Instant)>,
}

impl App {
    pub fn new(
        root: PathBuf,
        data_dir: &Path,
        bootstrap: AppBootstrap,
        options: AppOptions,
    ) -> Result<Self> {
        let start_time = Instant::now();
        log::debug!("App::new() started");

        let AppBootstrap {
            session_id,
            event_tx,
            input,
            preview_tx,
        } = bootstrap;
        let initial_filter = options.initial_filter;

        let db_path = crate::db::Database::get_db_path(data_dir);
        let db = crate::db::Database::new(&db_path)?;

        // Read while we have it open, before it moves into `Analytics`. The
        // worker needs these to start its walker, and opening a second
        // connection on this thread to fetch them would be silly.
        let hidden_prefixes = db.get_hidden_prefixes().unwrap_or_else(|e| {
            log::error!("Failed to load hidden directories: {}", e);
            Vec::new()
        });

        // Create analytics tracker
        let analytics = Analytics::new(session_id, db, options.no_click_logging);

        let (worker_tx, worker_handle) = search_worker::spawn(
            root.clone(),
            data_dir,
            event_tx.clone(),
            search_worker::WorkerOptions {
                hidden_prefixes,
                no_click_loading: options.no_click_loading,
                no_model: options.no_model,
                respect_gitignore: options.respect_gitignore,
            },
        )?;

        log::debug!("App::new() total time: {:?}", start_time.elapsed());

        let app = App {
            query: String::new(),
            page_cache: HashMap::new(),
            total_results: 0,
            total_files: 0,
            selected_index: 0,
            file_list_scroll: 0,
            preview: PreviewState::new(preview_tx),
            cwd: root.clone(),
            history: history::History::new(root),
            history_selected: 0,
            path_bar_scroll: 0,
            path_bar_scroll_direction: 1,
            path_bar_width: 0,
            visible_list_height: 0,
            last_path_bar_update: Instant::now(),
            model_stats_cache: None,
            currently_retraining: false,
            recent_logs: VecDeque::with_capacity(50),
            current_filter: initial_filter,
            status_message: None,
            ui_state: ui_state::UiState::new(),
            options,
            analytics,
            worker_tx: worker_tx.clone(),
            worker_handle: Some(worker_handle),
            input,
            tick_paused: Arc::new(AtomicBool::new(false)),
            something_animates: Arc::new(AtomicBool::new(false)),
            walker_done: false,
            startup_complete_logged: false,
            timings: Timings::default(),
            db_stats: None,
            db_stats_requested: false,
            data_dir: data_dir.to_path_buf(),
            event_tx,
            query_sent_at: None,
        };

        // Send initial query to worker with ID 0
        let _ = worker_tx.send(WorkerRequest::UpdateQuery(
            search_worker::UpdateQueryRequest {
                query: String::new(),
                query_id: 0,
                filter: initial_filter,
            },
        ));

        Ok(app)
    }

    /// Take the next query id, remembering when we asked so the round trip can be
    /// timed when the matching response arrives.
    ///
    /// Every path that sends work to the worker goes through here, which is what
    /// keeps "last search" honest: it measures whatever the user just did, whether
    /// that was typing, changing directory, or a filter change.
    pub fn next_query_id(&mut self) -> u64 {
        let query_id = self.analytics.next_subsession_id();
        self.query_sent_at = Some((query_id, Instant::now()));
        query_id
    }

    /// Record the round trip if `query_id` is the request we were waiting on.
    pub fn note_query_completed(&mut self, query_id: u64) {
        if let Some((pending_id, sent_at)) = self.query_sent_at
            && pending_id == query_id
        {
            let round_trip_ms = sent_at.elapsed().as_secs_f64() * 1000.0;
            log::info!(
                "TIMING {{\"op\":\"query_round_trip\",\"ms\":{}}}",
                round_trip_ms
            );
            self.timings.last_search_ms = Some(round_trip_ms);
            self.query_sent_at = None;
        }
    }

    /// Has the search worker stopped?
    ///
    /// It runs for as long as its request channel is open, which `App` holds,
    /// so while the UI is up a finished worker means a dead one. Without this
    /// the UI carried on with whatever results it had last been given, and
    /// nothing typed made any difference - a frozen list that still redraws.
    pub fn worker_has_died(&self) -> bool {
        self.worker_handle
            .as_ref()
            .is_some_and(|handle| handle.is_finished())
    }

    /// Load database statistics in the background, once.
    ///
    /// Counting rows scans the whole action index, so this is deliberately not
    /// done at startup: it happens the first time the debug pane is opened, off
    /// the UI thread, and the result arrives as an event like any other.
    pub fn request_db_stats(&mut self) {
        if self.db_stats_requested {
            return;
        }
        self.db_stats_requested = true;

        let db_path = crate::db::Database::get_db_path(&self.data_dir);
        let event_tx = self.event_tx.clone();
        std::thread::spawn(move || {
            let stats = crate::db::Database::new(&db_path).and_then(|db| db.stats(&db_path));
            match stats {
                Ok(stats) => {
                    let _ = event_tx.send(crate::AppEvent::DbStats(Box::new(stats)));
                }
                Err(e) => log::error!("Failed to gather database stats: {}", e),
            }
        });
    }

    /// Ask the worker to pick up the retrained model and the latest clicks,
    /// then rerank under `query_id`. One request does both.
    pub fn reload_ranker(&mut self, query_id: u64) -> Result<()> {
        log::info!("Requesting ranker reload from worker");
        self.worker_tx
            .send(WorkerRequest::Reload { query_id })
            .context("Failed to send Reload request to worker")?;
        Ok(())
    }

    /// Get file from page cache at a given global index
    /// Returns None if the page isn't loaded or index is out of range
    pub fn get_file_at_index(&self, index: usize) -> Option<&DisplayFileInfo> {
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

    /// Log the rows that were on screen as impressions.
    ///
    /// The rows on screen, not the top 25: an impression is the model's only
    /// evidence that something was *shown and passed over*, and a row the user
    /// never saw is not that. It used to log a fixed 25 from the top, which both
    /// invented negatives below the fold on a short terminal and missed real
    /// ones below row 25 on a tall one.
    ///
    /// Before the first frame is drawn nothing has been seen, so nothing is
    /// logged - `visible_list_height` is 0 until the renderer reports it.
    pub fn check_and_log_impressions(&mut self, force: bool) -> Result<()> {
        let first_visible = self.file_list_scroll;
        let last_visible =
            (first_visible + self.visible_list_height as usize).min(self.total_results);

        let mut top_n = Vec::new();
        for i in first_visible..last_visible {
            if let Some(display_info) = self.get_file_at_index(i) {
                top_n.push(FileMetadata {
                    relative_path: display_info.display_name.clone(),
                    full_path: display_info.full_path.to_string_lossy().to_string(),
                    mtime: display_info.mtime,
                    atime: display_info.atime,
                    size: display_info.file_size,
                    is_dir: display_info.is_dir,
                });
            }
        }

        // Delegate to analytics module
        self.analytics.check_and_log_impressions(force, top_n)
    }

    /// Ask for the preview of whatever is selected.
    ///
    /// Called after each frame, because the pane's size is a layout fact and the
    /// layout is only known once it has been computed. Repeat calls for a path
    /// already covered, or already requested in that much detail, cost nothing.
    pub fn update_preview(&mut self, pane: crate::preview::PreviewPane) {
        if self.options.no_preview {
            return;
        }

        let selection = if self.ui_state.history_mode {
            self.get_filtered_history()
                .get(self.history_selected)
                .map(|dir| (dir.clone(), true))
        } else {
            self.get_file_at_index(self.selected_index)
                .map(|info| (info.full_path.clone(), info.is_dir))
        };

        if let Some((path, is_dir)) = selection {
            self.preview.request(&path, is_dir, pane);
        }
    }

    pub fn move_selection(&mut self, delta: isize) {
        if self.total_results == 0 {
            return;
        }

        let len = self.total_results as isize;
        let new_index = (self.selected_index as isize + delta).rem_euclid(len);
        self.selected_index = new_index as usize;

        // Reset preview scroll and clear cache when changing selection
        self.preview.clear();

        // Reset marquee scroll
        self.path_bar_scroll = 0;
        self.path_bar_scroll_direction = 1;
        self.last_path_bar_update = Instant::now();
    }

    pub fn get_filtered_history(&self) -> Vec<PathBuf> {
        let query_lower = self.query.to_lowercase();

        // Get all history items (already in reverse order from the History module)
        let all_dirs = self.history.items_for_display();

        if query_lower.is_empty() {
            all_dirs
        } else {
            all_dirs
                .into_iter()
                .filter(|path| path.to_string_lossy().to_lowercase().contains(&query_lower))
                .collect()
        }
    }

    pub fn move_history_selection(&mut self, delta: isize) {
        let filtered_history = self.get_filtered_history();
        if filtered_history.is_empty() {
            return;
        }

        let len = filtered_history.len() as isize;
        let new_index = (self.history_selected as isize + delta).rem_euclid(len);
        self.history_selected = new_index as usize;

        // Clear preview cache when selection changes
        self.preview.clear();
    }

    pub fn handle_history_enter(&mut self) -> Result<()> {
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
            let query_id = self.next_query_id();
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

    pub fn log_preview_scroll(&mut self) -> Result<()> {
        if self.total_results == 0 {
            return Ok(());
        }

        // Force log impressions before scroll
        self.check_and_log_impressions(true)?;

        // Extract all data we need before borrowing analytics mutably
        if let Some(display_info) = self.get_file_at_index(self.selected_index) {
            let display_name = display_info.display_name.clone();
            let full_path = display_info.full_path.to_string_lossy().to_string();
            let mtime = display_info.mtime;
            let atime = display_info.atime;
            let file_size = display_info.file_size;
            let is_dir = display_info.is_dir;
            let query = self.query.clone();

            // Now we can safely borrow analytics
            let subsession_id = self.analytics.current_subsession_id();
            let session_id = self.analytics.session_id().to_string();

            self.analytics.log_scroll(
                &query,
                EventData {
                    query: &query,
                    file_path: &display_name,
                    full_path: &full_path,
                    mtime,
                    atime,
                    file_size,
                    subsession_id,
                    action: crate::db::UserInteraction::Scroll,
                    session_id: &session_id,
                    episode_queries: None,
                    rank: None, // not an impression
                    is_dir: Some(is_dir),
                },
            )?;
        }

        Ok(())
    }

    /// An `App` with nothing behind it, for tests that render a frame.
    ///
    /// Render takes `&App`, so a render test needs one; going through
    /// `App::new` would want a data directory, a worker thread and a terminal.
    /// This wires an in-memory database and a detached input handle instead,
    /// and leaves everything else at its default - the caller sets the two or
    /// three fields its case is about.
    #[cfg(test)]
    pub fn for_test() -> Self {
        let (event_tx, _event_rx) = mpsc::channel();
        let (preview_tx, _preview_rx) = mpsc::channel();
        let (worker_tx, _worker_rx) = mpsc::channel();
        let db = crate::db::Database::new(Path::new(":memory:")).expect("in-memory database");
        let root = PathBuf::from("/tmp");

        App {
            query: String::new(),
            page_cache: HashMap::new(),
            total_results: 0,
            total_files: 0,
            selected_index: 0,
            file_list_scroll: 0,
            preview: PreviewState::new(preview_tx),
            cwd: root.clone(),
            history: history::History::new(root),
            history_selected: 0,
            path_bar_scroll: 0,
            path_bar_scroll_direction: 1,
            path_bar_width: 0,
            visible_list_height: 0,
            last_path_bar_update: Instant::now(),
            model_stats_cache: None,
            currently_retraining: false,
            recent_logs: VecDeque::new(),
            current_filter: search_worker::FilterType::None,
            status_message: None,
            ui_state: ui_state::UiState::new(),
            options: AppOptions {
                on_dir_click: OnDirClickAction::Navigate,
                on_cwd_visit: OnCwdVisitAction::DropIntoShell,
                initial_filter: search_worker::FilterType::None,
                no_preview: false,
                no_click_loading: true,
                no_model: true,
                no_click_logging: true,
                respect_gitignore: true,
                editor: "true".to_string(),
            },
            analytics: Analytics::new("test-session".to_string(), db, true),
            worker_tx,
            worker_handle: None,
            input: crate::tty_input::TtyInput::detached(),
            tick_paused: Arc::new(AtomicBool::new(false)),
            something_animates: Arc::new(AtomicBool::new(false)),
            walker_done: false,
            startup_complete_logged: false,
            timings: Timings::default(),
            db_stats: None,
            db_stats_requested: false,
            data_dir: PathBuf::from("/tmp"),
            event_tx,
            query_sent_at: None,
        }
    }

    /// Move the path-bar marquee on by one step, if it is time.
    ///
    /// Called from the tick, not from the draw: the animation should be driven
    /// by the clock rather than by how often the screen happens to be redrawn,
    /// and having render mutate this was the last thing stopping it taking
    /// `&App`. It needs the bar's width, which only the layout knows, so the
    /// renderer reports that back in `FrameLayout`.
    pub fn advance_marquee(&mut self, delay: std::time::Duration, speed: std::time::Duration) {
        let width = self.path_bar_width as usize;
        if width == 0 {
            return; // no frame drawn yet
        }

        let Some(file) = self.get_file_at_index(self.selected_index) else {
            return;
        };
        // Borrowed, not owned: this runs on every tick, and a path that is valid
        // UTF-8 with nothing to escape - which is nearly all of them - costs no
        // allocation at all.
        let path = file.full_path.to_string_lossy();
        // Same padding the path bar draws, so the two agree on when it overflows.
        let padded_len = crate::path_display::printable(&path).chars().count() + 4;
        if padded_len <= width {
            return;
        }
        drop(path);

        let max_scroll = padded_len.saturating_sub(width) as u16;
        let at_an_end = self.path_bar_scroll == 0 || self.path_bar_scroll >= max_scroll;
        let waited = self.last_path_bar_update.elapsed();
        if waited <= if at_an_end { delay } else { speed } {
            return;
        }

        let (scroll, direction) = if self.path_bar_scroll_direction == 1 {
            if self.path_bar_scroll < max_scroll {
                (self.path_bar_scroll + 1, self.path_bar_scroll_direction)
            } else {
                (self.path_bar_scroll, -1) // turn around
            }
        } else if self.path_bar_scroll > 0 {
            (self.path_bar_scroll - 1, self.path_bar_scroll_direction)
        } else {
            (self.path_bar_scroll, 1) // turn around
        };
        self.path_bar_scroll = scroll;
        self.path_bar_scroll_direction = direction;
        self.last_path_bar_update = Instant::now();
    }

    /// Take the scroll position the frame was drawn at, and prefetch around it.
    ///
    /// The scroll itself is computed by the renderer, which is the only place
    /// that knows how many rows fit. This used to carry a second copy of that
    /// calculation for when the renderer had not supplied one - a branch that
    /// nothing could reach, since every caller comes straight from a frame.
    pub fn update_scroll(&mut self, visible_height: u16, scroll: usize) {
        let visible_height = visible_height as usize;

        // If we can't render any rows, skip scroll updates but keep selection.
        if visible_height == 0 {
            return;
        }

        if self.total_results == 0 {
            return;
        }

        let max_scroll = self.total_results.saturating_sub(1);
        self.file_list_scroll = scroll.min(max_scroll);

        let active_query_id = self.analytics.current_subsession_id();

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

#[cfg(test)]
mod impression_tests {
    //! Impressions are the rows that were on screen.
    //!
    //! They are the model's only evidence that something was shown and passed
    //! over, so logging a row nobody saw invents a negative, and missing one
    //! they did see loses a real one.

    use super::*;

    fn app_showing(total: usize) -> App {
        let mut app = App::for_test();
        app.total_results = total;
        app.page_cache.insert(
            0,
            Page {
                start_index: 0,
                end_index: total,
                files: (0..total)
                    .map(|i| crate::search_worker::DisplayFileInfo {
                        display_name: format!("file{}.rs", i),
                        full_path: PathBuf::from(format!("/tmp/file{}.rs", i)),
                        score: 0.0,
                        features: Vec::new(),
                        mtime: None,
                        atime: None,
                        file_size: None,
                        is_dir: false,
                        is_cwd: false,
                        is_historical: false,
                        is_under_cwd: true,
                        simple_score: None,
                        ml_score: None,
                        simple_weight: None,
                        ml_weight: None,
                        fuzzy_score: 0,
                    })
                    .collect(),
            },
        );
        app
    }

    /// The rows `check_and_log_impressions` would log, by name.
    fn would_log(app: &App) -> Vec<String> {
        let first = app.file_list_scroll;
        let last = (first + app.visible_list_height as usize).min(app.total_results);
        (first..last)
            .filter_map(|i| app.get_file_at_index(i))
            .map(|f| f.display_name.clone())
            .collect()
    }

    #[test]
    fn test_nothing_is_logged_before_a_frame_has_been_drawn() {
        // visible_list_height is 0 until the renderer reports it, and a row
        // that has not been drawn has not been seen.
        let app = app_showing(50);
        assert!(would_log(&app).is_empty());
    }

    #[test]
    fn test_a_short_terminal_logs_only_what_fits() {
        let mut app = app_showing(50);
        app.visible_list_height = 8;

        assert_eq!(
            would_log(&app).len(),
            8,
            "eight rows on screen, eight logged"
        );
        assert_eq!(would_log(&app)[0], "file0.rs");
    }

    #[test]
    fn test_a_tall_terminal_logs_past_the_old_limit_of_25() {
        let mut app = app_showing(50);
        app.visible_list_height = 40;

        assert_eq!(
            would_log(&app).len(),
            40,
            "forty rows were on screen; the old code logged 25 of them"
        );
    }

    #[test]
    fn test_scrolling_moves_the_window_rather_than_the_top() {
        let mut app = app_showing(50);
        app.visible_list_height = 10;
        app.file_list_scroll = 20;

        let logged = would_log(&app);
        assert_eq!(logged.first().unwrap(), "file20.rs");
        assert_eq!(logged.last().unwrap(), "file29.rs");
    }

    #[test]
    fn test_the_window_stops_at_the_last_result() {
        let mut app = app_showing(3);
        app.visible_list_height = 40;

        assert_eq!(
            would_log(&app).len(),
            3,
            "no rows invented below the results"
        );
    }
}

#[cfg(test)]
mod marquee_tests {
    //! The path bar scrolls long paths back and forth. The tick drives it, so
    //! it is testable without drawing anything - which is the point of having
    //! moved it out of render.

    use super::*;
    use std::time::Duration;

    /// An app showing one file, with the path bar as wide as `width`.
    fn app_showing(path: &str, width: u16) -> App {
        let mut app = App::for_test();
        app.total_results = 1;
        app.path_bar_width = width;
        app.page_cache.insert(
            0,
            Page {
                start_index: 0,
                end_index: 1,
                files: vec![crate::search_worker::DisplayFileInfo {
                    display_name: path.to_string(),
                    full_path: PathBuf::from(path),
                    score: 0.0,
                    features: Vec::new(),
                    mtime: None,
                    atime: None,
                    file_size: None,
                    is_dir: false,
                    is_cwd: false,
                    is_historical: false,
                    is_under_cwd: true,
                    simple_score: None,
                    ml_score: None,
                    simple_weight: None,
                    ml_weight: None,
                    fuzzy_score: 0,
                }],
            },
        );
        app
    }

    /// Long enough to overflow a 20-column bar: 24 characters plus 4 of padding.
    const LONG: &str = "/tmp/a/long/path/one.txt";
    const NOW: Duration = Duration::ZERO;

    #[test]
    fn test_a_path_that_fits_does_not_scroll() {
        let mut app = app_showing("/tmp/short.txt", 80);
        app.advance_marquee(NOW, NOW);
        assert_eq!(app.path_bar_scroll, 0, "nothing to scroll to");
    }

    #[test]
    fn test_a_long_path_scrolls_one_column_at_a_time() {
        let mut app = app_showing(LONG, 20);

        app.advance_marquee(NOW, NOW);
        assert_eq!(app.path_bar_scroll, 1);
        app.advance_marquee(NOW, NOW);
        assert_eq!(app.path_bar_scroll, 2);
    }

    #[test]
    fn test_it_turns_around_at_the_far_end_and_comes_back() {
        let mut app = app_showing(LONG, 20);
        // 24 characters + 4 of padding, in 20 columns, leaves 8 to scroll.
        let max_scroll = 8;

        for _ in 0..max_scroll {
            app.advance_marquee(NOW, NOW);
        }
        assert_eq!(app.path_bar_scroll, max_scroll, "walked to the end");
        assert_eq!(app.path_bar_scroll_direction, 1, "still facing forwards");

        app.advance_marquee(NOW, NOW);
        assert_eq!(app.path_bar_scroll, max_scroll, "the turn costs a step");
        assert_eq!(app.path_bar_scroll_direction, -1, "now facing back");

        app.advance_marquee(NOW, NOW);
        assert_eq!(app.path_bar_scroll, max_scroll - 1, "and comes back");
    }

    #[test]
    fn test_it_waits_the_long_delay_at_the_ends_and_the_short_one_between() {
        let mut app = app_showing(LONG, 20);
        let delay = Duration::from_secs(3600);
        let speed = Duration::ZERO;

        // At rest at column 0, so the long delay applies and nothing moves.
        app.advance_marquee(delay, speed);
        assert_eq!(app.path_bar_scroll, 0, "paused at the end");

        // Once moving, the short one does.
        app.advance_marquee(speed, speed);
        assert_eq!(app.path_bar_scroll, 1);
        app.advance_marquee(delay, speed);
        assert_eq!(app.path_bar_scroll, 2, "mid-scroll uses the scroll speed");
    }

    #[test]
    fn test_nothing_moves_before_the_first_frame() {
        // path_bar_width is 0 until a frame has been laid out, and a marquee
        // that guessed a width would scroll to the wrong place.
        let mut app = app_showing(LONG, 0);
        app.advance_marquee(NOW, NOW);
        assert_eq!(app.path_bar_scroll, 0);
    }
}
