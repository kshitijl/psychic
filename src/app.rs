use anyhow::{Context, Result};
use ratatui::text::Text;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver},
    thread::JoinHandle,
    time::Instant,
};

use crate::cli::{OnCwdVisitAction, OnDirClickAction};
use crate::db::{Database, EventData, FileMetadata};
use crate::search_worker::{self, DisplayFileInfo, WorkerRequest, WorkerResponse};
use crate::{history, ranker, ui_state};

/// A page of DisplayFileInfo for caching
#[derive(Debug, Clone)]
pub struct Page {
    pub start_index: usize,
    pub end_index: usize,
    pub files: Vec<DisplayFileInfo>,
}

/// Every time the query changes, as the user types, corresponds to a new
/// subsession. Subsession id is logged to the db.
pub struct Subsession {
    pub id: u64,
    pub query: String,
    pub created_at: jiff::Timestamp,
    pub events_have_been_logged: bool,
}

/// Preview cache state - tracks whether we have a light or full preview cached
#[derive(Clone)]
pub enum PreviewCache {
    /// No preview cached
    None,
    /// Light preview cached (first N lines only) for a file
    Light { path: String, text: Text<'static> },
    /// Full preview cached (entire file) for a file
    Full { path: String, text: Text<'static> },
    /// Directory preview cached (eza output)
    Directory {
        path: String,
        text: Text<'static>,
        extra_flags: String,
    },
}

impl PreviewCache {
    pub fn get_if_matches(&self, path: &str) -> Option<(Text<'static>, bool)> {
        match self {
            PreviewCache::None => None,
            PreviewCache::Light {
                path: cached_path,
                text,
            } if cached_path == path => Some((text.clone(), false)),
            PreviewCache::Full {
                path: cached_path,
                text,
            } if cached_path == path => Some((text.clone(), true)),
            PreviewCache::Directory { .. } => None, // Use get_dir_if_matches instead
            _ => None,
        }
    }

    pub fn get_dir_if_matches(&self, path: &str, extra_flags: &str) -> Option<Text<'static>> {
        match self {
            PreviewCache::Directory {
                path: cached_path,
                text,
                extra_flags: cached_flags,
            } if cached_path == path && cached_flags == extra_flags => Some(text.clone()),
            _ => None,
        }
    }
}

// Page-based caching constants
pub const PAGE_SIZE: usize = 128;
pub const PREFETCH_MARGIN: usize = 32;

pub struct App {
    pub query: String,
    pub page_cache: HashMap<usize, Page>, // Page-based cache
    pub total_results: usize,             // Total number of filtered files
    pub total_files: usize,               // Total number of files in index
    pub selected_index: usize,
    pub file_list_scroll: usize, // Scroll offset for file list
    pub preview_scroll: usize,
    pub preview_cache: PreviewCache,
    pub scrolled_files: HashSet<(String, String)>, // (query, full_path) - track what we've logged scroll for
    pub session_id: String,
    pub current_subsession: Option<Subsession>,
    pub next_subsession_id: u64,
    pub db: Database,
    pub cwd: PathBuf, // Current working directory
    pub history: history::History,
    pub history_selected: usize, // Selected item in history mode UI

    pub num_results_to_log_as_impressions: usize,

    // For marquee path bar
    pub path_bar_scroll: u16,
    pub path_bar_scroll_direction: i8,
    pub last_path_bar_update: Instant,

    // For debug pane
    pub model_stats_cache: Option<ranker::ModelStats>, // Cached from worker, refreshed periodically
    pub currently_retraining: bool,
    pub log_receiver: Receiver<String>,
    pub recent_logs: VecDeque<String>,

    // Filter state
    pub current_filter: search_worker::FilterType,

    // UI state machine
    pub ui_state: ui_state::UiState,

    // Action configuration
    pub on_dir_click: OnDirClickAction,
    pub on_cwd_visit: OnCwdVisitAction,

    // Performance flags
    pub no_preview: bool,
    pub no_click_logging: bool,

    // Search worker thread communication
    pub worker_tx: mpsc::Sender<WorkerRequest>,
    pub worker_handle: Option<JoinHandle<()>>,

    // Crossterm thread control (for pausing when launching child processes)
    pub input_control_tx: crossbeam::channel::Sender<bool>,

    // Startup tracking
    pub walker_done: bool,
    pub startup_complete_logged: bool,
}

impl App {
    pub fn new(
        root: PathBuf,
        data_dir: &Path,
        log_receiver: Receiver<String>,
        on_dir_click: OnDirClickAction,
        on_cwd_visit: OnCwdVisitAction,
        initial_filter: search_worker::FilterType,
        no_preview: bool,
        no_click_loading: bool,
        no_model: bool,
        no_click_logging: bool,
        event_tx: mpsc::Sender<crate::AppEvent>,
        input_control_tx: crossbeam::channel::Sender<bool>,
    ) -> Result<Self> {
        let start_time = Instant::now();
        log::debug!("App::new() started");

        // Get session ID from environment (set in main())
        let session_id =
            std::env::var("PSYCHIC_SESSION_ID").unwrap_or_else(|_| "unknown".to_string());

        let db_start = Instant::now();
        let db_path = Database::get_db_path(data_dir);
        let db = Database::new(&db_path)?;
        log::debug!("Database initialization took {:?}", db_start.elapsed());

        let (worker_tx, worker_handle) = search_worker::spawn(
            root.clone(),
            data_dir,
            event_tx.clone(),
            no_click_loading,
            no_model,
        )?;

        log::debug!("App::new() total time: {:?}", start_time.elapsed());

        let app = App {
            query: String::new(),
            page_cache: HashMap::new(),
            total_results: 0,
            total_files: 0,
            selected_index: 0,
            file_list_scroll: 0,
            preview_scroll: 0,
            preview_cache: PreviewCache::None,
            scrolled_files: HashSet::new(),
            session_id,
            current_subsession: None,
            next_subsession_id: 1, // Start with 1, 0 is for initial query
            num_results_to_log_as_impressions: 25,
            db,
            cwd: root.clone(),
            history: history::History::new(root),
            history_selected: 0,
            path_bar_scroll: 0,
            path_bar_scroll_direction: 1,
            last_path_bar_update: Instant::now(),
            model_stats_cache: None,
            currently_retraining: false,
            log_receiver,
            recent_logs: VecDeque::with_capacity(50),
            current_filter: initial_filter,
            ui_state: ui_state::UiState::new(),
            on_dir_click,
            on_cwd_visit,
            no_preview,
            no_click_logging,
            worker_tx: worker_tx.clone(),
            worker_handle: Some(worker_handle),
            input_control_tx,
            walker_done: false,
            startup_complete_logged: false,
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

    pub fn reload_model(&mut self, query_id: u64) -> Result<()> {
        log::info!("Requesting model reload from worker");
        self.worker_tx
            .send(WorkerRequest::ReloadModel { query_id })
            .context("Failed to send ReloadModel request to worker")?;
        Ok(())
    }

    pub fn reload_and_rerank(&mut self, query_id: u64) -> Result<()> {
        log::info!("Requesting clicks reload from worker");
        self.worker_tx
            .send(WorkerRequest::ReloadClicks { query_id })
            .context("Failed to send ReloadClicks request to worker")?;
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

    pub fn check_and_log_impressions(&mut self, force: bool) -> Result<()> {
        // Skip logging if disabled
        if self.no_click_logging {
            return Ok(());
        }

        // Extract values we need before borrowing subsession mutably
        let (subsession_id, subsession_query, created_at, already_logged) =
            match &self.current_subsession {
                Some(s) => (
                    s.id,
                    s.query.clone(),
                    s.created_at,
                    s.events_have_been_logged,
                ),
                None => return Ok(()),
            };

        // Skip if already logged
        if already_logged {
            return Ok(());
        }

        // Check if we should log: either forced or >200ms old
        let elapsed = jiff::Timestamp::now().duration_since(created_at);
        let threshold = jiff::SignedDuration::from_millis(200);
        let should_log = force || elapsed >= threshold;
        if !should_log {
            return Ok(());
        }

        // Log top N visible files with metadata
        // Collect from page cache starting at index 0
        let mut top_n = Vec::new();
        for i in 0..self
            .num_results_to_log_as_impressions
            .min(self.total_results)
        {
            if let Some(display_info) = self.get_file_at_index(i) {
                top_n.push(FileMetadata {
                    relative_path: display_info.display_name.clone(),
                    full_path: display_info.full_path.to_string_lossy().to_string(),
                    mtime: display_info.mtime,
                    atime: display_info.atime,
                    size: display_info.file_size,
                });
            }
        }

        if !top_n.is_empty() {
            self.db
                .log_impressions(&subsession_query, &top_n, subsession_id, &self.session_id)?;

            // Mark as logged
            if let Some(ref mut s) = self.current_subsession {
                s.events_have_been_logged = true;
            }
        }

        Ok(())
    }

    pub fn move_selection(&mut self, delta: isize) {
        if self.total_results == 0 {
            return;
        }

        let len = self.total_results as isize;
        let new_index = (self.selected_index as isize + delta).rem_euclid(len);
        self.selected_index = new_index as usize;

        // Reset preview scroll and clear cache when changing selection
        self.preview_scroll = 0;
        self.preview_cache = PreviewCache::None;

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
        self.preview_cache = PreviewCache::None;
        self.preview_scroll = 0;
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
            let query_id = self.next_subsession_id;
            self.next_subsession_id += 1;
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
        if self.no_click_logging || self.total_results == 0 {
            return Ok(());
        }

        // Force log impressions before scroll
        self.check_and_log_impressions(true)?;

        if let Some(display_info) = self.get_file_at_index(self.selected_index) {
            let full_path_str = display_info.full_path.to_string_lossy().to_string();
            let key = (self.query.clone(), full_path_str.clone());

            if !self.scrolled_files.contains(&key) {
                let subsession_id = self.current_subsession.as_ref().map(|s| s.id).unwrap_or(1);
                self.db.log_event(EventData {
                    query: &self.query,
                    file_path: &display_info.display_name,
                    full_path: &full_path_str,
                    mtime: display_info.mtime,
                    atime: display_info.atime,
                    file_size: display_info.file_size,
                    subsession_id,
                    action: crate::db::UserInteraction::Scroll,
                    session_id: &self.session_id,
                })?;
                self.scrolled_files.insert(key);
            }
        }

        Ok(())
    }

    pub fn update_scroll(&mut self, visible_height: u16) {
        let visible_height = visible_height as usize;
        if self.total_results == 0 {
            return;
        }

        // If all results fit on screen, don't scroll at all
        if self.total_results <= visible_height {
            self.file_list_scroll = 0;
        } else {
            // Auto-scroll the file list when selection is near top or bottom
            let selected = self.selected_index;
            let scroll = self.file_list_scroll;

            // If selected item is above visible area, scroll up
            if selected < scroll {
                self.file_list_scroll = selected;
            }
            // If selected item is below visible area, scroll down
            else if selected >= scroll + visible_height {
                // Smart positioning: leave some space from bottom (5 lines)
                // This makes wrap-around more comfortable
                let margin = 5usize;
                self.file_list_scroll = selected.saturating_sub(
                    visible_height
                        .saturating_sub(margin)
                        .min(visible_height - 1),
                );
            }
            // If we're in the bottom 5 items and there's more to see, keep scrolling
            else if selected >= scroll + visible_height.saturating_sub(5) {
                self.file_list_scroll = selected.saturating_sub(visible_height.saturating_sub(5));
            }
        }

        let active_query_id = self.current_subsession.as_ref().map(|s| s.id).unwrap_or(0);

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
