# How It Works

## About this document

This document describes the current state of the code, with a brief description of why certain things are done the way they are. It's not meant to be an exhaustive list of all previous iterations of the code. It shouldn't describe how things used to work. Anything learned from previous designs should only show up in the brie "why" statements.

## Overview

`psychic` is a terminal-based file browser with fuzzy search, ML-powered ranking, and click tracking analytics. Built in Rust using ratatui for the TUI.

## Architecture

### Core Components

1. **`db.rs`** - SQLite event logging
2. **`walker.rs`** - Background file discovery
3. **`context.rs`** - System context gathering
4. **`features.rs`** - ML feature generation for training
5. **`feature_defs/`** - Trait-based feature registry
6. **`ranker.rs`** - LightGBM model inference
7. **`search_worker.rs`** - Async worker thread for filtering/ranking
8. **`ui_state.rs`** - UI state machine (history mode, filter picker, debug pane)
9. **`history.rs`** - Directory navigation history with branch-point semantics
10. **`main.rs`** - TUI event loop and rendering

Why: Worker thread architecture prevents UI lag during expensive filtering/ranking operations.

### Thread Architecture

**3 threads:**
- **Main (UI)**: Renders UI, handles keyboard/mouse, owns visible file slice only
- **Worker**: Owns all file data, does filtering/ranking, sends results to main
- **Walker**: Discovers files via walkdir, sends to worker

**Communication:**
```
Walker → (path, mtime) → Worker

// UI generates a unique ID for every request
Main → UpdateQuery{query: "foo", query_id: 1} → Worker
Worker → QueryUpdated{query_id: 1, ...} → Main

Main → GetPage{query_id: 1, page_num: 2} → Worker
Worker → Page{query_id: 1, ...} → Main
```

Why: Worker owns file data to avoid blocking UI. Main thread only keeps what's visible (20-40 files), not all files (thousands).

### Robust Communication with Query IDs

A `query_id` represents a query string plus the immutable set of search results generated for it at a specific moment in time. By assigning a unique ID to every new result set, we can treat them as distinct, versioned objects.

This is critical for correctness in an asynchronous environment. For example, without it, a `GetPage` request for an old result set could be processed against a new result set (e.g., from an auto-refresh), which might have a different number of total items, leading to panics or data corruption in the UI's page cache.

To solve this, a robust request-response protocol was implemented:

1.  **UI as Client:** The UI thread acts as the client, and is the sole source of truth for request identity.
2.  **UI-Generated IDs:** For any action that will result in a new set of filtered files (typing, reloading the model, or an auto-refresh from file changes), the UI generates a new, unique `query_id`. This ID is reused from the existing `subsession_id` mechanism.
3.  **ID'd Requests:** Every request from the UI to the worker (`UpdateQuery`, `GetPage`, `ReloadModel`) carries the relevant `query_id`.
4.  **ID'd Responses:** Every response from the worker back to the UI (`QueryUpdated`, `Page`) also carries the `query_id` of the request it is responding to.

This ensures that both the UI and the worker can safely discard stale messages, preventing state corruption and crashes.

### Module: `db.rs`

Database at `~/.local/share/psychic/events.db` with two tables:

```sql
CREATE TABLE events (
    timestamp INTEGER,
    query TEXT,
    file_path TEXT,      -- relative path
    full_path TEXT,      -- absolute path
    mtime INTEGER,
    atime INTEGER,
    file_size INTEGER,
    subsession_id INTEGER,
    action TEXT,         -- 'impression', 'scroll', or 'click'
    session_id TEXT
);

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    cwd TEXT,
    gateway TEXT,
    subnet TEXT,
    dns TEXT,
    shell_history TEXT,
    running_processes TEXT,
    timezone TEXT,
    created_at INTEGER
);
```

**Index:**
```sql
CREATE INDEX idx_events_click_lookup ON events(action, timestamp, full_path);
```

Why: Composite index speeds up 30-day click count aggregation (O(log n) vs O(n)).

**Session ID:** Random 64-bit integer (not UUID).
Why: UUIDs are 36 chars. 64-bit int gives 18 quintillion IDs, more compact.

**File metadata:** Captured at event time, not discovery time.
Why: Files can be modified between discovery and impression. Event-time metadata reflects what user actually saw.

### Module: `walker.rs`

Background thread that recursively walks current directory using `walkdir`.

**Key points:**
- Streams results via mpsc channel using `WalkerMessage` enum
- Filters: `.git`, `node_modules`, `.venv`, `target`
- Sends both files and directories (with `is_dir` flag)
- Extracts mtime, atime, and file_size from walkdir's cached metadata
- Adaptive depth: switches to depth=1 if >8k items found (shallow mode)
- Sends `AllDone` message when walk completes
- Supports dynamic directory changes via `ChangeCwd` command
- Checks for commands every 100 files (COMMAND_CHECK_INTERVAL)

**Shallow mode:** If walker encounters more than 8,000 items during full-depth exploration, it aborts and restarts with `max_depth(1)` to only show first-level items. This prevents performance issues in large directory trees like `~/`.

**Command support:** Walker can receive `ChangeCwd` commands to restart walking in a new directory. It checks for commands periodically (every 100 files) during walks to remain responsive to navigation requests.

**WalkerMessage enum:**
```rust
pub enum WalkerMessage {
    FileMetadata(WalkerFileMetadata),
    AllDone,
}
```

The `AllDone` message bypasses the worker's 200ms debounce to ensure the UI updates immediately when the walk completes, even if it finishes quickly.

Why background thread: Large directories take seconds to scan. Streaming keeps UI responsive.
Why send metadata: Avoids re-fetching later (performance).
Why AllDone: Ensures UI updates even when walker completes in <200ms (debounce bypass).
Why periodic command checks: Allows aborting long walks when user navigates to different directory.

### Module: `context.rs`

Gathers system context at startup in background thread:
- `cwd` - Current working directory
- `gateway` - Default gateway from `netstat -nr`
- `subnet` - First two octets of local IP
- `dns` - First DNS nameserver from `scutil --dns`
- `shell_history` - Last 10 commands from ~/.zsh_history or ~/.bash_history
- `running_processes` - Output of `ps -u $USER -o pid,comm`
- `timezone` - Multi-tier fallback: $TZ env var → /etc/localtime symlink → "UTC"

Why gather this: Network context (home/office/cafe), shell history (user intent), and running processes help analyze search patterns and could become ML features.

**Error handling:** All fields fallback to "unknown" on error. Never crash due to missing tools.

**Timezone detection:** Uses sophisticated fallback chain:
1. Check $TZ environment variable
2. Read /etc/localtime symlink and extract timezone from path (handles both /usr/share/zoneinfo and /var/db/timezone paths)
3. Default to "UTC" if all methods fail

### Module: `search_worker.rs`

Worker thread owns all file data and processes queries asynchronously.

**WorkerState owns:**
- `file_registry: Vec<FileInfo>` - All files with metadata
- `path_to_id: HashMap<PathBuf, FileId>` - Deduplication
- `filtered_files: Vec<FileId>` - Ranked result IDs
- `file_scores: Vec<FileScore>` - Scores and features
- `current_filter: FilterType` - Active filter (None, OnlyCwd, OnlyDirs, OnlyFiles)
- `ranker: Ranker` - ML model

**Worker loop:**
1. Process walker updates (non-blocking `try_recv`)
   - `FileMetadata`: Add file to registry, mark files_changed
   - `AllDone`: Mark walker_done, force immediate FilesChanged notification
2. Send FilesChanged if files changed AND (walker_done OR >200ms since last notification)
3. Process work requests with 5ms timeout
4. Debounce queries (drain channel, keep latest)

**Debouncing:** If user types "hello" quickly, only process final query (not 5 intermediate queries).
Why: Avoids wasted computation and improves responsiveness.

**Auto-refresh:** When new files arrive from the walker, the worker sends a `FilesChanged` notification to the UI thread (debounced to 200ms intervals). When walker sends `AllDone`, the debounce is bypassed to ensure immediate UI update. The UI is then responsible for triggering a new query with a new `query_id` to get fresh results. This preserves the "UI generates IDs" architecture and ensures the page cache is handled correctly.

**Filtering:** The worker applies filters in `filter_and_rank()`. Text queries use **case-insensitive substring matching** (`.contains()`), not fuzzy matching. Filters are additive - both the text query and the filter type must match. Filter logic:
- `FilterType::None`: No additional filtering beyond text query
- `FilterType::OnlyCwd`: Only files with `origin == FileOrigin::CwdWalker` (excludes historical files)
- `FilterType::OnlyDirs`: Only files where `is_dir == true`
- `FilterType::OnlyFiles`: Only files where `is_dir == false`

The filter is part of `UpdateQueryRequest` and persists across query changes until the user selects a different filter.

**File Registry Design:**

```rust
struct FileId(usize);  // Newtype for type safety

struct FileInfo {
    full_path: PathBuf,
    display_name: String,  // Computed once (relative path or ".../filename" or directory name for cwd)
    mtime: Option<i64>,    // From walker or historical load
    origin: FileOrigin,    // CwdWalker or UserClickedInEventsDb
}
```

Why FileId: O(1) lookups. No string cloning in hot paths. Type-safe (can't mix with other usize).
Why display_name computed once: Filtering checks display_name repeatedly. Computing it once at registration avoids repeated allocations.
Why origin tracking: Historical files (from other directories) show as ".../filename" to indicate they're not local.

**Current Working Directory Display:**
The current working directory itself appears in search results as just its directory name (not the full path) with a " (cwd)" suffix in magenta (or yellow when selected). This makes it easy to navigate into the current directory for exploration. Clicking on it is properly logged in the events database for ML training.

On startup, psychic automatically logs a click event for the initial directory (with an empty query string), ensuring that directories you browse to via psychic will appear in your historical files in future sessions.

**Historical files:** Loads previously clicked/scrolled files from events.db at startup.
Why: User can find files from other projects they've accessed before.

### Module: `feature_defs/`

Trait-based feature registry - single source of truth for all features.

**Architecture:**
```rust
// schema.rs
pub trait Feature: Send + Sync {
    fn name(&self) -> &'static str;
    fn feature_type(&self) -> FeatureType;
    fn compute(&self, inputs: &FeatureInputs) -> Result<f64>;
}

// implementations.rs - all features implement trait
pub struct FilenameStartsWithQuery;
impl Feature for FilenameStartsWithQuery {
    fn name(&self) -> &'static str { "filename_starts_with_query" }
    fn feature_type(&self) -> FeatureType { FeatureType::Binary }
    fn compute(&self, inputs: &FeatureInputs) -> Result<f64> { /* ... */ }
}

// registry.rs - THE SINGLE SOURCE OF TRUTH
pub static FEATURE_REGISTRY: Lazy<Vec<Box<dyn Feature>>> = Lazy::new(|| {
    vec![
        Box::new(FilenameStartsWithQuery),
        Box::new(ClicksLast30Days),
        Box::new(ModifiedToday),
        Box::new(IsUnderCwd),
        Box::new(IsHidden),
    ]
});
```

Why: Adding a new feature only requires: (1) implement trait, (2) add to registry. No manual synchronization across files. Feature vector order comes from registry order automatically.

**Schema export:** `generate-features` command outputs `feature_schema.json` with feature names, types, and monotonicity hints.
Why: Python training script reads schema to know feature order, types, and can apply monotonic constraints during training. No manual duplication.

**Monotonicity hints:** Each feature can declare its monotonicity (increasing=1, decreasing=-1, or null for none). For example:
- `clicks_last_hour`: increasing (more clicks → higher relevance)
- `modified_age`: decreasing (older files → lower relevance)
- `filename_starts_with_query`: no monotonicity (binary feature)

**Query-specific features:** The `clicks_for_this_query` feature tracks clicks for specific (query, file) pairs. This distinguishes between files clicked for different search contexts - e.g., a file clicked 10 times for query "config" vs 0 times for query "test" is more relevant for "config" searches.
Why: General click counts don't capture query-specific relevance. A frequently clicked file for one query may be irrelevant for another.

### Module: `features.rs`

Generates training data from events database.

**Approach:** Single-pass fold with temporal correctness.

```rust
struct Accumulator {
    clicks_by_file: HashMap<String, Vec<ClickEvent>>,
    scrolls_by_file: HashMap<String, Vec<ScrollEvent>>,
    pending_impressions: HashMap<(String, u64, String), PendingImpression>,
    output_rows: Vec<HashMap<String, String>>,
    current_group_id: u64,
}
```

**Process:**
1. Load all events and sessions from database
2. Sort by timestamp (critical for temporal correctness)
3. Single forward pass:
   - Impression: Compute features from accumulator (only past data), store as pending
   - Click/Scroll: Record in accumulator, mark matching pending impressions as label=1, increment group_id
4. Output all pending impressions as CSV rows

Why single-pass: O(n) instead of O(n²). No future data leakage (features only see past events).

**Group-based ranking:** Each group spans from one engagement event to the next.
Why: LambdaRank needs groups. Each group = impressions leading to an action. More meaningful than subsession-based grouping.

**Features computed:** See `feature_defs/implementations.rs` for full list. Examples:
- Query matching: filename_starts_with_query
- Click history: clicks_last_30_days, clicks_last_7_days, clicks_last_hour, clicks_today, clicks_for_this_query
- File properties: is_hidden, is_under_cwd
- Temporal: modified_today, modified_age
- Directory features: clicks_last_week_parent_dir

### Module: `ranker.rs`

LightGBM model inference for scoring files.

```rust
pub struct Ranker {
    model: Booster,
    clicks: ClickData,
    stats: Option<ModelStats>,
}

pub struct ClickData {
    clicks_by_file: HashMap<String, Vec<ClickEvent>>,
    clicks_by_parent_dir: HashMap<PathBuf, Vec<ClickEvent>>,
    clicks_by_query_and_file: HashMap<(String, String), Vec<ClickEvent>>,
}
```

**Preloading clicks:** All click events from last 30 days loaded at startup into multiple HashMaps:
- `clicks_by_file`: Indexed by full file path
- `clicks_by_parent_dir`: Indexed by parent directory path
- `clicks_by_query_and_file`: Indexed by (query, full_path) tuple for query-specific click tracking

Why: O(1) lookup per file vs O(n) query per file. Database query runs once with composite index. Multiple indices enable different features without re-querying the database.

**Ranking:**
```rust
pub fn rank_files(
    &self,
    query: &str,
    file_candidates: &[FileCandidate],
    current_timestamp: i64,
    cwd: &PathBuf,
) -> Result<Vec<FileScore>>
```

Returns `Vec<FileScore>` sorted by predicted relevance (descending).

**FileScore:**
```rust
pub struct FileScore {
    pub file_id: usize,  // Index into file registry
    pub score: f64,
    pub features: Vec<f64>,  // For debug display
}
```

Why file_id instead of path string: No string cloning in hot path. Direct O(1) mapping back to file registry.

**Thread safety:** Ranker wrapped in `SendRanker` with `unsafe impl Send`.
Why: LightGBM Booster contains raw pointers (not Send by default). Safe because model is read-only.

### Training: `train.py`

Trains LightGBM LambdaRank model from features CSV.

**Key parameters:**
- Objective: `lambdarank`
- Metric: NDCG (Normalized Discounted Cumulative Gain)
- Grouping: `group_id` (engagement-based sequences)

**Usage:**
```bash
psychic generate-features  # Outputs features.csv + feature_schema.json
python train.py features.csv output
# Outputs:
#   - output.txt (model file)
#   - ~/.local/share/psychic/model.txt (copy for TUI)
#   - output_viz.pdf (feature importance, SHAP, metrics)
```

**Schema integration:** Reads `feature_schema.json` to get feature names and types. Errors if missing.
Why: Ensures Rust and Python agree on feature order.

**Visualizations:** Training curves, feature importance, SHAP analysis, score distributions, rank position analysis.

### Module: `ui_state.rs`

Pure state machine for UI mode transitions (no IO, just state).

**State components:**
```rust
pub struct UiState {
    pub history_mode: bool,
    pub filter_picker_visible: bool,
    pub debug_pane_mode: DebugPaneMode,  // Hidden, Small, or Expanded
}
```

**Methods:**
- `toggle_history_mode()`, `enter_history_mode()`, `exit_history_mode()`
- `toggle_filter_picker()`, `hide_filter_picker()`
- `cycle_debug_pane_mode()` - cycles through Hidden → Small → Expanded → Hidden
- `get_eza_flags(width)` - returns compact flags for narrow screens (<80 cols)
- `is_debug_pane_visible()`, `is_debug_pane_expanded()`

Why separate module: Testable pure functions. State transitions are tested with expect tests (no IO).
Why DebugPaneMode enum: Prevents invalid states, makes cycling logic explicit.

### Module: `history.rs`

Directory navigation history with browser-style branch-point semantics.

**Model:**
```rust
pub struct History {
    dirs: Vec<PathBuf>,        // Chronological order (dirs[i+1] came after dirs[i])
    current_index: usize,      // Current position in history
}
// Invariant: cwd == dirs[current_index]
```

**Key operations:**
- `items_for_display()` - Returns dirs in reverse order (most recent first)
- `navigate_to_display_index(i)` - Navigate to item in display list
- `navigate_to(new_dir)` - Navigate forward or branch
- `current_display_index()` - Get current position for UI cursor

**Branch-point behavior:**
When navigating to a directory:
1. If it's the next item in history (dirs[current_index + 1]), just increment current_index (preserve future)
2. Otherwise, truncate history after current_index and append new directory (create new branch, discard future)

Why: Browser-style history is intuitive. Users can go back, then navigate elsewhere to create a new branch.
Why chronological storage + reverse display: Natural for append, efficient for display.
Why tested: 11 unit tests verify invariants, edge cases, and the bug fix for history disappearing.

### Module: `main.rs`

TUI event loop using ratatui + crossterm.

**Startup behavior:**
- Spawns a background thread to retrain the model using collected events
- Training runs asynchronously and doesn't block the UI
- Training output is appended to `~/.local/share/psychic/training.log`
- Worker loads the new model automatically when retraining completes
Why: Fresh model on every launch ensures ranking improves as you use the tool. Background execution means no startup delay.

**Layout:**
```
┌─────────────┬─────────────┬──────────────┐
│  File List  │   Preview   │ Debug/Stats  │
│   (35%)     │    (45%)    │    (20%)     │
│             │             │              │
│ 1. file.rs  │ [bat output]│ Score: 0.72  │
│ 2. main.rs  │             │ Features:    │
│ ...         │             │  Clicks: 3   │
└─────────────┴─────────────┴──────────────┘
┌──────────────────────────────────────────┐
│   Search Input (bottom)                  │
└──────────────────────────────────────────┘
```

**Debug pane modes:** Ctrl-O cycles through three states:
- Hidden (default): No debug pane visible
- Small: Debug pane at 20% width
- Expanded: Debug pane at 75% width, file list at 25%, preview hidden
Why: Progressive disclosure - hide when not needed, expand for detailed debugging.

**App structure:**
```rust
struct App {
    // Search state
    query: String,
    page_cache: HashMap<usize, Page>,
    total_results: usize,
    total_files: usize,
    selected_index: usize,

    // UI state
    ui_state: ui_state::UiState,
    history: history::History,
    history_selected: usize,

    // Analytics
    current_subsession: Option<Subsession>,
    next_subsession_id: u64,
    scrolled_files: HashSet<(String, String)>,
    session_id: String,

    // Configuration
    on_dir_click: OnDirClickAction,    // Navigate, PrintToStdout, or DropIntoShell
    on_cwd_visit: OnCwdVisitAction,    // PrintToStdout or DropIntoShell
    current_filter: search_worker::FilterType,

    // Worker communication
    worker_tx: mpsc::Sender<WorkerRequest>,
    worker_rx: mpsc::Receiver<WorkerResponse>,

    // ... (other fields)
}
```

**Subsession tracking:**
```rust
struct Subsession {
    id: u64,
    query: String,
    created_at: Instant,
    events_have_been_logged: bool,
}
```

New subsession metadata is created when the UI receives a valid `QueryUpdated` response from the worker. The `id` is generated from a counter in the UI thread (`App.next_subsession_id`) and serves a dual purpose:
1.  It links analytics events (`impression`, `click`) together for a given search.
2.  It acts as the `query_id` for the robust communication protocol with the search worker.

Impressions are logged to the database after a 200ms debounce period (to avoid logging on every keystroke) or immediately before a click/scroll event (to ensure temporal correctness).

**Action configuration:**
- `on_dir_click`: What happens when user presses Enter on a directory
  - `Navigate`: Change into that directory (default)
  - `PrintToStdout`: Print path and exit (for shell integration)
  - `DropIntoShell`: Spawn a shell in that directory
- `on_cwd_visit`: What happens when user presses Ctrl-J
  - `DropIntoShell`: Spawn a shell in current directory (default)
  - `PrintToStdout`: Print path and exit (for shell integration)

**Event loop:**
1. Poll worker responses (non-blocking `try_recv`)
2. Drain logging channel (non-blocking)
3. Poll keyboard/mouse events (50ms timeout)
4. Check and log impressions if >200ms elapsed
5. Draw UI

**Keyboard:**
- Type → send UpdateQuery to worker
- Up/Down → move selection, request new visible slice if needed
- Ctrl-P/Ctrl-N → move selection up/down (same as Up/Down)
- Enter → log click, launch editor, resume TUI
- Ctrl-J → open shell in CWD (or print path and exit in shell integration mode)
- Ctrl-H → toggle history navigation mode
- Ctrl-U → clear query
- Ctrl-O → toggle debug pane
- Ctrl-F → toggle filter picker
- 0/c/d/f (when filter picker visible) → select filter (0=none, c=cwd, d=dirs, f=files)
- Ctrl-C/Ctrl-D/Esc → quit (Esc also closes filter picker or history mode if open)

**Mouse:**
- ScrollUp/ScrollDown → scroll preview pane

Why mouse scrolls preview: Keyboard for navigation (fast), mouse for exploration (natural). Most users don't use mouse for results list.

**Shell Integration:**
Run `eval "$(psychic zsh)"` in your ~/.zshrc to enable the `p` and `pd` commands:
- `p` - launch psychic, press Ctrl-J to cd your shell to the selected directory
- `pd` - launch psychic in directories-only mode (same as `psychic --shell-integration --filter=dirs`)
- Without shell integration, Ctrl-J spawns a new shell in the selected directory (old behavior)

**Filters:**
Filter picker appears as a popup overlay in the bottom-right when Ctrl-F is pressed. Four filter options:
- 0: No filter (show all matching files)
- c: Only CWD (show only files from current working directory, excludes historical files)
- d: Only directories
- f: Only files (non-directories)

The current filter is shown in the file list title with green highlighting when active (e.g., "CWD (50/200)" in green).
When no filter is active, it shows "All (200/200)" without highlighting.
Filters are applied in addition to the text query - both must match for a file to be included.

You can also set the initial filter via CLI: `psychic --filter=dirs` or `--filter=cwd` or `--filter=files`.

**History Navigation Mode:**
Pressing Ctrl-H enters history navigation mode, which provides a browser-style back/forward navigation through directories visited during the current session.

**State Management:**
- `dir_history: Vec<PathBuf>` - chronologically ordered list of visited directories
- `history_index: usize` - current position in history (where we are in the timeline)
- `history_selected: usize` - UI selection within filtered history list
- `history_mode: bool` - whether history mode is active

**Navigation behavior:**
When you navigate to a directory via Enter:
1. If selecting the directory at `history_index` (next in chronological order), increment `history_index` to preserve history
2. Otherwise, this is a branch point: truncate history at `history_index`, append current `cwd`, then set `history_index` to the new end

This creates a branch-point model similar to browser history - you can go back, then navigate to a different directory, which creates a new branch and discards the "future" history.

**UI in history mode:**
- Left pane: List of directories in reverse chronological order (most recent at top, with line numbers)
  - Includes current directory as the first entry (most recent)
  - Works even when starting fresh with no history (shows just current dir)
- Right pane: Preview of selected directory using `eza -al` (or `ls -lah` fallback)
- Bottom: Search bar filters history using case-insensitive substring matching
- Navigation: Up/Down, Ctrl-P/Ctrl-N to move selection
- Enter: Navigate to selected directory and exit history mode
- Ctrl-H or Esc: Exit history mode without navigating
- Title: "History (most recent at top) — N/M" where N is filtered count, M is total

Why reverse chronological: Most recent directories are most relevant, so they should be at the top for quick access.
Why include current dir: Allows seeing where you are in context of history, and provides consistent behavior even when history is empty.
Why substring filtering: Consistent with normal search behavior (both use `.contains()`).
Why auto-exit on Enter: Most common use case is "go back to X" - staying in history mode would require extra keypress.

**Preview:**
- Uses `bat --color=always --style=numbers --paging=never`
- `ansi-to-tui` crate converts ANSI codes to ratatui Text
- Cached by file path (invalidated on selection change)
- Scrollable with mouse wheel

Why cache: Running bat on every frame is slow. Cache entire file once, scroll offset is instant.
Why ansi-to-tui: ratatui doesn't parse ANSI codes natively. Without it, raw escape codes appear as literal text.

**Editor launch:**
```rust
disable_raw_mode()?;
terminal.backend_mut().execute(LeaveAlternateScreen)?;
Command::new("hx").arg(&full_path).status()?;
enable_raw_mode()?;
terminal.backend_mut().execute(EnterAlternateScreen)?;
terminal.clear()?;  // CRITICAL
```

Why `terminal.backend_mut().execute()`: Must use same terminal instance (not stdout()).
Why `terminal.clear()`: Wipes leftover state from editor. Without it, blank screen on resume.

**Logging:** Uses `fern` crate with dual dispatch:
- File output: `~/.local/share/psychic/app.log`
- Memory output: mpsc channel → VecDeque (circular buffer, 50 lines max)

Why dual dispatch: File for persistence, memory for debug pane. Never use eprintln (disrupts TUI).

**Scroll event deduplication:** HashSet<(query, full_path)> tracks scrolled files per session.
Why: User might scroll up/down many times. Only log once per (query, file) combination.

## Page-Based Caching

**Problem:** Scrolling results list had noticeable lag when file list was large.

**Solution:** Page-based caching with prefetch.

```rust
struct Page {
    start_index: usize,
    end_index: usize,
    files: Vec<DisplayFileInfo>,
}

struct App {
    page_cache: HashMap<usize, Page>,
    current_page: usize,
}
```

**Parameters:**
- PAGE_SIZE = 128
- PREFETCH_MARGIN = 32

**Behavior:**
- When selection enters bottom 32 items of current page, prefetch next page
- When selection enters top 32 items, prefetch previous page
- Prefetch wraps around (last page → first page)

Why: Smooth scrolling through large result sets. Prefetch prevents stuttering at page boundaries.

**Integration with worker:**
- Worker sends initial page (page 0) with QueryUpdated response
- Main thread requests additional pages via GetPage request
- Worker returns Page response with slice

Why worker sends page 0: Avoids extra round-trip. Main thread has immediate results.

## Performance Optimizations

1. **Batched file updates:** Don't call update_filtered_files() for every file from walker. Batch with flag.
   Why: 800x speedup during startup (1 rank operation vs 800).

2. **Zero-copy ranker API:** Takes `&[FileCandidate]` instead of `Vec<FileCandidate>`.
   Why: No clone at call site.

3. **File registry:** Metadata fetched once, display names computed once.
   Why: Filtering is O(n) with no syscalls or allocations.

4. **Preloaded click counts:** HashMap loaded at startup from database query with index.
   Why: O(1) lookup vs O(n) query per file.

5. **Worker thread:** All expensive operations off main thread.
   Why: UI never blocks. Can type ahead while worker processes.

6. **Debouncing:** Drain query channel, process only latest.
   Why: Fast typing = 1 query processed, not many intermediate queries.

7. **Page caching:** Request visible slice only, cache with prefetch.
   Why: Large result sets don't slow down rendering.

8. **Preview caching:** Bat runs once per file, cached as Text<'static>.
   Why: Scrolling is 60fps (no re-rendering).

## Shutdown Sequence

**Order:**
1. Drop worker_tx (signals worker to stop)
2. Join worker thread (wait for completion)
3. Drop app (closes logging channel)
4. Disable raw mode
5. Leave alternate screen
6. Disable mouse capture

Why this order: Worker can log its shutdown message before logging channel closes. Prevents "Error performing logging" messages.

## Dependencies

- `ratatui` - TUI framework
- `crossterm` - Terminal backend
- `walkdir` - Recursive directory traversal
- `rusqlite` - SQLite (bundled feature for static linking)
- `lightgbm3` - LightGBM inference
- `anyhow` - Error handling
- `jiff` - Timestamps
- `ansi-to-tui` - ANSI to ratatui Text conversion
- `clap` - CLI argument parsing
- `fern` - Logging dispatch
- `log` - Logging facade
- `once_cell` - Lazy static for feature registry
- `timeago` - Human-readable relative timestamps
- External: `bat` - Syntax highlighting (optional, has fallback)

## CLI Commands

```bash
# Run TUI
psychic

# Run with initial filter
psychic --filter=dirs      # Show only directories
psychic --filter=cwd       # Show only files in current working directory
psychic --filter=files     # Show only files (not directories)

# Configure directory click behavior
psychic --on-dir-click=navigate        # Navigate into directory (default)
psychic --on-dir-click=print-to-stdout # Print path and exit
psychic --on-dir-click=drop-into-shell # Open shell in that directory

# Configure Ctrl-J behavior (current directory visit)
psychic --on-cwd-visit=drop-into-shell # Open shell in cwd (default)
psychic --on-cwd-visit=print-to-stdout # Print path and exit (for shell integration)

# Custom data directory
psychic --data-dir=/path/to/data

# Generate training data
psychic generate-features
# Outputs: features.csv + feature_schema.json

# Retrain model (runs feature generation + training)
psychic retrain

# Output shell integration script
psychic zsh > ~/.psychic.zsh
# Then add to ~/.zshrc: source ~/.psychic.zsh

# Train model (standalone Python script)
python train.py features.csv output
# Outputs: output.txt + ~/.local/share/psychic/model.txt + output_viz.pdf
```

## Data Files

- `~/.local/share/psychic/events.db` - SQLite database
- `~/.local/share/psychic/model.txt` - LightGBM model (optional)
- `~/.local/share/psychic/model_stats.json` - Model training statistics (feature importance, accuracy, etc.)
- `~/.local/share/psychic/features.csv` - Generated training features
- `~/.local/share/psychic/feature_schema.json` - Feature definitions for Python training script
- `~/.local/share/psychic/app.log` - Application logs
- `~/.local/share/psychic/training.log` - Background model retraining output
