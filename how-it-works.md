# How It Works

## Overview

`psychic` is a terminal-based file browser with fuzzy search, file preview, and click tracking analytics. It's built in Rust using a TUI (Terminal User Interface) framework and logs user interactions to a SQLite database for analysis.

## Architecture

### Core Components

The application is split into five modules:

1. **`db.rs`** - Database layer for event logging
2. **`walker.rs`** - Background file discovery
3. **`context.rs`** - System context gathering
4. **`features.rs`** - Feature generation for machine learning
5. **`main.rs`** - TUI event loop and rendering

### Module: `db.rs`

Manages SQLite database at `~/.local/share/psychic/events.db` with two tables:

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER NOT NULL,
    query TEXT NOT NULL,
    file_path TEXT NOT NULL,           -- relative path from cwd
    full_path TEXT NOT NULL,           -- absolute path
    mtime INTEGER,                     -- file modification time (unix epoch)
    atime INTEGER,                     -- file access time (unix epoch)
    action TEXT NOT NULL,              -- 'impression' or 'click'
    session_id TEXT NOT NULL
);

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    cwd TEXT NOT NULL,
    gateway TEXT NOT NULL,
    subnet TEXT NOT NULL,
    dns TEXT NOT NULL,
    shell_history TEXT NOT NULL,
    running_processes TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
```

**Key decisions:**
- Session ID is a random 64-bit integer (not UUID - UUIDs were too long for analysis)
- Timestamps use `jiff::Timestamp::now().as_second()` for Unix epoch seconds
- Store both relative paths (for display) and full paths (for analysis)
- File metadata (mtime/atime) captured at event time, not file discovery time
- Context data gathered once per session in separate table

**API:**
- `log_session()` - Log session context data at startup
- `log_impressions()` - Batch log top 10 visible results with metadata
- `log_click()` - Log when user opens a file with metadata

### Module: `walker.rs`

Spawns a background thread that recursively walks the current directory using `walkdir`.

**Key decisions:**
- Streams results via `mpsc::channel` as files are discovered (don't wait for scan to complete)
- Filters out common ignored directories: `.git`, `node_modules`, `.venv`, `target`
- Only sends files, not directories

**Why background thread:**
- Large directories (like monorepos) can take seconds to scan
- UI remains responsive while discovery happens
- Results populate incrementally

### Module: `context.rs`

Gathers system/environment context at startup in a background thread. Captures:

- **`cwd`** - Current working directory
- **`gateway`** - Default gateway IP from `netstat -nr`
- **`subnet`** - First two octets of local IP (e.g., "192.168")
- **`dns`** - First DNS nameserver from `scutil --dns`
- **`shell_history`** - Last 10 shell commands (tries `~/.bash_history` then `~/.zsh_history`)
- **`running_processes`** - Output of `ps -u $USER -o pid,comm`

**Key decisions:**
- Runs in separate thread to avoid blocking file walker or UI startup
- All commands wrapped in shell scripts via `sh -c` for robustness
- Falls back to `"unknown"` on any error (don't crash on missing tools)
- Shell history tries both bash and zsh (zsh history needs `cut -d';' -f2-` to strip timestamps)

**Why gather this data:**
- Network context helps understand if user is at home/office/coffee shop
- Shell history provides clues about what user was doing before launching `psychic`
- Running processes show what else user is working on
- All useful features for analyzing search patterns and file access

### Module: `features.rs`

This module is responsible for reading the `events` and `sessions` tables from the database, computing a wide range of features for each impression event, and exporting the results to a CSV file suitable for training a machine learning model (e.g., with LightGBM in Python).

**Key Features Generated:**
- **Static File Features:** `path_depth`, `filename_len`, `extension`, `is_hidden`, `file_size`, `in_src`, `has_uuid`, `filename_entropy`, `path_entropy`, `digit_ratio`, `special_char_ratio`, `uppercase_ratio`, `num_underscores_in_filename`, `num_hyphens_in_filename`, etc.
- **Temporal Features:** `seconds_since_mod`, `modified_today`, `seconds_since_access`, `session_duration_seconds`.
- **Query-based Features:** `query_len`, `query_exact_match`, `filename_contains_query`, `path_contains_query`.
- **Historical Features:** `prev_session_clicks`, `prev_session_scrolls`, `ever_clicked`, `current_session_clicks`.
- **Ranking Features:** `rank`, `rank_most_recently_modified`, `rank_most_recently_clicked` of the file in the search results for a given query.
- **Label:** `1` if the file was clicked or scrolled in the same subsession, `0` otherwise.

### Module: `main.rs`

Main event loop using `ratatui` + `crossterm` for TUI rendering. Also handles command-line argument parsing to switch between TUI mode and feature generation mode.

**Layout (fzf-style):**
```
┌─────────────────┬─────────────────┐
│                 │                 │
│   File List     │    Preview      │
│   (left 50%)    │    (right 50%)  │
│                 │                 │
└─────────────────┴─────────────────┘
┌───────────────────────────────────┐
│   Search Input (bottom)           │
└───────────────────────────────────┘
```

**Key decisions:**
- Input box at bottom (like fzf, not at top)
- Split top area 50/50 for results and preview
- Preview uses `bat` for syntax highlighting

## Feature Generation Mode

To support machine learning experiments, `psychic` can be run in a non-interactive mode to generate a feature dataset from the existing event logs.

**Usage:**
```bash
# Generate features and save to features.csv (default)
psychic generate-features

# Specify a custom output path
psychic generate-features --output my_features.csv
```

This is implemented using the `clap` crate for command-line argument parsing. When the `--generate-features` flag is provided, the application calls the `features::generate_features` function and exits, skipping the TUI entirely.

## Refactoring for Testability and Performance

The `features.rs` module was initially written to query the database for each feature, for each impression. This resulted in a large number of database queries, making feature generation slow and difficult to test.

To address this, the module is being refactored to:

1.  **Fetch all data upfront:** All `events` and `sessions` are read from the database into in-memory data structures at the beginning of the feature generation process.
2.  **Compute features from in-memory data:** All feature computation functions now operate on these in-memory data structures, eliminating the need for repeated database queries.

This change has several benefits:
- **Performance:** Feature generation is significantly faster as it avoids thousands of small database queries.
- **Testability:** Unit tests can be written for the feature computation logic without needing to interact with a database. This makes the tests faster, more reliable, and easier to write.

## Key Technical Decisions

### 1. Session ID: 64-bit Random Integer

**Original plan:** Use UUID v4
**Changed to:** Random 64-bit integer as string

**Reason:** UUIDs are too long (36 characters) for a session identifier. A 64-bit int gives us 18 quintillion possible IDs, more than enough to avoid collisions, and it's much more compact in logs and queries.

**Implementation:**
```rust
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};

let mut hasher = RandomState::new().build_hasher();
Instant::now().hash(&mut hasher);
std::process::id().hash(&mut hasher);
let session_id = hasher.finish().to_string();
```

### 2. Subsession Tracking and Impression Logging

**Problem:** Every keystroke changes query → changes results → would spam database with impressions. Also need to ensure clicks/scrolls are associated with the correct query and impressions.

**Solution:** Subsession-based tracking with debounced impression logging

```rust
struct Subsession {
    id: u64,
    query: String,
    created_at: Instant,
    impressions_logged: bool,
}
```

**How it works:**
1. When query changes, a new `Subsession` is created with incremented ID
2. Every frame, `check_and_log_impressions(false)` is called - logs impressions if subsession is >200ms old and not yet logged
3. When user clicks or scrolls, `check_and_log_impressions(true)` is called first - forces immediate impression logging if not already done
4. Subsession is marked as logged to prevent duplicate logging

**Key guarantees:**
- All clicks/scrolls have their impressions logged first (temporal consistency)
- Same query always maps to same subsession_id
- Impressions are debounced by 200ms for normal typing
- Engagement events force immediate logging to ensure data completeness

**Why this matters for ML:**
- Label=1 means "user clicked/scrolled this file in this subsession"
- Features are computed from the impressions shown in that subsession
- Prevents query/subsession mismatches that would create incorrect training data

### 3. Feature Generation with Fold-Based Architecture

**Problem:** Initial implementation had O(n²) complexity - for each impression, it scanned all events to compute features. Also had future data leakage - features could see clicks that happened AFTER the impression.

**Solution:** Single-pass fold with Accumulator struct

```rust
struct Accumulator {
    clicks_by_file: HashMap<String, Vec<ClickEvent>>,
    scrolls_by_file: HashMap<String, Vec<ScrollEvent>>,
    pending_impressions: HashMap<(String, u64, String), PendingImpression>,
    output_rows: Vec<HashMap<String, String>>,
}
```

**How it works:**
1. Load all events and sessions from database into memory
2. Sort events by timestamp (critical for temporal correctness)
3. Single forward pass through events:
   - **Impression**: Compute features using only data in accumulator (past events), add to pending_impressions
   - **Click/Scroll**: Record in accumulator, mark matching pending impressions as label=1
4. After pass completes, output all pending impressions as feature rows

**Temporal correctness:**
- Features like `clicks_last_30_days` filter with `c.timestamp <= impression.timestamp`
- Only past events contribute to features
- Labels CAN use future events (click happens after impression in same subsession)
- This matches production: at inference time, model only sees past data

**Performance:** O(n) instead of O(n²) - single pass through events

### 4. Editor Launch and TUI Suspend/Resume

**Challenge:** Need to suspend TUI, launch `hx` editor, then resume TUI when editor closes.

**Gotcha #1:** Terminal state wasn't properly restored after editor closed - blank screen on resume.

**Solution:**
```rust
// Suspend
disable_raw_mode()?;
terminal.backend_mut().execute(LeaveAlternateScreen)?;

// Launch editor
std::process::Command::new("hx").arg(&full_path).status();

// Resume
enable_raw_mode()?;
terminal.backend_mut().execute(EnterAlternateScreen)?;
terminal.clear()?;  // CRITICAL: must clear terminal
```

**Key insight:** Must use `terminal.backend_mut().execute()` (not `stdout().execute()`) to ensure we're using the same terminal instance. The `terminal.clear()` call is essential to wipe any leftover state.

### 4. File Preview with Bat

**Goal:** Show syntax-highlighted preview of selected file

**First attempt:** Just read file contents and display as plain text
```rust
std::fs::read_to_string(&path)
```

**Problem:** No syntax highlighting, looks bland

**Second attempt:** Use `bat --color=always` and display output
```rust
Command::new("bat")
    .arg("--color=always")
    .arg("--style=plain")
    .arg("--line-range").arg(":100")
    .output()
```

**Gotcha #2:** Terminal rendering was completely broken - saw raw ANSI escape codes like `^[[38;5;123m` mixed with colored text. Also saw "weird remnants of previous previews" when scrolling.

**Root cause:** `bat` outputs ANSI escape codes for colors (e.g., `\x1b[31m` for red), but `ratatui`'s `Paragraph` widget doesn't parse these - it renders them literally. Need to convert ANSI codes to `ratatui::text::Text` with proper styling.

**Solution:** Use `ansi-to-tui` crate
```rust
use ratatui::text::Text;

match Command::new("bat")...output() {
    Ok(output) => {
        match ansi_to_tui::IntoText::into_text(&output.stdout) {
            Ok(text) => text,
            Err(_) => Text::from("[Unable to parse preview]"),
        }
    }
}
```

**Key insight:** `ansi-to-tui::IntoText` expects `&[u8]` (byte slice), which is exactly what `output.stdout` is. The library handles UTF-8 conversion internally, so we don't need `from_utf8_lossy`. Just pass the raw bytes directly.

**Why this works:**
- `bat` outputs ANSI escape codes to `stdout`
- `ansi-to-tui` parses these codes and converts them to `ratatui::text::Text` with proper `Style` applied to each span
- `ratatui` renders the `Text` with colors intact
- Terminal clears properly on each redraw, no remnants

**Fallback:** If `bat` isn't installed, falls back to plain `std::fs::read_to_string`.

## Event Loop

Main loop does three things:

1. **Receive files from walker** (non-blocking)
   ```rust
   while let Ok(path) = rx.try_recv() {
       app.files.push(path);
       app.update_filtered_files();
   }
   ```

2. **Draw UI** using `ratatui`
   - File list (left)
   - Preview (right)
   - Search input (bottom)

3. **Handle keyboard input**
   - Type characters → update query → filter files → log impressions
   - Up/Down → move selection → updates preview
   - Enter → log click, launch editor, resume TUI
   - Ctrl-C or Esc → quit

## Search and Filtering

**Algorithm:** Case-insensitive substring match

```rust
let query_lower = self.query.to_lowercase();
self.files.iter().filter(|path| {
    relative_path.to_lowercase().contains(&query_lower)
})
```

**No fuzzy matching yet** - just plain substring. Could add fuzzy matching later (e.g., using `fuzzy-matcher` crate) but kept it simple for MVP.

## Dependencies

- `ratatui` - TUI framework
- `crossterm` - Terminal backend (handles raw mode, key events)
- `walkdir` - Recursive directory traversal
- `rusqlite` - SQLite database (with `bundled` feature for static linking)
- `anyhow` - Error handling
- `jiff` - Timestamps
- `ansi-to-tui` - Convert ANSI escape codes to ratatui Text
- `clap` - Command-line argument parsing
- External: `bat` - Syntax highlighting (optional, has fallback)

## Gotchas and Lessons Learned

### 1. TUI Suspend/Resume is Tricky
Must properly cleanup terminal state before launching child process, then restore. The `terminal.clear()` call is non-obvious but critical.

### 2. ANSI Escape Codes Aren't Automatically Parsed
`ratatui` doesn't magically handle ANSI codes - need explicit parsing with `ansi-to-tui`. Raw escape codes will render as literal text otherwise.

### 3. Don't Double-Convert Bytes
`ansi-to-tui` works on `&[u8]` and handles UTF-8 internally. Don't call `from_utf8_lossy` first - you'll lose information and create unnecessary allocations.

### 4. Background File Walker Needs Channel
Can't just block and wait for file scan to complete - large directories would freeze UI. Use `mpsc::channel` to stream results as they're discovered.

### 5. Debounce Impression Logging
Without debouncing, every keystroke would write to database. 200ms debounce batches impressions sensibly.

### 6. Cargo Edition "2024" Doesn't Exist
Initial `Cargo.toml` had `edition = "2024"` (probably from spec) but latest stable is `edition = "2021"`. Build would fail otherwise.

### 7. Preview Scroll Lag and Blank Lines

**Problem:** When scrolling preview, text below scroll position showed as blank. Also scrolling felt laggy/slow.

**Root causes:**
1. `bat --line-range :100` only fetched first 100 lines. Scrolling past line 100 showed blank because no data existed.
2. Running `bat` on every frame (60+ times per second) was extremely slow. Each scroll triggered full re-render with new `bat` call.

**Solution:** Preview caching
- Fetch entire file once with `bat --paging=never` (no line limit)
- Cache rendered `Text<'static>` in `App` state keyed by file path
- On scroll, just adjust `Paragraph::scroll()` offset - no re-rendering needed
- Invalidate cache when selection changes to different file

**Result:** Smooth 60fps scrolling through arbitrarily long files. No more blank lines, no more lag.

## Enhanced Context Tracking (Added)

### What Changed

Added comprehensive context tracking to understand the environment in which searches happen:

1. **New `sessions` table** - Stores per-session context data
2. **New `context.rs` module** - Gathers system/network/shell data
3. **Enhanced `events` table** - Now includes full paths and file timestamps
4. **Background context gathering** - Runs in separate thread at startup

### Data Collected

**Session-level (once per run):**
- Working directory
- Network gateway/subnet/DNS (helps identify location)
- Last 10 shell commands (what user was doing before)
- Running processes (what else is open)

**Event-level (per impression/click):**
- Both relative and absolute file paths
- File modification time (mtime)
- File access time (atime)

### Implementation Details

**File metadata timing:** Metadata (mtime/atime) is captured at event time, not discovery time. This is important because:
- File could be modified between discovery and impression/click
- Capturing at event time gives accurate "file state when user saw it"
- Slight performance cost but more accurate data

**Context gathering is asynchronous:**
```rust
std::thread::spawn(move || {
    let context = context::gather_context();
    if let Ok(db) = db::Database::new() {
        let _ = db.log_session(&session_id_clone, &context);
    }
});
```

Spawns immediately at startup but doesn't block file walker or UI. Context commands can take 100-500ms combined (especially `ps` and network commands), so doing this async keeps startup snappy.

**Shell script wrapping:** All system commands run via `sh -c` rather than direct `Command::new()`. This handles:
- Piping (e.g., `netstat | grep | awk`)
- Shell globbing
- Command chaining

**Error handling:** Every context field falls back to `"unknown"` on error. Never crash the app because of missing/unavailable system tools.

### Why This Data?

**Network context** (gateway/subnet/DNS) helps answer:
- Is user at home, office, or coffee shop?
- Different networks might correlate with different file access patterns
- Could reveal patterns like "always searches for config files when on public WiFi"

**Shell history** reveals:
- What commands user ran before launching `psychic`
- Possible intent (e.g., ran `git log` → likely searching for commit-related files)
- Workflow patterns

**Running processes** shows:
- What applications are open (IDEs, browsers, terminals)
- Multi-tasking behavior
- Could correlate with file types accessed

**File timestamps** enable:
- "User tends to click recently modified files"
- "Files accessed recently are clicked more"
- Time-based relevance signals

All of this enables richer analysis and potentially ML-based ranking in the future.

## UI Enhancements (Added)

### What Changed

Added several UX improvements to make the TUI more informative and interactive:

1. **Rank numbers** - Each file in results list shows its position (1., 2., 3., etc.)
2. **Time ago column** - Shows how long ago each file was modified (e.g., "2 hours ago", "3 days ago")
3. **Preview scrolling** - Mouse wheel scrolls the preview pane instead of results list
4. **Scroll event tracking** - Logs "scroll" action when user scrolls preview

### Implementation Details

**Rank numbers:** Simple enumeration in the file list rendering:
```rust
let rank = i + 1;
let line = format!("{:2}. {:<50} {}", rank, file, time_ago);
```

**Time ago formatting:** Uses `timeago` crate for human-readable relative timestamps:
```rust
let formatter = timeago::Formatter::new();
formatter.convert(duration)
```

Produces output like "just now", "5 minutes ago", "2 days ago", "3 weeks ago", etc. Much more readable than unix timestamps or ISO dates.

Used in multiple places:
- Historical files list (showing when files were last accessed)
- Debug pane: Model trained timestamp
- Debug pane: Model load time ("Model load: 2 minutes ago")
- Debug pane: Clicks reload time ("Clicks reload: 5 seconds ago")

**In-memory logging for debug pane:** Uses `fern` crate to dispatch logs to multiple outputs:
```rust
let (log_tx, log_rx) = mpsc::channel();

fern::Dispatch::new()
    .format(|out, message, record| {
        out.finish(format_args!(
            "[{} {} {}] {}",
            timestamp, record.level(), record.target(), message
        ))
    })
    .level(log::LevelFilter::Debug)
    .chain(fern::log_file(log_file)?)  // File output
    .chain(log_tx)  // Memory output via channel
    .apply()?;
```

Recent logs are collected in a `VecDeque<String>` with max 50 entries (circular buffer):
- Event loop drains `log_rx` channel (non-blocking)
- Pushes messages to `app.recent_logs`
- Pops oldest when over limit
- Debug pane shows last 10 lines (30 when maximized), truncated to 60 chars (120 when maximized)

**Debug pane maximization:** Press `Ctrl-O` to toggle:
- Normal view: 35% file list, 45% preview, 20% debug
- Maximized view: 25% file list, 0% preview (hidden), 75% debug
- Shows more logs and longer lines when maximized
- Title updates to show current state and keybinding

**Preview scrolling:**
- Added `preview_scroll: u16` to `App` state
- `Paragraph::scroll((app.preview_scroll, 0))` controls vertical offset
- Mouse events (ScrollUp/ScrollDown) adjust scroll position by 3 lines
- Scroll position resets to 0 when selection changes (different file = start at top)

**Preview caching (performance fix):**
- Added `preview_cache: Option<(String, Text<'static>)>` to cache rendered preview
- Fetch entire file once with `bat --color=always --style=numbers --paging=never`
- `--style=numbers` shows line numbers in preview
- Cache keyed by full file path
- Cache invalidated when selection changes
- Eliminates lag - `bat` only runs once per file, not on every frame
- Fixes blank text issue - entire file is cached, not just first 100 lines

**Mouse capture:**
```rust
stdout().execute(crossterm::event::EnableMouseCapture)?;
```

Must enable at startup and disable on cleanup. Also disable/enable when launching editor to avoid interfering with editor's mouse handling.

**Scroll event logging:** New "scroll" action type in events table. Indicates user interest without full click commitment. Useful for distinguishing:
- **Impression** - File appeared in top 10
- **Scroll** - User scrolled preview (definitely looked at it)
- **Click** - User opened file in editor (highest intent)

This creates a hierarchy of engagement levels for later analysis.

**Scroll event deduplication:**
- Track scrolled files in `HashSet<(String, String)>` - key is `(query, full_path)`
- Only log scroll once per session for each unique (query, file) combination
- Prevents spam: user might scroll up/down many times on same file
- Check `scrolled_files.contains(&key)` before logging
- Per-session deduplication (not global) - resets each time app starts

### Why Mouse Scroll for Preview?

Original behavior: mouse scroll moved selection up/down in results list.

**Problem:** Most users navigate results with keyboard (up/down arrows are faster). Mouse scroll on results list is redundant and not commonly used.

**Better use:** Scroll the preview pane. Users want to see more of the file before deciding to open it. Preview was limited to first ~100 lines with no way to see more. Now mouse wheel scrolls through the file content.

**Result:** Much more useful interaction. Keyboard for navigation (fast), mouse for exploration (natural).

### Clippy and Release Builds

**Added policy:** Always run `cargo clippy --release -- -D warnings` before considering a build done. Treats all warnings as errors for code quality.

**Clippy fixes made:**
1. Removed unused imports (`Scrollbar`, `ScrollbarOrientation`, `ScrollbarState`) that were added but not used
2. Fixed "too many arguments" warning by introducing `EventData` struct to bundle event parameters

**Release mode:** All builds now `--release` for production-ready performance. Debug builds are slow, especially for TUI rendering.

## ML-Powered Ranking with LightGBM (Added)

### Overview

Integrated a **LightGBM LambdaRank** model for learning-to-rank file search results based on user interactions. The ranker scores files using features like query matching, click history, and modification time, then re-orders results to show most relevant files first.

### Architecture

**New components:**
1. **`ranker.rs`** - LightGBM model inference and feature computation
2. **`train.py`** - Python script for training ranking model
3. **Model storage** - `~/.local/share/psychic/model.txt` (next to events.db)
4. **3-column TUI layout** - Added ML features display panel

### Model: `ranker.rs`

**Purpose:** Load trained LightGBM model and score files in real-time.

```rust
pub struct Ranker {
    model: Booster,
    click_counts: HashMap<String, usize>, // Preloaded click data
}
```

**Key features computed:**
- `filename_starts_with_query` (binary 0/1)
- `clicks_last_30_days` (integer count)
- `modified_today` (binary 0/1)

**Performance optimization:**
- Preloads ALL click data from last 30 days at startup into HashMap
- O(1) lookup per file vs O(n) database query per file
- Database query runs once: `SELECT full_path, COUNT(*) FROM events WHERE action='click' AND timestamp >= ?1 GROUP BY full_path`
- Composite index on `(action, timestamp, full_path)` speeds up query

**Initialization:**
```rust
let ranker = Ranker::new(&model_path, db_path)?;
// Loads model from ~/.local/share/psychic/model.txt
// Preloads all click counts into memory
```

**Ranking process:**
```rust
let scored = ranker.rank_files(&query, &matching_files, &session_id)?;
// Returns Vec<FileScore> sorted by predicted relevance (descending)
```

**FileScore struct:**
```rust
pub struct FileScore {
    pub path: String,
    pub score: f64,
    pub features: HashMap<String, f64>,
}
```

### Training Script: `train.py`

**Purpose:** Train LightGBM ranking model from feature CSV exported by `psychic generate-features`.

**Key characteristics:**
- **Objective:** `lambdarank` (learning-to-rank)
- **Metric:** NDCG (Normalized Discounted Cumulative Gain)
- **Grouping:** By subsession_id (each search query = one ranking problem)
- **Features:** Only numeric features (Rust lightgbm3 doesn't support categorical)

**Usage:**
```bash
psychic generate-features  # Export features.csv
python train.py features.csv output  # Train model
# Outputs:
#   - output.txt (model for current directory)
#   - ~/.local/share/psychic/model.txt (model for TUI)
#   - output_viz.pdf (feature importance, SHAP analysis, NDCG metrics)
```

**Model saves to two locations:**
1. `output.txt` in current directory (for version control/inspection)
2. `~/.local/share/psychic/model.txt` (where TUI looks for it)

**Visualizations generated:**
- Training curves (NDCG@5 over iterations)
- Feature importance (gain, split count, correlation)
- SHAP summary plots (mean |SHAP value| and impact distribution)
- SHAP dependence plots for top 4 features
- Score distributions (clicked vs not clicked)
- Rank position analysis (where do clicked items appear?)

**Categorical features removed:**
- Original features included `query` and `file_path` as categorical
- Rust `lightgbm3` crate only supports numeric features
- Removed from training data: `df.drop(columns=["query", "file_path", ...])`
- Can add back later with manual mapping to numeric codes

### UI Changes: 3-Column Layout

**New layout:**
```
┌────────────────┬──────────────────┬─────────────────┐
│  File List     │   Preview        │  ML Features    │
│  (35%)         │   (45%)          │  (20%)          │
│                │                  │                 │
│ 1. file.rs     │  [bat output]    │ Score: 0.7234   │
│ 2. main.rs     │                  │                 │
│ ...            │                  │ Features:       │
│                │                  │  Query Match: 1 │
│                │                  │  Clicks (30d): 3│
│                │                  │  Modified Today:│
└────────────────┴──────────────────┴─────────────────┘
┌───────────────────────────────────────────────────────┐
│   Search Input (bottom)                               │
└───────────────────────────────────────────────────────┘
```

**Features panel shows:**
- **Score** - LightGBM predicted relevance score
- **Features** - All computed features with readable labels
- Shows "No features (ranking enabled)" when model loaded but no file selected
- Shows "No features (ranking disabled)" when no model found

**Fallback behavior:**
- If `~/.local/share/psychic/model.txt` doesn't exist → simple substring filtering
- Logs: `"No ranking model found at {:?} - using simple filtering"`
- If model load fails → logs warning and falls back to filtering
- Application always works, ranking is optional enhancement

### Performance Improvements

**Problem:** Startup was slow with many files due to calling `update_filtered_files()` for every single file received from walker.

**Root cause:**
```rust
// BAD: Called 816 times, 817 times, 818 times...
while let Ok(path) = rx.try_recv() {
    app.files.push(path);
    app.update_filtered_files();  // ← Re-filters and re-ranks ALL files!
}
```

Each call:
1. Filters all files (with mtime lookup for each)
2. Ranks all matching files with ML model
3. Sorts results

With 800 files, this is 800 filter+rank+sort operations instead of 1.

**Solution:** Batch file updates
```rust
let mut received_files = false;
while let Ok(path) = rx.try_recv() {
    app.files.push(path);
    received_files = true;
}

// Only update once after receiving all available files
if received_files {
    app.update_filtered_files();
}
```

**Result:** Dramatic startup speedup. Instead of hundreds of rank operations, only 1 per event loop iteration.

**Timing instrumentation added:**
- App::new() timing (database init, ranker init)
- update_filtered_files() timing (filtering, ranking)
- All logs go to `~/.local/share/psychic/app.log` (not stderr, which disrupts TUI)

### Logging Infrastructure

**Problem:** Using `eprintln!` for debugging made terminal display "all weird" because it interferes with TUI rendering.

**Solution:** File-based logging with `log` + `env_logger` crates

**Setup:**
```rust
// Initialize at start of main()
let log_file = PathBuf::from(home)
    .join(".local/share/sg/app.log");

env_logger::Builder::new()
    .target(env_logger::Target::Pipe(Box::new(file)))
    .filter_level(log::LevelFilter::Debug)
    .init();
```

**All eprintln! replaced with log macros:**
- `log::info!()` - Model loading, startup messages
- `log::warn!()` - Ranking failures, fallbacks
- `log::error!()` - Editor launch failures
- `log::debug!()` - Timing data, performance metrics

**Logs written to:** `~/.local/share/psychic/app.log`

**Example log output:**
```
[2025-10-07T02:08:35Z INFO  psychic] Started sg in directory /path/to/project, session 12345
[2025-10-07T02:08:35Z DEBUG psychic] Database initialization took 2.3ms
[2025-10-07T02:08:35Z DEBUG psychic] Loaded 143 click counts from last 30 days
[2025-10-07T02:08:35Z DEBUG psychic] Ranker initialization took 45.2ms
[2025-10-07T02:08:35Z DEBUG psychic] Filtering 820 files (with metadata) took 1.4ms
[2025-10-07T02:08:35Z DEBUG psychic] Ranking took 2.2ms
```

### Database Enhancements

**New method:**
```rust
pub fn get_previously_interacted_files(&self) -> Result<Vec<String>> {
    // Get all unique full_paths that have been clicked or scrolled
    let mut stmt = self.conn.prepare(
        "SELECT DISTINCT full_path
         FROM events
         WHERE action IN ('click', 'scroll')
         ORDER BY timestamp DESC"
    )?;
    // ...
}
```

**Purpose:** Include historical files in search results even if not found by current directory walker.

**Use case:**
- User previously clicked `config/database.yml` in different project
- Current project doesn't have that file
- But similar file path might exist and should be surfaced
- Useful for cross-project patterns

**Implementation:**
- Load historical files at startup
- Store in `App.historical_files: Vec<PathBuf>`
- Filter out files that don't exist (check `path.exists()`)
- Include in search results if they match query and aren't already in results
- Merge with walkdir results before ranking

### UI/UX Improvements

**1. Keyboard shortcut: Clear search**
- **Key:** Ctrl-U
- **Action:** Clears entire search query
- **Standard:** Follows readline convention (Ctrl-U = kill line backward)

**2. Auto-scroll for file list**
- **Problem:** When navigating down, selected item would go off-screen
- **Solution:** Automatic scrolling when selection near bottom
- **Implementation:**
  - Track `file_list_scroll: u16` offset
  - `update_scroll(visible_height)` adjusts scroll position
  - When selected item in bottom 5 rows → scroll to keep it visible
  - Skip/take on file list rendering based on scroll offset

**3. Right-justified modification time**
- **Problem:** Fixed-width formatting caused misaligned time column
- **Solution:** Dynamic width calculation based on actual terminal size
```rust
let list_width = top_chunks[0].width.saturating_sub(2); // subtract borders
let available = list_width.saturating_sub(prefix_len + time_len);
let file_width = available.saturating_sub(2); // padding
format!("{}{:<width$}  {}", rank_prefix, file, time_ago, width = file_width)
```
- **Result:** Time column always right-aligned regardless of filename length

**4. Historical files in results**
- Previously clicked/scrolled files included in search
- Expands search beyond current directory
- Useful for finding files you accessed before but aren't in current tree

### Performance: Zero-Copy Optimizations

**Problem:** Unnecessary clones in hot path

**Ranker signature change:**
```rust
// BEFORE: Takes owned Vec (requires clone at call site)
pub fn rank_files(&self, files: Vec<(String, PathBuf, Option<i64>)>) -> Result<Vec<FileScore>>

// AFTER: Takes slice reference (zero-copy)
pub fn rank_files(&self, files: &[(String, PathBuf, Option<i64>)]) -> Result<Vec<FileScore>>
```

**Call site improvement:**
```rust
// BEFORE: Clone matching_files just to pass to ranker
ranker.rank_files(&self.query, matching_files.clone(), &self.session_id)

// AFTER: Pass reference, no clone
ranker.rank_files(&self.query, &matching_files, &self.session_id)
```

**Also removed:**
- Clone of features HashMap when creating FileScore (move instead)
- Clone of scored results when updating filtered_files

**Result:** Reduced memory allocations in ranking hot path.

### Database Indexes

**Added index for click count queries:**
```sql
CREATE INDEX IF NOT EXISTS idx_events_click_lookup
ON events(action, timestamp, full_path)
```

**Covers query:**
```sql
SELECT full_path, COUNT(*) as count
FROM events
WHERE action = 'click' AND timestamp >= ?1
GROUP BY full_path
```

**Performance:** O(log n) index lookup vs O(n) table scan for 30-day click aggregation.

## LambdaMART Group-Based Ranking (Added)

### What Changed

Modified the grouping strategy for LambdaMART training from subsession-based to engagement-based sequences:

**Previous behavior:**
- Groups were based on `subsession_id` (one group per query/search)
- Each unique query within a session created a new group
- All impressions for that query belonged to the same group

**New behavior:**
- Groups span from one engagement event (click/scroll) to the next
- `group_id` starts at 0 and increments on:
  - Each click or scroll event
  - Each new session boundary
- Creates more meaningful ranking tasks where each group represents impressions leading to an action

### Implementation Details

**features.rs changes:**
1. Added `current_group_id: u64` to `Accumulator` struct
2. Added `last_session_id: Option<String>` to track session boundaries
3. `group_id` field added to CSV output
4. In `add_impression()`: checks for session changes and assigns current `group_id`
5. In `mark_impressions_as_engaged()`: increments `group_id` after marking labels
6. New feature: `is_under_cwd` - binary feature indicating if file is under the session's working directory

**train.py changes:**
- Changed from `subsession_id` to `group_id` for grouping
- Updated to exclude `group_id` from features (used only for grouping)
- Added `is_under_cwd` to binary features list

**Why this approach:**
- More granular ranking tasks - each group represents a coherent search-to-action sequence
- Better aligns with user behavior - impressions before an action vs after
- Reduces group size variance - subsessions could have very different lengths
- Session boundaries create natural breaks in the data

**Example:**
```
Session 1:
  Impression A, B, C → Click on B  [group_id: 0]
  Impression D, E → Scroll on E    [group_id: 1]
  Impression F, G, H              [group_id: 2]
Session 2:
  Impression I, J → Click on J     [group_id: 3]
```

Each group becomes a learning-to-rank problem where the model learns which files should be ranked higher.

## Summary of Session Work

This session added significant ML and performance enhancements:

**ML Infrastructure:**
1. LightGBM ranking model integration (`ranker.rs`)
2. Model training pipeline (`train.py` improvements)
3. 3-column TUI layout with ML features display
4. Preloaded click data for O(1) feature computation
5. Database index for efficient click aggregation

**Performance:**
1. Batched file updates (800x speedup during startup)
2. Zero-copy ranker API (slice references)
3. File-based logging (no TUI disruption)
4. Timing instrumentation for bottleneck analysis

**UX:**
1. Ctrl-U to clear search
2. Auto-scroll in file list
3. Right-justified time column
4. Historical files in search results

**Infrastructure:**
1. Model storage in `~/.local/share/psychic/`
2. Log output to `app.log`
3. Startup logging with session/directory info
4. Comprehensive debug timing

All improvements maintain backward compatibility - app works without model (falls back to simple filtering).

## Trait-Based Feature System Refactor (Added)

### Problem

Adding new features required updating 7+ places manually:
- Feature computation in features.rs
- Feature computation in ranker.rs
- Feature name ordering in multiple places
- CSV column order
- Python feature lists
- Binary vs numeric type tracking

This was brittle and error-prone.

### Solution: Single Source of Truth

Created a trait-based feature registry system where **all features are defined once** in `src/feature_defs/`:

**Core architecture:**
```rust
// schema.rs - trait definition
pub trait Feature: Send + Sync {
    fn name(&self) -> &'static str;
    fn feature_type(&self) -> FeatureType;
    fn compute(&self, inputs: &FeatureInputs) -> Result<f64>;
}

// implementations.rs - all features in one file
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

**Key benefits:**
- **Single implementation** - Each feature has one `compute()` method used everywhere
- **Automatic ordering** - Registry order determines feature vector order (no manual synchronization)
- **Type safety** - Features export their type (binary/numeric) automatically
- **Schema export** - JSON schema generated directly from registry
- **Easy to add features** - Implement trait, add to registry, done

**Cross-language integration:**
```bash
# Rust exports schema
cargo run --release -- generate-features
# Outputs: features.csv + feature_schema.json

# Python reads schema
python train.py features.csv output
# Reads feature_schema.json to get feature names and types
```

**Schema format (simplified):**
```json
{
  "features": [
    {"name": "filename_starts_with_query", "type": "binary"},
    {"name": "clicks_last_30_days", "type": "numeric"},
    {"name": "is_hidden", "type": "binary"}
  ]
}
```

**Files changed:**
- `src/feature_defs/` - New module with schema, implementations, registry
- `src/features.rs` - Now uses registry for computation
- `src/ranker.rs` - Now uses registry for inference
- `src/main.rs` - CLI changed to `generate-features` subcommand, auto-generates schema
- `train.py` - Reads feature_schema.json (errors if missing)
- `Cargo.toml` - Added once_cell dependency

**New features added using this system:**
- `is_under_cwd` - File is under current working directory (binary)
- `is_hidden` - Path contains hidden directory component (binary)

### Renaming: sg → psychic

Changed binary and directory names:
- Binary: `sg` → `psychic` (in Cargo.toml)
- Data directory: `~/.local/share/sg/` → `~/.local/share/psychic/`
- Updated: db.rs, main.rs, train.py, how-it-works.md
- Log file: `~/.local/share/psychic/app.log`
- Database: `~/.local/share/psychic/events.db`
- Model: `~/.local/share/psychic/model.txt`

### Performance Limit

Added MAX_FILES = 5000 limit in walker.rs to cap file discovery for now. Will optimize later.

## Future Improvements

Potential enhancements (not implemented):

- Fuzzy matching (not just substring)
- Configurable ignored directories
- Custom editor (currently hardcoded to `hx`)
- Better binary file handling (currently tries to preview everything)
- More ranking features (file size, directory depth, extension patterns)
- Online learning (update model from user interactions)
- Categorical features in ranking model (need manual encoding for Rust lightgbm3)
- Cross-session feature aggregation (global click patterns, not just session-local)
- A/B testing framework for ranking experiments
