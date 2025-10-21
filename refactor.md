# Refactor Plan: Extract Deep Modules from main.rs

**Original state:** main.rs was 2,866 lines with too many concerns mixed together.

**Current progress:**
- ✅ Phase 1: path_display.rs (271 lines) and cli.rs (116 lines) extracted
- ✅ Phase 2: app.rs (556 lines) extracted
- ✅ Phase 3: render.rs (155 lines) extracted
- **main.rs reduced from 2,866 → 1,782 lines (1,084 lines extracted, 38% reduction)**

**All tests passing:** 35/35 ✅

**Deep Modules Strategy** (per John Ousterhout):
- **Small interface** = minimal public API, few functions/types exposed
- **Deep implementation** = lots of complexity hidden inside
- Goal: Each module owns a complete subsystem with simple boundaries

---

## Proposed Module Extractions

### 1. **`src/preview.rs`** - Preview Generation & Caching Module

**Interface (small):**
```rust
pub struct PreviewManager {
    cache: PreviewCache,
    scroll_offset: usize,
}

impl PreviewManager {
    pub fn new() -> Self;
    pub fn render(&mut self, path: &Path, is_dir: bool, width: u16) -> Text<'static>;
    pub fn scroll(&mut self, delta: isize);
    pub fn reset_scroll(&mut self);
}
```

**Deep implementation inside:**
- `PreviewCache` enum (Light/Full/Directory)
- bat command execution + ANSI parsing
- eza command execution for directories
- Width-based flag selection
- Scroll offset management
- Cache invalidation logic
- Timing instrumentation

**Why deep:** Preview has complex logic (caching, bat vs eza, width adaptation, ANSI conversion, scrolling) but simple interface (just render, scroll, reset).

---

### 2. **`src/render.rs`** - UI Rendering Module ✅ PARTIALLY COMPLETED

**Interface (actual):**
```rust
pub fn render_history_mode(f: &mut Frame, app: &App);
```

**Deep implementation inside:**
- History mode layout logic (dir list + preview split)
- Directory list rendering with rank numbers
- Path truncation for display (uses path_display module)
- Eza command execution for directory previews
- Fallback to ls if eza unavailable
- ANSI-to-TUI conversion
- Timing instrumentation
- Search input rendering

**Why deep:** History rendering has complex layout, command execution, ANSI parsing, but simple interface (just render).

**Status:** ✅ Extracted `render_history_mode` to `src/render.rs` (155 lines)

**Note:** Normal mode rendering remains in main.rs as it's tightly coupled with preview cache mutation. Could be extracted in future with further refactoring.

---

### 3. **`src/path_display.rs`** - Path Formatting Module ✅ COMPLETED

**Interface (small):**
```rust
pub fn truncate_path(path: &str, max_len: usize) -> String;
pub fn truncate_absolute_path(path: &str, max_len: usize) -> String;
pub fn get_time_ago(mtime: Option<i64>) -> String;
```

**Deep implementation inside:**
- Path component parsing
- Smart truncation (keep first + last components)
- Abbreviation logic
- Timeago formatting

**Why deep:** Path formatting has tricky edge cases but ultra-simple interface (string in, string out).

**Status:** ✅ Extracted to `src/path_display.rs` (271 lines including 12 tests)

---

### 4. **`src/cli.rs`** - CLI Argument Parsing Module ✅ COMPLETED

**Interface (small):**
```rust
pub struct Cli { /* clap-derived fields */ }
pub enum Commands { GenerateFeatures, Retrain, Zsh, Internal }
pub enum InternalCommands { AnalyzePerf }
pub fn get_default_data_dir() -> Result<PathBuf>;
```

**Deep implementation inside:**
- All clap structs (Cli, Commands, InternalCommands, etc.)
- Argument validation
- Default value logic
- Enum conversions (FilterArg, OnDirClickAction, etc.)

**Why deep:** CLI parsing has many options but interface is just "give me a command".

**Status:** ✅ Extracted to `src/cli.rs` (116 lines)

---

### 5. **`src/app.rs`** - Application State Module ✅ COMPLETED

**Interface (public API):**
```rust
pub struct App { /* 25+ public fields */ }
pub struct Page { pub start_index, pub end_index, pub files }
pub struct Subsession { pub id, pub query, pub created_at, ... }
pub enum PreviewCache { None, Light, Full, Directory }
pub const PAGE_SIZE: usize = 128;
pub const PREFETCH_MARGIN: usize = 32;

impl App {
    pub fn new(...) -> Result<Self>;
    pub fn reload_model(&mut self, query_id: u64) -> Result<()>;
    pub fn reload_and_rerank(&mut self, query_id: u64) -> Result<()>;
    pub fn get_file_at_index(&self, index: usize) -> Option<&DisplayFileInfo>;
    pub fn check_and_log_impressions(&mut self, force: bool) -> Result<()>;
    pub fn move_selection(&mut self, delta: isize);
    pub fn get_filtered_history(&self) -> Vec<PathBuf>;
    pub fn move_history_selection(&mut self, delta: isize);
    pub fn handle_history_enter(&mut self) -> Result<()>;
    pub fn log_preview_scroll(&mut self) -> Result<()>;
    pub fn update_scroll(&mut self, visible_height: u16);
}

impl PreviewCache {
    pub fn get_if_matches(&self, path: &str) -> Option<(Text<'static>, bool)>;
    pub fn get_dir_if_matches(&self, path: &str, extra_flags: &str) -> Option<Text<'static>>;
}
```

**Deep implementation inside:**
- All App state (query, page_cache, selected_index, preview_cache, etc.)
- Worker communication (worker_tx)
- Page cache management with prefetching logic
- Analytics logging (impressions, scrolls, clicks)
- Selection movement with wrap-around
- History navigation
- Preview cache management (Light/Full/Directory variants)
- Subsession tracking

**Why deep:** App has tons of state and complex prefetching/caching logic, but provides clear methods for each operation.

**Status:** ✅ Extracted to `src/app.rs` (556 lines including Page, Subsession, PreviewCache)

---

### 6. **`src/analytics.rs`** - Analytics/Logging Module

**Interface (small):**
```rust
pub struct Analytics {
    db: Database,
    session_id: String,
    current_subsession: Option<Subsession>,
}

impl Analytics {
    pub fn new(db: Database, session_id: String) -> Self;
    pub fn log_impression(&mut self, query: &str, files: &[FileInfo]) -> Result<()>;
    pub fn log_click(&mut self, query: &str, file: &FileInfo) -> Result<()>;
    pub fn log_scroll(&mut self, query: &str, file: &FileInfo) -> Result<()>;
}
```

**Deep implementation inside:**
- Subsession tracking
- Impression debouncing (200ms logic)
- Scroll deduplication (HashSet tracking)
- Event data formatting
- Database writing

**Why deep:** Analytics has complex timing/deduplication but simple interface (just log events).

---

## Migration Strategy

**Phase 1:** Extract independent utilities (no App dependencies)
1. `path_display.rs` - just string functions
2. `cli.rs` - just argument parsing

**Phase 2:** Extract rendering (depends on state but no mutation)
3. `preview.rs` - move PreviewCache + generation logic
4. `render.rs` - move all rendering functions

**Phase 3:** Extract state management
5. `analytics.rs` - extract logging from App
6. `app.rs` - refactor App to use extracted modules

**Phase 4:** Final cleanup
- main.rs becomes tiny: just `main()` + `run_app()` glue code
- Each module tested independently
- Update how-it-works.md

---

## Expected Result

**main.rs after refactor:** ~300-400 lines
- `main()` function
- `run_app()` event loop
- Module imports
- Glue code between modules

**Benefits:**
- Each module is independently testable
- Clear ownership of concerns
- Easy to find code ("preview issues? check preview.rs")
- Reduced cognitive load (work on one module at a time)
- Follows Ousterhout's philosophy: simple interfaces hiding complexity

---

## Current Modules (for reference)

Existing modules that already follow this pattern:
- `walker.rs` - Background file discovery (simple interface: send paths, receive WalkerMessage)
- `search_worker.rs` - Async filtering/ranking (simple interface: send query, receive results)
- `ranker.rs` - ML model inference (simple interface: rank_files())
- `db.rs` - SQLite operations (simple interface: log_event(), query methods)
- `history.rs` - Directory navigation history (simple interface: navigate_to(), items_for_display())
- `ui_state.rs` - UI state machine (simple interface: toggle methods, getters)
- `feature_defs/` - Feature registry (simple interface: compute(), export())
- `context.rs` - System context gathering (simple interface: gather_context())
- `analyze_perf.rs` - Performance analysis (simple interface: analyze_perf())
