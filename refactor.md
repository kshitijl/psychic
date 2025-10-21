# Refactor Plan: Extract Deep Modules from main.rs

**Current state:** main.rs is 2,866 lines with too many concerns mixed together.

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

### 2. **`src/render.rs`** - UI Rendering Module

**Interface (small):**
```rust
pub fn render_normal_mode(f: &mut Frame, app: &RenderState);
pub fn render_history_mode(f: &mut Frame, app: &HistoryRenderState);
pub fn render_filter_picker(f: &mut Frame, area: Rect);
```

**Deep implementation inside:**
- All layout logic (horizontal vs vertical based on width)
- Path truncation functions (`truncate_path`, `truncate_absolute_path`, `abbreviate_component`)
- File list rendering with styling
- Path bar with marquee animation
- Debug pane rendering
- Input field rendering
- All ratatui widget construction

**Why deep:** Rendering has complex layout calculations, truncation logic, styling, but simple interface (just render given state).

---

### 3. **`src/path_display.rs`** - Path Formatting Module

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

---

### 4. **`src/cli.rs`** - CLI Argument Parsing Module

**Interface (small):**
```rust
pub struct CliConfig {
    pub data_dir: PathBuf,
    pub on_dir_click: OnDirClickAction,
    pub on_cwd_visit: OnCwdVisitAction,
    pub initial_filter: FilterType,
    // ... other config
}

pub enum CliCommand {
    Run(CliConfig),
    GenerateFeatures { format: OutputFormat },
    Retrain,
    Zsh,
    Internal(InternalCommands),
}

pub fn parse_cli() -> Result<CliCommand>;
```

**Deep implementation inside:**
- All clap structs (Cli, Commands, InternalCommands, etc.)
- Argument validation
- Default value logic
- Enum conversions

**Why deep:** CLI parsing has many options but interface is just "give me a command".

---

### 5. **`src/app.rs`** - Application State Module

**Interface (small):**
```rust
pub struct App { /* fields */ }

impl App {
    pub fn new(...) -> Result<Self>;
    pub fn handle_event(&mut self, event: AppEvent) -> Result<EventResult>;
    pub fn render_state(&self) -> RenderState;  // Extract what renderer needs
}

pub enum EventResult {
    Continue,
    Quit,
    OpenEditor(PathBuf),
    ChangeCwd(PathBuf),
}
```

**Deep implementation inside:**
- All App state (query, page_cache, selected_index, etc.)
- Worker communication
- Page cache management
- Analytics logging
- Selection movement
- Query management

**Why deep:** App has tons of state but external code just sends events and gets results.

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
