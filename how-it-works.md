# How It Works

## About this document

This document describes the current state of the code, with a brief description of why certain things are done the way they are. It's not meant to be an exhaustive list of all previous iterations of the code. It shouldn't describe how things used to work. Anything learned from previous designs should only show up in the brie "why" statements.

## Overview

`psychic` is a terminal-based file browser with fuzzy search, ML-powered ranking, and click tracking analytics. Built in Rust using ratatui for the TUI.

## Architecture

### Core Components

The codebase follows John Ousterhout's "deep modules" philosophy: simple interfaces hiding complex implementations. After a major refactoring, the codebase has been split into focused, independently testable modules.

**Data & Analytics:**
1. **`db.rs`** - SQLite event logging (clicks, scrolls, impressions)
2. **`analytics.rs`** - Subsession tracking, impression debouncing (200ms logic), scroll deduplication

**File Discovery & Ranking:**
3. **`walker.rs`** - Background file discovery via walkdir
4. **`context.rs`** - System context gathering for ML features
5. **`features.rs`** - ML feature generation for training
6. **`feature_defs/`** - Trait-based feature registry
7. **`ranker.rs`** - LightGBM model inference

**Worker Thread:**
8. **`search_worker.rs`** - Async worker thread for filtering/ranking

**UI & Interaction:**
9. **`ui_state.rs`** - UI state machine (history mode, filter picker, debug pane, help screen)
10. **`history.rs`** - Directory navigation history with branch-point semantics
11. **`keymap.rs`** - Keybinding registry: every binding and its help text, once
12. **`input.rs`** - Dispatches keymap actions, terminal suspension
13. **`render.rs`** - All UI rendering (normal mode, history mode, help screen, layouts)
14. **`help.rs`** - Help screen layout, derived from the keymap, clap and the zsh script
15. **`preview.rs`** - Preview generation with three-state caching (None/Light/Full)

**Application State:**
16. **`app.rs`** - Application state (App struct, page cache management)

**Utilities:**
17. **`path_display.rs`** - Path formatting utilities (truncation, abbreviation)
18. **`cli.rs`** - CLI argument parsing with clap

**Main Entry Point:**
19. **`main.rs`** - Event loop glue (~600 lines, down from ~2000+)

**Development Tools:**
20. **`analyze_perf.rs`** - Performance analysis for timing logs

`internal analyze-perf` reports the most recent **TUI** session, found by looking
for the last `first_render` line. Every invocation logs under its own session id,
including CLI subcommands like `retrain`, which emit no startup timings; anchoring
on the last session id in the file instead produced an empty report whenever the
most recent run was a CLI one. It also stops reading at `startup_complete`, so
per-query timings after startup (`filter_and_rank_total` for later searches,
`query_round_trip`) reach the log but not this report - the debug pane is where
those are meant to be read.

**Timing convention:** every latency is measured from `PROCESS_START` (a `Lazy<Instant>`
in `main.rs`, forced by the first statement of `main`). The worker uses it too, via
`crate::PROCESS_START`, so `walker_complete` is comparable to `first_query_complete`
and `startup_complete`. It previously measured from the worker loop's own start,
which made it read ~10ms faster than it was and not comparable to the numbers printed
beside it.

**Why this architecture:** Each module has a simple, focused interface hiding complex implementation. Main.rs is now just the event loop and module coordination. All business logic is in focused modules that can be tested independently.

### Thread Architecture

**5 threads:**
- **Main (UI)**: Renders UI, blocks on unified event channel, owns visible file slice only
- **Worker**: Owns all file data, does filtering/ranking, sends results to unified channel
- **Walker**: Discovers files via walkdir, sends to worker
- **Input**: Sleeps in the kernel until the terminal has bytes, decodes them with crossterm, sends to unified channel. Stoppable, so a child process can own the terminal (see "The input thread" below)
- **Tick timer**: Sends tick events every 200ms for UI animations (marquee)

**Communication via Unified Event Channel:**
```
Walker → (path, mtime) → Worker

// Worker sends directly to unified AppEvent channel
Worker → AppEvent::Worker(QueryUpdated{...}) → Main
Worker → AppEvent::Worker(Page{...}) → Main
Worker → AppEvent::Worker(FilesChanged) → Main
Worker → AppEvent::Worker(WalkerDone{walk_ms}) → Main

// Input thread forwards keyboard and mouse events
Input → AppEvent::Input(Event) → Main

// Tick timer for animations
Tick → AppEvent::Tick → Main

// Retrain thread sends status
Retrain → AppEvent::Retrain(bool) → Main

// One-shot: database counts for the debug pane, gathered on first open
DbStats thread → AppEvent::DbStats(..) → Main

// All events go through single unified channel
Main blocks on event_rx.recv() → instant wake on ANY event
```

**Why unified event channel:**
The main thread uses a single blocking `recv()` on a unified event channel instead of polling multiple channels with timeouts. This provides instant wake-up when any event arrives (worker response, keyboard input, tick, retrain status), eliminating the 0-100ms polling delay that previously caused sluggish first renders. All event sources send to the same channel, and main wakes instantly.

**Why separate input thread:**
crossterm's `event::poll()` blocks, which would prevent instant response to worker
events. Running it on its own thread that forwards into the unified channel keeps
every event type instant.

### The input thread

Lives in `tty_input.rs`. It has to do two things that are harder to combine than
they look: wait for the terminal, and stop on demand. It must stop because
pressing Enter on a file hands the terminal to `$EDITOR`, and two readers on one
terminal means the editor loses keystrokes to us.

crossterm's synchronous API offers exactly two ways to wait: `read()`, which
blocks forever, and `poll(timeout)`, which blocks for at most the timeout.
Neither can be woken by another thread. That is a missing feature, not a bug, and
three earlier designs worked around it: blocking in `read()` and forwarding to a
channel (`83ccd32`, which ate the editor's keystrokes, because a thread inside
`read()` cannot be told to stop); then `crossbeam::select!` on a pause channel
with a 10ms `default` arm followed by `event::poll(Duration::ZERO)` (`9058bec`),
which never blocked and so could always be paused, at the price of checking the
terminal only 100 times a second; then a 50ms sleep after signalling a pause
(`da7d259`), hoping the thread had noticed by then.

**Why a pipe is the answer.** `crossbeam::select!` waits on *channels*. The
terminal is not a channel, it is a file descriptor, and the only thing that can
wait on a file descriptor is the kernel:

```
    crossbeam::select!   channels: yes    file descriptors: no
    kernel poll(2)       channels: no     file descriptors: yes
```

To wait on the terminal *and* on "please stop" in one call with no timeout, one
of them has to cross over into the other's world. Making the terminal look like a
channel is the design that ate keystrokes. So the stop signal is made to look
like a file descriptor instead: a pipe. `TtyInput::pause` writes one byte, and
the kernel wakes the thread out of a `poll` that had no timeout at all.

The thread waits on three descriptors: the terminal, that pipe, and a second pipe
fed by a SIGWINCH handler. The third is needed because a window resize arrives as
a signal rather than as bytes, so `poll` sleeps straight through it. crossterm
has an `Event::Resize` ready, but only hands it over when asked, and we only ask
when the kernel wakes us. The signal cannot do the waking either: `signal-hook`
registers with `SA_RESTART`, so the kernel restarts the interrupted `poll`
rather than returning `EINTR`.

Without that pipe a resize is not lost, only late. The 200ms tick redraws, and
ratatui re-reads the terminal size on every `draw`, so the UI reflows on the next
tick: measured at 134-151ms, against about 1ms with the pipe. That fallback stops
existing the moment ticks are made conditional on something actually animating,
which is the remaining half of this cleanup.

`signal-hook` keeps a registry of handlers per signal, so ours is added alongside
crossterm's rather than replacing it: crossterm still turns the signal into the
event, and our pipe only wakes us up to ask for it.

**What crossterm still does** is everything except deciding when to read: escape
sequence decoding, the kitty keyboard protocol, SGR mouse, bracketed paste,
sequences split across reads, raw mode, the alternate screen, and being ratatui's
backend. It is called only with a zero timeout, after the kernel has already said
a byte is waiting, so it can never block and the eaten-keystroke bug cannot come
back.

Two consequences worth knowing:

- **Crossterm's queue has to be drained to exhaustion** before going back to the
  kernel. It reads the terminal in blocks and can decode several events out of
  one read - a paste, or a mouse drag - and those extra events sit in its queue,
  not in the kernel's buffer. Waiting on the descriptor while they are queued
  would hang until the user happened to press another key.
- **Hangup is checked before readability.** A terminal that has gone away reports
  both, and at end of file the descriptor is readable forever, so treating that
  as "there is input" would spin the thread at 100%. Nothing is lost: the thread
  drains crossterm one last time on its way out.

**Pausing is now an answer, not a guess.** `pause()` returns only once the thread
has confirmed on an ack channel that it has stopped reading, so the child process
provably has the terminal to itself. It returns a guard whose `Drop` resumes the
thread, so no early return on the way back from a child can leave the terminal
permanently deaf. The 50ms sleeps are gone.

**Why separate tick thread:**
Previously relied on event poll timeout for periodic UI updates (marquee animation). Now that we use blocking recv(), a dedicated tick thread sends periodic events to trigger redraws for animations.

Why: Worker owns file data to avoid blocking UI. Main thread only keeps what's visible (20-40 files), not all files (thousands). Unified event channel ensures instant rendering of worker responses (~10ms) vs previous polling delay (~50ms average).

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

```sql
CREATE TABLE hidden_prefixes (
    path TEXT PRIMARY KEY,   -- absolute, canonical
    created_at INTEGER
);
```

**Why hiding is its own table:** it is mutable, undoable state, not a log. Hiding
suppresses results and nothing else - the events under a hidden directory stay exactly
as they are, so a directory you stop using still contributes everything it taught the
model. Hiding is deliberately *not* recorded as an event and *not* a training signal:
it is a rare, explicit act of curation, and treating it as a negative label would put
weight on something the user does a handful of times a year.

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

Background thread that walks the current directory with `walkdir`, in two passes.

**Two passes, because the two halves of a tree are worth very different amounts.**

1. **The root's own children** (`min_depth(1).max_depth(1)`). One `readdir`, and
   it is exactly what the user is looking at, so it is sent immediately and
   followed by `ChildrenDone`.
2. **Everything below them** (`min_depth(2)`). This is the pass that can be
   enormous, so it is collected rather than sent, and only handed over if it
   finishes under `SHALLOW_MODE_THRESHOLD` (8,000). Past that the tree is
   declared too big to index and pass one stands alone.

The threshold earns its keep twice.

- **It bounds the walk.** Hitting it stops the descent then and there. Without
  it, launching in `/`, or a system directory, or anywhere else with a few
  hundred thousand files under it, would have the walker stat every one of them
  in the background and hand them all to the worker. Since pass one has already
  delivered the directory's own children, the rest of that descent is work
  nobody is waiting for. Measured from `/`: its 21 children are on screen at
  13ms, and the walk below them is abandoned rather than run out.
- **It bounds every keystroke after the walk.** Each entry reported becomes a
  registry entry that is filtered and ranked on every keypress. Eight thousand of
  those is already tens of milliseconds per keystroke against an empty query, so
  the limit is protecting the steady state as much as the startup.

This used to be a single full-depth walk that buffered everything and, on passing
the threshold, threw all of it away and started again at `max_depth(1)`. So the
common case of launching in `~` paid for two walks and showed *nothing* until
both had finished. Now the expensive pass is the one that gets abandoned, and
abandoning it costs nothing that was already on screen. Measured on this machine,
time from launch to the file list appearing, in `~`:

| | before | after |
|---|---|---|
| median of 5 runs | 80.9ms | 13.3ms |

A small tree is unchanged (13.7ms to 12.3ms in this repository): it never hit the
threshold, so it never paid for the restart.

**Key points:**
- Streams the root's children immediately; holds everything deeper until the walk
  is known to be small enough to keep
- Filters: `.git`, `node_modules`, `.venv`, `target`, plus any directory the user
  has hidden that applies to the current root (see "Hiding directories")
- The root itself is exempt from that name filter, so launching inside a
  directory called `target` shows its contents instead of an empty screen
- Sends both files and directories (with `is_dir` flag)
- Extracts mtime, atime, and file_size from walkdir's cached metadata
- Sends `AllDone` when a walk runs to the end
- Checks for commands every 100 entries (COMMAND_CHECK_INTERVAL)

**WalkerMessage enum:**
```rust
pub enum WalkerMessage {
    FileMetadata(WalkerFileMetadata),
    ChildrenDone,
    AllDone,
}
```

`ChildrenDone` and `AllDone` both bypass the worker's 200ms debounce, so the two
moments worth showing reach the screen as soon as they happen rather than on the
next tick of the debounce. Without that bypass, publishing the children early
would have bought nothing: they would have sat in the worker for up to 200ms.

**Command support:** the walker takes `ChangeCwd` commands, and checks for one
every 100 entries so a walk of somewhere enormous can be abandoned when the user
moves on. An interrupted walk *returns* the command that interrupted it, and does
not send `AllDone`. That return is load-bearing: the interrupt used to `try_recv`
the command and drop it, after which the blocking `recv` at the top of the loop
waited for a command that had already been delivered - so navigating during a
long walk meant the directory you navigated to was never walked at all.

Why background thread: large directories take seconds to scan.
Why send metadata: avoids re-fetching it later.

### Module: `context.rs`### Module: `context.rs`

Gathers system context at startup in background thread:
- `cwd` - Current working directory
- `gateway` - Default gateway from `netstat -nr`
- `subnet` - First two octets of local IP
- `dns` - First DNS nameserver from `scutil --dns`
- `shell_history` - Last 10 commands from ~/.zsh_history or ~/.bash_history
- `running_processes` - Output of `ps -u $USER -o pid,comm`
- `timezone` - Multi-tier fallback: $TZ env var → /etc/localtime symlink → "UTC"

Why gather this: Network context (home/office/cafe), shell history (user intent), and running processes help analyze search patterns and could become ML features.

**Note on `timezone`:** still recorded per session, but no feature reads it - all
time windows are rolling (see "Time windows are rolling, not calendar days"). It is
session context for later analysis, not a model input.

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

**Eviction:** `WorkerRequest::Evict { path, query_id }` drops a path the UI found
missing from disk, then re-runs the current query so the row disappears at once. The
`FileInfo` is *marked* `evicted`, not removed: `FileId` is an index into
`file_registry`, and those indices are held by `filtered_files` and by pages already
sent to the UI, so removing an element would invalidate them. `filter_and_rank` tests
the flag before doing any matching work, making an evicted entry cost a bool test.

`add_file` clears the flag if the walker later rediscovers the path - the walker
seeing it on disk is proof it exists, so the eviction should not outlive that.

### Hiding directories

Sometimes a directory is still on disk and still worth having in the training data, but
you never want to click it again - you reorganised, and the old copy keeps winning
searches. `Ctrl-X` on a result hides it: the selected row if it is a directory,
otherwise its parent. `psychic hidden list|add|remove` manages the set from the shell,
which is also how you undo a mis-hit.

**The rule:** a hidden directory is suppressed everywhere *except* from inside it. If
the search root is at or under a hidden prefix, the user has deliberately navigated in
there, and hiding what they are standing in would leave them staring at an empty screen
with no explanation.

The obvious alternative - "show it whenever it is under the current root" - does not
work. Running psychic from a parent directory puts the walker inside the hidden tree,
and the hiding stops meaning anything at exactly the moment it matters.

`active_hidden_for(hidden, root)` is that rule, in one place. It drops the prefixes
containing `root`, and both consumers take its output:

- **The walker** never descends into an active hidden directory (`should_descend`), so
  hiding makes a search tree *cheaper*: the subtree is not walked, rather than walked
  and then discarded. The worker narrows the set before sending it, so starting psychic
  inside a hidden directory walks it normally, with nothing to skip.
- **The registry** carries a `hidden` bool per `FileInfo`, recomputed whenever the root
  or the hidden set changes (startup, `change_cwd`, `hide`). The query path only tests
  the bool, so hiding costs nothing per keystroke.

**Impressions:** hidden rows are dropped in `filter_and_rank`, so they never enter
`filtered_files`, never reach a page, and never reach the UI's page cache - which is
what `check_and_log_impressions` reads. A hidden file therefore cannot be logged as
seen when it was not. `test_hidden_files_never_reach_a_page` pins this down, because it
is the property that keeps hiding from quietly corrupting the training data.

**Hiding an ancestor of the current directory is refused** (`input.rs`, asserted in the
worker). It would do nothing at the time - the current directory is exempt - and then
swallow everything the moment the user walked out of it.

**Known gap:** hiding something while the walker is still running does not stop the
current walk from descending into it. The results are filtered either way; only the
walk is wasted, and only until the next `change_cwd`.

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

On startup, psychic automatically logs a startup_visit event for the initial directory (with an empty query string). This ensures directories you begin in will show up in history without being treated as positive click signals for ranking.

**Historical files:** Loads previously clicked/scrolled files from events.db at startup.
Why: User can find files from other projects they've accessed before.

The query is bounded two ways, because this is the one query on the startup path
whose cost grew with total history (events are never purged):

```sql
SELECT full_path FROM events
WHERE action IN ('click', 'scroll', 'startup_visit')
  AND timestamp >= ?          -- HISTORY_LOOKBACK_DAYS = 365
GROUP BY full_path
ORDER BY MAX(timestamp) DESC
LIMIT ?                       -- HISTORY_MAX_PATHS = 2000
```

- The **time cutoff bounds the work.** `action` and `timestamp` are the first two
  columns of `idx_events_click_lookup`, so this became a range seek over one year
  instead of a scan of all history - the same shape as `load_clicks`. Startup is
  now proportional to recent activity everywhere, not to database size.
- The **limit bounds the result.** Each path returned becomes a file registry entry
  that every later search filters over, so the cap protects steady-state search
  cost too. Ordering by recency means the cap drops the stalest paths.

Neither bound bites at present scale (207 paths from 3,174 rows, oldest 312 days),
which is the point: they are there so an unusual history degrades rather than
slows everything down.

**Stale entries:** the registry is a cache of the filesystem, and the `path.exists()`
filter above validates it only once, at startup. A file deleted mid-session stays in
the results, because nothing else revalidates it: `FilesChanged` fires on walker
*additions* only. The registry is therefore revalidated a second time at the moment
the user acts on a row - see "Act-time validation" under `input.rs`.

The ordering is load-bearing: the worker registers these paths in order and file
registry order breaks ties between equally scored results. It was previously
`SELECT DISTINCT full_path ... ORDER BY timestamp DESC`, which only approximates
recency order - with `DISTINCT` the sort key is not in the result, so which of a
path's timestamps wins is up to SQLite. It comes out close, but on this database
the two orderings genuinely differ in a few places. `MAX(timestamp)` says what we
mean, with the same query plan.

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
        Box::new(ModifiedLast24h),
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

**Time windows are rolling, not calendar days.**

Every time-based feature counts backwards from the moment being scored:
`clicks_last_24h` means "in the 24 hours before this impression", not "since
midnight". There is no timezone anywhere in feature computation, by choice.

Why:

- **Speed.** Resolving a timezone cost more than every other feature combined:
  ~75ms of a 97ms time-to-first-results.

  The reason is worth remembering, because it is not the obvious one. jiff caches
  timezones, so "only the first lookup is expensive" is true - a warm lookup is
  ~2µs. But features are computed per file across a rayon `par_iter`, so the first
  ranking pass after launch had ~170 threads arrive at the cold cache at the same
  moment and queue on its initialization lock. Every one of them paid: ~4.4ms per
  file, ~740ms of CPU, ~75ms of wall clock. Measured directly (`jiff` 0.1, macOS):
  one thread initializing costs ~12ms, 170 threads racing to initialize costs
  ~52ms, and once warm the same work is ~2µs per file.

  **A lazily-initialized global inside a parallel per-item loop is a startup
  cliff, not a per-call cost, and a warm benchmark will not show it.** That is the
  transferable lesson; the timezone was just where it happened to bite.
- **One meaning everywhere.** A calendar day needs to know *whose* day. Feature
  generation replays historical events, and sessions are recorded in whatever
  zone the user was in at the time (this database has sessions in five zones,
  28% of them 9.5 hours off the most common one). "Since midnight" therefore
  meant different spans for different training rows. A rolling window is the
  same number in every zone.
- **Simpler code.** Each of these features is now one call to `clicks_for_file`
  with a window length. The window helpers and their constants live at the top of
  `implementations.rs`.

The cost of the choice: a click at 11pm still counts as "recent" at 10pm the next
day, and stops counting 24 hours later rather than at midnight. For a relevance
signal that is arguably better - it decays smoothly instead of falling off a cliff
when the date rolls over.

Two of these features were renamed to say what they measure: `clicks_today` is now
`clicks_last_24h`, and `modified_today` is now `modified_last_24h`. `modified_today`
had *always* been a 24-hour check (`hours < 24`) despite its name, so only the name
changed there. Feature names are positional in the model, so an existing `model.txt`
keeps working and is replaced by the next automatic retrain.

Measured effect, before -> after:

| | before | after |
|---|---|---|
| `first_query_complete` (time to first results) | 96.85ms | **12.28ms** |
| `rank_files` | 77.02ms | 1.72ms |
| `ml_compute_features` (171 files) | 75.23ms | 0.69ms |
| `clicks_last_30_days`, CPU across all files | 740.90ms | 0.02ms |
| `generate-features` over 52,620 training rows | 0.90s | 0.73s |

Feature computation is no longer the expensive part of ranking - at 0.69ms it now
costs less than the model inference it feeds (0.71ms), and the most expensive
single feature is `file_size_bytes` at 0.25ms total, which is a `stat` syscall
doing real work.

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
    current_episode_id: u64,
}
```

**Process:**
1. Load all events and sessions from database
2. Sort by timestamp (critical for temporal correctness)
3. Single forward pass:
   - Impression: Compute features from accumulator (only past data), store as pending
   - Click/Scroll: Record in accumulator, mark matching pending impressions as label=1, increment episode_id
4. Output all pending impressions as CSV rows

Why single-pass: O(n) instead of O(n²). No future data leakage (features only see past events).

**Episode-based ranking:** Each episode spans from one engagement event to the next.
Why: LambdaRank needs episodes (groups of impressions). Each episode = impressions leading to an action. More meaningful than subsession-based grouping.

**Features computed:** See `feature_defs/implementations.rs` for full list. Examples:
- Query matching: filename_starts_with_query
- Click history: clicks_last_30_days, clicks_last_7_days, clicks_last_24h, clicks_last_hour, clicks_for_this_query
- File properties: is_hidden, is_under_cwd
- Temporal: modified_last_24h, modified_age
- Directory features: clicks_last_week_parent_dir

### Module: `ranker.rs`

Hybrid ranking system that blends simple linear model with LightGBM model inference.

```rust
pub struct Ranker {
    model: Option<Booster>,
    clicks: ClickData,
    stats: Option<ModelStats>,
    total_clicks: usize,
}

pub struct ClickData {
    clicks_by_file: HashMap<String, Vec<ClickEvent>>,
    clicks_by_parent_dir: HashMap<PathBuf, Vec<ClickEvent>>,
    clicks_by_query_and_file: HashMap<(String, String), Vec<ClickEvent>>,
}
```

**Preloading clicks:** All click and scroll events from the last 30 days are loaded at startup into multiple HashMaps:
- `clicks_by_file`: Indexed by full file path
- `clicks_by_parent_dir`: Indexed by parent directory path
- `clicks_by_query_and_file`: Indexed by (query, full_path) tuple for query-specific click tracking
- `total_clicks`: How many of those events there were, used to weight the two models

Why: O(1) lookup per file vs O(n) query per file. Database query runs once with composite index. Multiple indices enable different features without re-querying the database.

**Hybrid Ranking Approach:**

The ranker uses a two-model hybrid system to handle cold-start scenarios (new installations or few clicks):

1. **Simple Linear Model:**
   - Raw score: `3.0 * clicks_last_7_days + 1.0 / (1 + modified_age_in_days) + 2.0 * fuzzy_score`
   - Normalized to [0, 1] with a sigmoid (`k = 0.1`, midpoint at raw score 10)
   - Prioritizes recently clicked files, recently modified files, and query matches
   - Fast to compute, works with zero training data
   - Always computed for all files

2. **LightGBM Model:**
   - Sophisticated ML model trained on full feature set, predicting in [0, 1]
   - Requires training data (model file may not exist on first run)
   - More accurate but only useful with sufficient click history
   - If there is no model at all, ranking is 100% simple score and blending is skipped

3. **Blending:**
   - Final score: `w_simple * simple_score + w_ml * ml_score`
   - Weights come from a tanh ramp over `total_clicks`, with `k = 15` and `l = 2`:

```rust
let ml_weight = (1.0 + (total_clicks as f64 / k - l).tanh()) / 2.0;
let simple_weight = 1.0 - ml_weight;
```

| `total_clicks` | w_simple | w_ml |
|---|---|---|
| 0 | 0.982 | 0.018 |
| 15 | 0.881 | 0.119 |
| 30 | 0.500 | 0.500 |
| 45 | 0.119 | 0.881 |
| 60 | 0.018 | 0.982 |

The ramp crosses over at `k * l` = 30 events and is effectively saturated by 60, so
the interesting range is narrow. Near the midpoint each additional event moves the
ML weight by about `1 / (2k)` = 3.3 percentage points, so the blend can shift
noticeably within a single session.

**The weighting is a rolling 30-day window, not a lifetime total.**

`total_clicks` is a count of click and scroll events from the last 30 days only
(see `load_clicks`), recomputed at every startup. It does not accumulate over the
life of the installation. Consequences worth remembering:

- A month of light use pushes the blend back toward the simple model, even on an
  installation that has been used for years and has a well-trained model. The
  model file is unaffected; only its weight drops.
- Heavy use of one project does not carry over as "trust" once that month passes.
- Scrolls count the same as clicks here, so scrolling through results raises the
  ML weight even when nothing is opened.

Why blend this way: on a new installation the ML model either doesn't exist or has
nothing meaningful to learn from, so the simple model provides reasonable ranking
from recency and click counts. The tanh ramp hands over to the ML model once there
is recent evidence that it was trained on real usage. Tying that to a rolling window
rather than a lifetime counter means a model trained on stale behaviour loses
influence on its own, instead of being trusted forever on the strength of clicks
from a year ago.

**Ranking:**
```rust
pub fn rank_files(
    &mut self,
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
    pub score: f64,      // Blended score from hybrid system
    pub features: Vec<f64>,  // For debug display
    pub simple_score: Option<f64>,  // Debug: score from simple model
    pub ml_score: Option<f64>,      // Debug: score from ML model
}
```

Why file_id instead of path string: No string cloning in hot path. Direct O(1) mapping back to file registry.

**Thread safety:** Ranker wrapped in `SendRanker` with `unsafe impl Send`.
Why: LightGBM Booster contains raw pointers (not Send by default). Safe because model is read-only.

### Training: `train.py`

Trains LightGBM LambdaRank model from features CSV.

**PEP 723 inline metadata:** the script declares its own `requires-python` and
dependencies in a `# /// script` block, so `uv run train.py` builds an isolated
environment on any machine with uv - no virtualenv, and no dependence on this
repository's `pyproject.toml`. That matters because psychic embeds this file,
writes it into the data directory, and runs it from whatever directory the user
launched from, which may be an unrelated project with its own `pyproject.toml`;
inline metadata takes precedence over a surrounding project. Every directly
imported module is listed, including ones that would otherwise arrive
transitively (numpy), so an upstream resolver change cannot break it. Verified
from a cold uv cache: ~10s to resolve, install and run.

`cargo install --path .` users don't need to copy ancillary files manually—the `psychic` binary embeds `train.py` and writes it into the data directory on demand (default `~/.local/share/psychic/train.py`) whenever training runs, overwriting stale copies if the script changed.

**Key parameters:**
- Objective: `lambdarank`
- Metric: NDCG (Normalized Discounted Cumulative Gain)
- Grouping: `episode_id` (engagement-based sequences)

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

**Atomic output:** `model.txt` and `model_stats.json` are written to a temp file in
the same directory and then `os.replace`d over the target. Retraining runs in a
background thread while the TUI is live, and the worker reloads the model on its
own schedule, so a plain write exposed a window where a reader saw a truncated or
empty file. `os.replace` is atomic on POSIX within a filesystem, so a reader gets
either the whole old file or the whole new one. (The temp file's permissions are
reset from `mkstemp`'s 0600 to what a normal write would produce.)

`save_model` also used to write the model twice: once to `<output_prefix>.txt` and
again to a hardcoded `~/.local/share/psychic/model.txt`. In the default case that
was the same path written twice, doubling the window in which it was truncated; with
`--data-dir` it silently wrote to the user's real data directory instead of the one
requested. It now writes once, to the directory it was told to use.

**Loading is fault-tolerant:** `load_ranker` falls back to the simple model if
`model.txt` is missing *or* unreadable, rather than propagating the error. An
unusable model must not stop psychic from starting - ranking degrades to what a
fresh install runs on, and the retrain launched at startup replaces the bad file.

**Visualizations:** Training curves, feature importance, SHAP analysis, score distributions, rank position analysis.

### Module: `ui_state.rs`

Pure state machine for UI mode transitions (no IO, just state).

**State components:**
```rust
pub struct UiState {
    pub history_mode: bool,
    pub filter_picker_visible: bool,
    pub debug_pane_mode: DebugPaneMode,  // Hidden, Small, or Expanded
    pub help_visible: bool,
    pub help_scroll: u16,       // First visible line of the help screen
    pub help_scroll_max: u16,   // Measured by the renderer, clamps help_scroll
}
```

**Methods:**
- `toggle_history_mode()`, `enter_history_mode()`, `exit_history_mode()`
- `toggle_filter_picker()`, `hide_filter_picker()`
- `cycle_debug_pane_mode()` - cycles through Hidden → Small → Expanded → Hidden
- `toggle_help()`, `hide_help()`, `scroll_help(delta)` - help always reopens at the top
- `is_debug_pane_visible()`, `is_debug_pane_expanded()`

Why `help_scroll_max` lives here: only the renderer knows how much of the help
screen overflowed, so it measures during the draw and writes the limit back.
Scrolling is then clamped without the state machine knowing about terminals.

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

### Module: `analytics.rs`

**Simple interface:**
```rust
pub struct Analytics { /* ... */ }

impl Analytics {
    pub fn check_and_log_impressions(&mut self, force: bool, top_n: Vec<FileMetadata>) -> Result<()>
    pub fn log_scroll(&mut self, query: &str, event_data: EventData) -> Result<()>
    pub fn log_click(&self, event_data: EventData) -> Result<()>
    pub fn new_subsession(&mut self, query_id: u64, query: String)
    pub fn next_subsession_id(&mut self) -> u64
    pub fn current_subsession_id(&self) -> u64
    pub fn session_id(&self) -> &str
}
```

**Complex implementation:**
- Subsession tracking (query changes create new subsessions)
- 200ms impression debouncing (don't log on every keystroke)
- Scroll deduplication (HashSet tracks scrolled files to avoid duplicates)
- Event data formatting for database
- Temporal correctness (force flush before click/scroll events)

**Why this module:** Encapsulates all analytics complexity behind simple log_* methods. Main code just calls the methods without worrying about debouncing, deduplication, or subsession management.

### Module: `input.rs`

**Simple interface:**
```rust
pub enum InputAction { Continue, Exit, PrintAndExit(String) }

pub fn handle_input(
    app: &mut App,
    event: Event,
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
) -> Result<InputAction>
```

**Complex implementation:**
- Mouse scrolling (wheel events)
- Text input for search query
- Directory navigation (Enter on files/dirs)
- Terminal suspension for editor/shell
- Terminal cleanup and restoration
- Stopping and restarting the input thread around a child process

**Why this module:** Hides all terminal management complexity. Main event loop just calls handle_input() and gets back a simple action to take.

**Act-time validation:** every action that touches the selected row - open, navigate,
print-and-exit, drop-into-shell - goes through `resolve_selection()`, which is the one
place a row becomes an actionable path and therefore the one place its existence is
checked. It returns a `Selection` only for a path still on disk, so holding one is
evidence the check was made. `handle_enter` and `handle_ctrl_enter` both take this
path, and both log their click through the shared `log_selection_click()`.

If the path is gone, the click is *not* logged (it would be a click on a nonexistent
file in the training data), a `WorkerRequest::Evict` drops the row, and the search bar
title is replaced with `Gone: <name>` until the next keypress. The user stays in the
TUI with the bad row removed, instead of being dropped back into a shell whose `cd`
then fails.

Why here and not at render time: `App::get_file_at_index` is called for every visible
row on every frame and must stay IO-free. This is one `stat`, on one path, per
keypress. Display-time revalidation was considered and deliberately not done.

**What it does not decide:** which key does what. `handle_input` resolves the
event to a `keymap::Action` and then dispatches on that action, so this module
says what each action *does*, never which key triggers it. The dispatch `match`
is exhaustive over `Action`, so a new binding cannot compile until it is handled.

### Module: `keymap.rs`

The single source of truth for input, in the same spirit as `feature_defs/registry.rs`.

```rust
pub static KEYMAP: &[Binding] = &[
    Binding::new(
        Action::ClearQuery,
        &[ctrl('u')],
        Context::Global,
        Section::Search,
        "clear the query",   // the help text the user reads
    ),
    // ...
];

pub fn lookup(code: KeyCode, modifiers: KeyModifiers, context: Context) -> Option<Action>
pub fn lookup_mouse(kind: MouseEventKind) -> Option<Action>
```

**Why a registry:** a keybinding used to be a match arm in `input.rs` and a line
of prose in the README, which drifted. Now the key, the behaviour and the help
text are one row, and a binding that is not documented cannot be expressed:

- `Binding::new` is the only constructor and asserts the description and trigger
  list are non-empty. `KEYMAP` is a `static`, so those asserts are evaluated at
  compile time - an undocumented binding fails the build.
- `input.rs` never inspects a raw key, so behaviour cannot be attached to a key
  without a row here.
- The dispatch `match` is exhaustive over `Action`; `test_every_action_is_bound`
  checks the other direction.
- Every row names a `Section`, and the help screen renders all of `Section::ALL`.

**Lookup order:** bindings in the active `Context` win over `Global` ones, and
explicit chords win over the `AnyChar` catch-all. That lets the filter picker
claim plain letters (`0 c i d f`) without swallowing `Ctrl-C`, and lets every
other letter fall through into the search query.

**Modifier matching:** ctrl and alt must match exactly; shift is ignored, because
crossterm already reports it in the character itself (and as `BackTab`).

**Key names are derived:** `Ctrl-J`, `Alt-Up`, `Shift-Tab` and `Wheel up` are
generated from the triggers, so rebinding a key rewrites the help screen and the
on-screen hint. Declaration order is preference order: `primary_trigger()` takes
the first one, which is why `Ctrl-G` is advertised for help rather than `F1`
(on macOS F1 is a brightness key unless the user has changed that setting).

### Module: `help.rs`

The help screen, opened with `Ctrl-G` (or `F1`) and advertised in the bottom
right of the search box on every frame.

It owns no content. It assembles what is already declared elsewhere:

- keyboard and mouse bindings from `keymap::KEYMAP`
- command line subcommands from the clap definition in `cli.rs`
- shell functions (`p`, `pd`, `pc`) parsed out of the embedded `shell/psychic.zsh`,
  taking each function's description from the comment above it

**Layout:** `lay_out(&blocks, available_width)` returns equal-width columns:
two when both would still be readable, otherwise one, never wider than the space
given. Descriptions are truncated to the column (on a word boundary where that
does not cost most of the line), so one long doc comment cannot stretch the
screen. Everything is pure and tested without a terminal; `render.rs` styles the
lines and handles scrolling for terminals too short to show it all.

**Dismissal:** the help screen is a cheat sheet, not a mode. Up/Down scroll it,
Ctrl-C/Ctrl-D still quit, and any other key closes it - so no keypress acts on
the UI hidden behind it.

### Module: `render.rs`

**Simple interface:**
```rust
pub fn render_normal_mode(f: &mut Frame, app: &mut App, marquee_delay: Duration, marquee_speed: Duration)
pub fn render_history_mode(f: &mut Frame, app: &App)
```

**Complex implementation:**
- Layout calculation (horizontal vs vertical, adaptive based on terminal width)
- Debug pane modes (Hidden, Small, Expanded)
- File list rendering with ranking, colors, truncation
- Preview pane rendering, from text the preview thread has already generated
- Marquee scrolling for long paths
- Filter picker overlay popup
- Path bar with scrolling animation
- History mode layout (directory list + preview)

**Why this module:** Hides all UI rendering complexity. Main loop just calls render_*() functions without understanding layout logic, colors, or animations.

### Module: `preview.rs`

**Simple interface:**
```rust
pub fn spawn<T: From<Preview>>(event_tx: Sender<T>) -> Sender<PreviewRequest>

impl PreviewState {
    pub fn request(&mut self, path: &Path, is_dir: bool, width: u16)
    pub fn ready(&mut self, preview: Preview)
    pub fn visible(&self, path: &Path, height: u16) -> Text<'static>
    pub fn scroll(&mut self, delta: isize)
    pub fn clear(&mut self)
}
```

**On a thread, because how long a preview takes is not up to us.** It used to
run inside `terminal.draw`, so every frame that changed the selection paid for
it before anything could be painted, and holding Down meant one preview per row
in the way of each redraw. The thread keeps only its newest request, since
everything older describes a row the user has already left. The UI shows a
preview only when the path it was generated for is the path selected now, so
there is no moment where one file's contents sit under another's name; the pane
is empty for the frame or two in between.

**In process, because a process spawn cost more than the work.** `bat` and `eza`
were 12-16ms of spawn each, and their output then had to be parsed back out of
ANSI into ratatui spans. Highlighting is now `syntect` - the library `bat` is
built on - and a listing is a `read_dir`, so styles are constructed directly.
That also removes two things psychic had to be installed alongside, and the
silent degradation to `ls` and unhighlighted text when they were missing.

**Only what the pane can show is highlighted, in two states.** Unscrolled, a
screenful is generated and nothing more: that is all anyone can see, and
highlighting is the expensive part. The moment the user scrolls, the rest of the
file is generated in one pass, and scrolling is free from then on. This is what
the `bat` version did with `--line-range :height` and then a full run; generating
the whole file up front instead made a large markdown preview take 150ms.

A sliding window that grows with the scroll offset looks tidier and is worse.
Syntect carries state from line to line, so every pass has to start at line one:
a budget that grows by steps re-highlights the whole preamble each time, costing
about twice the total work and paying it in a series of visible hiccups instead
of one. Measured on a real scroll, the two-state version generates once per file
and then not again - twelve wheel clicks produced one regeneration.

**Syntect is built with `oniguruma` rather than `fancy-regex`.** The pure-Rust
engine looked like the tidier dependency, but measured on markdown - syntect's
heaviest grammar, since it embeds every language it might find in a code fence -
it was several times slower *and* produced a larger binary:

| | fancy-regex | oniguruma |
|---|---|---|
| generate, median | 11.81ms | 2.53ms |
| generate, worst seen | 102.57ms | 25.79ms |
| binary | 12.7MB | 11.2MB |

Measured, moving the selection one row, which is what regenerates a preview:

| | before | after |
|---|---|---|
| median | 25.95ms | 1.71ms |
| p90 | 37.45ms | 3.06ms |
| first full draw | 37.0ms | 3.3ms |

The syntax definitions take 3ms to deserialize, once, on the preview thread
while the walker is still running.

**Complex implementation:**
- syntect highlighting, with the theme's foreground colours only: a theme
  background would paint over the terminal's own, which the rest of psychic
  honours
- Directory listings: permissions, size, date, name, sorted case-insensitively,
  dropping the wide columns below 80 columns the way the `eza` flags used to
- Generated to a line budget and sliced at draw time, so a preview is never
  larger than it needs to be and is not cloned every frame
- Capped at 5,000 lines and 4MB: a preview is not a pager, and the old code
  would happily read an entire file into memory as styled text

**Nothing a file contains may reach the terminal as an instruction.** Control
characters are not merely ugly: an ESC starts an escape sequence the terminal
obeys, and a carriage return or backspace moves the cursor out from under what
is being drawn. Ratatui passes a cell's contents straight through. So every
string that reaches a cell goes through `path_display::printable` first, which
turns tabs into spaces and everything else in the control categories into a dot:
file contents, directory entry names, file list rows, and the path bar. Filenames
are whatever someone managed to create on disk, so they need it as much as file
contents do.

A file whose first 8KB contain a NUL is named as binary rather than painted, but
that check is a courtesy, not the safety net - a file that turns to rubbish
halfway through is caught by sanitising, not by sniffing.

### Module: `main.rs`

Now a clean ~600-line event loop and application glue (down from 2000+ lines before refactoring).

**Startup behavior:**
- Spawns a background thread to retrain the model using collected events
- Training runs asynchronously and doesn't block the UI
- Training output is appended to `~/.local/share/psychic/training.log`
- Worker loads the new model automatically when retraining completes

**Why:** Fresh model on every launch ensures ranking improves as you use the tool. Background execution means no startup delay.

**Main responsibilities:**
- Create unified event channel
- Spawn threads (worker, walker, input, tick timer, retraining)
- Initialize App state
- Run event loop (draw UI, receive events, dispatch to modules)
- Coordinate modules (analytics, input, render, preview)
- Shutdown sequence

**Why now so small:** All business logic moved to focused modules. Main.rs is just coordination and glue code.

**Layout:**

**Wide terminals (≥100 columns):** Horizontal layout
```
┌─────────────┬─────────────┬──────────────┐
│  File List  │   Preview   │ Debug/Stats  │
│   (35%)     │    (45%)    │    (20%)     │
│             │             │              │
│ 1. file.rs  │ [highlighted│ Score: 0.72  │
│ 2. main.rs  │             │ Features:    │
│ ...         │             │  Clicks: 3   │
└─────────────┴─────────────┴──────────────┘
┌──────────────────────────────────────────┐
│   Search Input (bottom)                  │
└──────────────────────────────────────────┘
```

**Narrow terminals (<100 columns):** Vertical stack layout
```
┌──────────────────────────────────────────┐
│  File List (40%)                         │
│                                          │
│ 1. file.rs                               │
│ 2. main.rs                               │
├──────────────────────────────────────────┤
│  Preview (60%)                           │
│                                          │
│  [highlighted preview]                   │
│                                          │
├──────────────────────────────────────────┤
│   Search Input                           │
└──────────────────────────────────────────┘
```
Why adaptive layout: Narrow terminals benefit from vertical stacking (file list + preview stacked) for better readability. Debug pane is automatically hidden in narrow mode to save space.

**Debug pane contents:** selection scores and the full feature vector, then:

- **Latency.** Five numbers: `first paint`, `first results` and `fs walk` are
  measured once from process start; `this search` and `of it, rank` are replaced
  on every query.

  **These are the same measurements the TIMING log lines carry**, not a parallel
  implementation. Each value is computed once and then both logged and displayed:
  `first paint` and `first results` in the main loop, `fs walk` by the worker and
  shipped back on `WalkerDone`, `of it, rank` by the worker as
  `filter_and_rank_total` and shipped back on `QueryUpdated`. The two the worker
  measures are passed through the response rather than re-measured on arrival,
  which would have made the pane read a channel hop slower than the log. Every
  latency in psychic - pane, log and `internal analyze-perf` - starts from the
  single `PROCESS_START` instant in `main.rs`.

  Two differences remain between the pane and `analyze-perf`, both by design:
  `analyze-perf` stops reading at `startup_complete`, so the `filter_and_rank_total`
  it prints is the one from startup while the pane's `of it, rank` is the most
  recent query; and `query_round_trip` ("this search") happens after startup, so it
  reaches the log but never the `analyze-perf` output. The last two are the interesting pair - `this search` is the
  round trip the user actually feels (request sent to results in hand), and
  `of it, rank` is the worker's share of that, so the gap between them is channel
  hop and scheduling. Times are captured in `App::next_query_id` /
  `App::note_query_completed`; every path that sends work to the worker takes its
  query id from `next_query_id`, which is what keeps "this search" honest whether
  the user typed, changed directory, or switched filters. The worker reports its
  own `rank_ms` in `QueryUpdated` rather than the UI guessing.
- **Database contents.** Event counts by action, sessions, file size, and how many
  days of history that represents. Counting rows is a covering-index scan
  proportional to total events, so it is loaded **lazily**: the first time the pane
  is opened, on a background thread, arriving as `AppEvent::DbStats`. A launch that
  never opens the pane never pays for it. Until it arrives the pane shows
  "counting...".

**Debug pane modes (wide terminals only):** Ctrl-O cycles through three states:
- Hidden (default): No debug pane visible
- Small: Debug pane at 20% width
- Expanded: Debug pane at 75% width, file list at 25%, preview hidden
Why: Progressive disclosure - hide when not needed, expand for detailed debugging. Not available in narrow mode (<100 columns) where space is limited.

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

    // Debug pane
    timings: Timings,                  // Latencies shown in the pane
    db_stats: Option<db::DbStats>,     // Loaded on first open, in the background
    query_sent_at: Option<(u64, Instant)>,  // Times the round trip of the live query

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
```rust
loop {
    // Check and log impressions if >200ms elapsed
    app.check_and_log_impressions(false)?;

    // Draw UI
    terminal.draw(|f| { render_ui(f, &mut app, ...) })?;

    // Drain logging channel (non-blocking, for logs sent before unified channel migration)
    while let Ok(log_msg) = app.log_receiver.try_recv() { ... }

    // Block until ANY event arrives (instant wake!)
    let app_event = event_rx.recv()?;

    // Handle event
    match app_event {
        AppEvent::Worker(response) => { /* handle worker response */ }
        AppEvent::Input(event) => { /* handle keyboard/mouse */ }
        AppEvent::Tick => { /* trigger animation redraw */ }
        AppEvent::Retrain(status) => { /* update retraining status */ }
        AppEvent::DbStats(stats) => { /* fill in the debug pane's database counts */ }
    }
}
```

**Why this structure:**
- Blocking `recv()` instead of polling with timeout eliminates 0-100ms delay (avg ~50ms)
- Worker responses render instantly (~10ms total: 6ms worker + ~4ms render)
- Zero CPU usage when idle (recv() blocks until event arrives)
- All event sources wake main thread immediately via unified channel
- Draw happens after each event to ensure UI stays responsive

**Keyboard:** (declared in `keymap.rs`, which is also what the help screen shows)
- Ctrl-G or F1 → show the help screen listing every binding and command
- Type → send UpdateQuery to worker
- Up/Down → move selection, request new visible slice if needed
- Ctrl-P/Ctrl-N → move selection up/down (same as Up/Down)
- Alt-Up → navigate to parent directory
- Left/Right → navigate directory history (back/forward, like browser)
- Enter → log click, launch editor (files) or execute on-dir-click action (directories)
- Ctrl-Enter → execute on-cwd-visit action for selected directory (drop into shell or print path)
- Ctrl-J → execute on-cwd-visit action for current working directory (drop into shell or print path)
- Ctrl-H → toggle history navigation mode
- Ctrl-U → clear query
- Ctrl-O → toggle debug pane
- Tab → cycle to next filter
- Shift-Tab → cycle to previous filter
- Ctrl-F → toggle filter picker
- 0/c/i/d/f (when filter picker visible) → select filter (0=none, c=cwd recursive, i=direct cwd only, d=dirs, f=files)
- Ctrl-C/Ctrl-D/Esc → quit (Esc also closes filter picker or history mode if open)

**Mouse:**
- ScrollUp/ScrollDown → scroll preview pane

Why mouse scrolls preview: Keyboard for navigation (fast), mouse for exploration (natural). Most users don't use mouse for results list.

**Shell Integration:**
Run `eval "$(psychic zsh)"` in your ~/.zshrc to enable the `p`, `pd`, and `pc` commands, plus automatic directory tracking:
- `p` - launch psychic, press Ctrl-J to cd your shell to the selected directory
- `pd` - launch psychic in directories-only mode (same as `psychic --filter=dirs --on-dir-click=print-to-stdout`)
- `pc` - launch psychic in current-directory-only mode (same as `psychic --filter=cwd --on-cwd-visit=print-to-stdout`)
- **Automatic directory tracking**: The shell integration adds a `chpwd` hook that automatically tracks every directory you visit via `cd`. This allows psychic to learn which directories you frequently visit without requiring explicit clicks.
- Without shell integration, Ctrl-J spawns a new shell in the selected directory (old behavior)

**How automatic tracking works:**
The shell integration installs a `__psychic_hook()` function that runs after every directory change. This hook calls `psychic track-visit <directory>` in the background, which logs a `startup_visit` event to the database. These visits appear in search results and help the ML model learn your directory preferences, but they don't count as positive click signals (unlike actual clicks), preventing bias in the ranking.

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
- Right pane: listing of the selected directory, from the same preview thread as the file list
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
- Highlighted in process with `syntect`; see "Module: `preview.rs`"
- Generated once per path and sliced to the visible window when drawn
- Scrollable with mouse wheel, clamped to the preview that exists

Why generate once: scrolling then costs nothing, since the offset is applied when slicing.

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

8. **Preview off the UI thread:** generated once per path on its own thread and sliced when drawn.
   Why: the redraw never waits for a file read, and scrolling re-copies only what is on screen.

## Shutdown Sequence

**The rule: the log sink must outlive everything that logs.**

Background threads log as they wind down, and several of them are only *told* to
stop by `App` being dropped. If `App` also owned the receiving end of the logging
channel, dropping it would take the sink with it - and fern reports a send to a
dead channel by printing the entire record to stderr, over the terminal that is
at that moment being restored. The user sees a wall of `Error performing logging`
after quitting.

So `log_rx` is owned by `main`, not by `App`, and dropped last. That is
structural rather than a matter of ordering: `App` cannot close the sink because
it does not hold it. The earlier fix - join each thread before `drop(app)` - only
ever fixed the thread being joined, and had to be done again the next time a
thread was added. The retraining and context threads are detached and can log at
any moment, so they could not have been fixed that way at all.

**Order:**
1. Drop `worker_tx` (signals the worker to stop) and join the worker
2. `input.shutdown()` - not for the logging, but so nothing is still reading the
   terminal while it is being put back
3. Drop `app`
4. Restore the terminal: raw mode, enhancement flags, mouse capture, alt screen
5. Drop `log_rx` last

## Dependencies

- `ratatui` - TUI framework
- `crossterm` - Terminal backend
- `walkdir` - Recursive directory traversal
- `rusqlite` - SQLite (bundled feature for static linking)
- `lightgbm3` - LightGBM inference
- `anyhow` - Error handling
- `jiff` - Timestamps
- `syntect` - syntax highlighting, in process
- `clap` - CLI argument parsing
- `fern` - Logging dispatch
- `log` - Logging facade
- `once_cell` - Lazy static for feature registry
- `timeago` - Human-readable relative timestamps
- `rand` - Random number generation (session IDs)

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
psychic zsh
# Add to ~/.zshrc: eval "$(psychic zsh)"

# Track a directory visit (used by shell integration hook)
psychic track-visit /path/to/directory
# Logs a startup_visit event for the directory

# Internal commands (development/debugging)
psychic internal analyze-perf
psychic internal print-log
psychic internal clear-log
psychic internal summarize-events

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
