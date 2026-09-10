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
15. **`preview.rs`** - Previews on their own thread, highlighted in process

**Application State:**
16. **`app.rs`** - Application state (App struct, page cache management)

**Terminal:**
17. **`tty_input.rs`** - The input thread: waits in `libc::poll` on the terminal
    and a self-pipe, decodes with crossterm, and can be stopped on demand

**Utilities:**
18. **`path_display.rs`** - Path formatting utilities (truncation, abbreviation)
19. **`cli.rs`** - CLI argument parsing with clap

**Main Entry Point:**
20. **`main.rs`** - Event loop glue (~900 lines, down from ~2000+)

**Development Tools:**
21. **`analyze_perf.rs`** - Performance analysis for timing logs
22. **`bench/`** - Not a module: the benchmark harness, in Python. `run.py`
    measures speed against another commit, `model.py` measures ranking quality.
    See `llm.md`.

**One TIMING line per query.** Ranking used to write about 23 lines per query -
eight op lines plus one per feature - and fern flushes per record, so each was a
write syscall on the worker thread standing between the user's keystroke and
their results. A three-query session left 66 of them in app.log. Now
`WorkerState::log_query_timings` writes one:

```
TIMING {"op":"query","filter_ms":1.5,"simple_ms":0.25,"features_ms":0.75,
        "predict_ms":0.5,"blend_ms":0.125,"total_ms":3.25,"count":171,
        "per_feature":{"fuzzy_score":0.25, ...}}
```

Three things follow from that, all deliberate:

- **The line is written after the results are sent.** `rank_files` returns a
  `Ranking` - scores plus a `RankTimings` - instead of logging as it goes, and
  `send_query_updated` sends `QueryUpdated` first and logs second, so the write
  syscall is behind the user's results rather than in front of them.
- **Per-feature timing costs an accumulator per rayon chunk, not a map per
  file.** It used to build a `FxHashMap` with 15 freshly allocated `String` keys
  for every file on every keystroke. `compute_features_into` adds into a
  `&mut [Duration]` indexed by registry position, and the parallel pass is a
  `fold`/`reduce` so each chunk carries one accumulator. Measured over 171 files,
  feature computation went from 0.329ms to 0.225ms a pass, a third of it gone.
  The chunks recombine in order, which
  `test_features_come_back_aligned_with_their_files` pins by giving every file a
  unique size and checking each scored file gets its own back.
- **`Instant::now` was never the cost** and per-feature timing is still on. It is
  tens of nanoseconds; the allocations and the syscalls were the expense.

`QueryTimings::to_json` builds the line, and its shape is pinned from both sides:
by expect tests in `search_worker.rs` and by `analyze_perf.rs`'s own tests, which
parse the same string back. Numbers are rounded to microseconds, which is past
what `Instant` resolves and several times shorter than full f64 precision.

`internal analyze-perf` reports the most recent **TUI** session, found by looking
for the last `first_render` line. Every invocation logs under its own session id,
including CLI subcommands like `retrain`, which emit no startup timings; anchoring
on the last session id in the file instead produced an empty report whenever the
most recent run was a CLI one. It also stops reading at `startup_complete`, so
per-query timings after startup (later `query` lines, `query_round_trip`) reach
the log but not this report - the debug pane is where those are meant to be read.

**Timing convention:** every latency is measured from `PROCESS_START` (a `Lazy<Instant>`
in `main.rs`, forced by the first statement of `main`). The worker uses it too, via
`crate::PROCESS_START`, so `walker_complete` is comparable to `first_query_complete`
and `startup_complete`. It previously measured from the worker loop's own start,
which made it read ~10ms faster than it was and not comparable to the numbers printed
beside it.

**Why this architecture:** Each module has a simple, focused interface hiding complex implementation. Main.rs is now just the event loop and module coordination. All business logic is in focused modules that can be tested independently.

### Thread Architecture

**Five that live for the session:**
- **Main (UI)**: Renders UI, blocks on unified event channel, owns visible file slice only
- **Worker**: Owns all file data, does filtering/ranking, sends results to unified channel
- **Walker**: Discovers files via walkdir, sends to worker
- **Input**: Sleeps in the kernel until the terminal has bytes, decodes them with crossterm, sends to unified channel. Stoppable, so a child process can own the terminal (see "The input thread" below)
- **Tick timer**: Sends tick events every 200ms for UI animations (marquee)
- **Preview**: Reads and highlights the selected file off the UI thread (see `preview.rs`)

**And four that do one job and exit**, all detached: the retrainer (`ranker.rs`,
which itself spawns `uv` for `train.py`), the context gatherer and the database
statistics query (both in `main.rs`/`app.rs`, feeding the debug pane), and the
walker's own restart on a directory change. Nine `thread::spawn` sites in all,
which is worth knowing when reading a stack trace.

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
3.  **ID'd Requests:** Every request from the UI to the worker (`UpdateQuery`, `GetPage`, `Reload`) carries the relevant `query_id`.
4.  **ID'd Responses:** Every response from the worker back to the UI (`QueryUpdated`, `Page`) also carries the `query_id` of the request it is responding to.

This ensures that both the UI and the worker can safely discard stale messages, preventing state corruption and crashes.

### Module: `db.rs`

**Impressions are the rows that were on screen.** Not the top 25, which is what
they used to be: `check_and_log_impressions` walks
`file_list_scroll .. file_list_scroll + visible_list_height`, the window the
renderer actually drew. An impression is the model's only evidence that
something was *shown and passed over*, so a fixed 25 both invented negatives
below the fold on a short terminal and missed real ones below row 25 on a tall
one. Before the first frame nothing has been seen and nothing is logged -
`visible_list_height` is 0 until the renderer reports it in `FrameLayout`.

**Impressions record where they were shown, and what they were.**
`events.rank` is the row's position in the list, counting from 1, and it is set
only for impressions - a click has no position because a click is not a list.
`events.is_dir` is what the row was at the moment it was shown.

`is_dir` is recorded rather than looked up because training used to answer it by
stat-ing the path during feature generation - today's filesystem answering a
question about last March, wrong for anything since deleted or replaced, and a
syscall per row across 80k rows. Rows written before the column exists keep
`NULL` and still fall back to that `stat`, so the old answer ages out as history
accumulates rather than needing a migration that cannot be written. It matters because a row
nobody clicked at position 1 is a far stronger "no" than the same row at
position 24, which may never have been looked at, and today the training data
treats those two identically.

Nothing reads it yet, and that is the point of collecting it now: it cannot be
backfilled. Rows written before this keep `NULL`, so any future use has to cope
with a mixture until enough history accumulates.

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

**Indexes:**
```sql
CREATE INDEX idx_events_engagement
  ON events(action, timestamp, full_path)
  WHERE action IN ('click', 'scroll', 'startup_visit');

CREATE INDEX idx_events_action ON events(action);
```

The first serves the two queries on the start-up path, which both seek by action
and timestamp and read `full_path`. It is **partial** because 96% of the table is
impressions, which nothing ever looks up by action, and index entries carry
`full_path`: covering them all cost 10MB against 0.3MB for the rows that are
actually queried.

**A partial index is only used when SQLite can prove the query's WHERE implies
the index's**, and it does not work out that `IN ('click','scroll')` implies
`IN ('click','scroll','startup_visit')`. `Ranker::load_clicks` therefore states
the index's own predicate as a redundant extra clause; without it the query
silently becomes a full table scan. `db::plan_tests` runs `EXPLAIN QUERY PLAN`
over every query psychic actually issues and fails if one stops using its index,
including a test that pins *why* the redundant clause is needed.

The second index exists because the debug pane counts events by action, and a
partial index cannot count the rows it excludes. It holds no `full_path`, so it
is 1.7MB rather than 10MB.

**Migration.** `Database::migrate` runs on every open and is a no-op once done.
It lives there rather than at start-up because every entry point - the TUI,
`retrain`, `track-visit`, `hidden` - writes through the same statements, and one
of them inserts a session row that would fail against the older, wider table. It
`VACUUM`s only when it actually dropped something, which took 83ms once on a
67MB database.

### The first launch on a new machine

There is no model, no click history and no database. Everything below has a test
in `search_worker::fresh_install_tests`, because the pieces of this were each
covered while the whole was not, and it stayed broken for a long time without
anyone noticing.

- **The database is created on first open**, tables and both indexes together;
  the migration path only runs against an older one.
- **Ranking runs on the simple model.** `load_ranker` finds no `model.txt`,
  falls back, and every score comes back with `simple_weight` 1.0 and no ML
  score at all. The first query still returns results, including the row for the
  directory the user is standing in.
- **Training does not run.** A fresh install has impressions but nothing
  clicked, so there is nothing to learn from. `generate_features` reports how
  many rows it wrote and how many were clicked, and `retrain_model` returns
  early when none were. Handing an empty set to `train.py` produced a Python
  traceback and an `ERROR` in the log on every first launch, for a state that is
  entirely normal.
- **Once the user has clicked, it trains and the model is used.** The blend
  hands over as engagement accumulates, crossing at 30. That whole loop -
  use it, train, load, rank with the trained model outweighing the simple one -
  is `trained_model_tests`. It runs `train.py` for real, which costs about seven
  seconds of the suite's eight; that is the price of the only test that covers
  the loop end to end, and `uv` is required to use psychic regardless.

**Two ordering bugs lived here**, both only reachable on a first launch, which
is why nothing caught them:

- Feature generation opened its own connection without going through
  `Database`, so on a first launch it reached the file before anything had
  created the schema and every launch logged `no such table: events`.
- `PRAGMA busy_timeout` was set *after* `PRAGMA journal_mode = WAL` in the same
  batch. Switching a fresh database to WAL takes a write lock, and on a first
  launch several threads reach that line at once, so the losers failed
  immediately with `database is locked` instead of waiting. The first open of a
  file is now done while holding the registry lock, so the others wait for it
  rather than racing.

### Why the same database is opened several times

A `rusqlite::Connection` is `Send` but **not `Sync`**: it can be moved to another
thread but not shared with one. Putting it behind a mutex would not help either,
it would just serialise the UI's impression logging against the worker's queries
and the retrainer's full-table read. So a thread that needs the database opens
its own. That is the design, not an oversight.

Who holds one, on a normal launch:

| | | |
|---|---|---|
| main thread | `Analytics` | for the life of the process; every click, scroll and impression |
| worker thread | `WorkerState.db` | for the life of the thread; click history, hiding, reloads |
| history loader | transient | one query and a `stat` per path, then gone |
| context thread | transient | one session row and one startup visit |
| retrainer | transient | reads every event to build training data |

What is *not* per connection is the schema. `CREATE TABLE IF NOT EXISTS` and the
migration used to run down every one of them; they now run once per database
file per process, tracked in `PREPARED`. In-memory databases are excluded,
because every `:memory:` connection is a separate database that happens to share
the name, and remembering it would leave the second one empty.

**The file descriptor count is not the connection count. Do not tune this by
reading `lsof`.** A normal run shows four handles on `events.db` while only two
connections are live. The other two are descriptors SQLite has parked rather
than closed. From `unixClose` in its unix layer (`setPendingFd`, and the comment
above the call in `sqlite3.c`):

> If there are outstanding locks, do not actually close the file just yet
> because that would clear those locks. Instead, add the file descriptor to
> `pInode->pUnused` list. It will be automatically closed when the last lock is
> cleared.

POSIX advisory locks belong to the *process*, not to the descriptor, so closing
any descriptor for a file drops every lock the process holds on it. SQLite
therefore cannot close a connection's descriptor while another connection is
holding a lock on the same file; it parks it and closes it when the last one
goes. Consequences worth knowing before anyone tries to "fix" a count:

- **Count the `-shm` files, not the `events.db` lines.** Each live connection to
  a WAL database has one shared-memory file open, so `lsof -p <pid> | grep -c
  'events.db-shm'` is the number that means something. The `events.db` count
  includes the parked ones.
- **It is timing-dependent.** Whether a transient connection's descriptor gets
  parked depends on whether another connection happened to hold a lock at the
  moment it closed, so the number moves between runs and between builds for
  reasons that have nothing to do with how many connections the code opens.
- **Fewer connections can mean more parked descriptors.** Removing an open
  moved this count from three to four, with the live count unchanged at two.
  That is measured, not hypothetical: a variant built without the worker's
  long-lived connection showed four as well.

**Session ID:** Random 64-bit integer (not UUID).
Why: UUIDs are 36 chars. 64-bit int gives 18 quintillion IDs, more compact.

**Impressions are written as one transaction.** They arrive two dozen at a time,
on the UI thread, after every query the user pauses on; a statement each meant a
transaction and an fsync each.

**File metadata:** Captured at event time, not discovery time.
Why: Files can be modified between discovery and impression. Event-time metadata reflects what user actually saw.

### Module: `walker.rs`

Background thread that walks the current directory in two passes.

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

**What is skipped.** The walk is built on the `ignore` crate, the one ripgrep and
fd use, so `.gitignore`, `.ignore`, `.git/info/exclude` and the user's global
gitignore all apply, to files as well as directories - `*.o` and `*.pyc` are most
of the noise in a built project, and a directory-only filter never saw them.
Gitignore semantics are not worth reimplementing: negation, `**`, directory-only
rules, anchoring, nested files and precedence between them.

Three deliberate departures from that crate's defaults:

- **Dotfiles are walked.** It hides them, because that is what ripgrep wants.
  psychic is for finding `.zshrc` as much as `main.rs`, and the model has an
  `is_hidden` feature that would go blind if they never appeared.
- **`.git`, `node_modules`, `.venv` and `target` are skipped by name anyway.**
  A floor under the ignore rules rather than a replacement: outside a git
  repository there is nothing to read, and these four are noise everywhere.
  `.git` needs naming because no gitignore ever lists it - git excludes it
  implicitly, and we walk dotfiles.
- **Ignore files are honoured outside a repository** (`require_git(false)`). If
  someone wrote one, it means what it says wherever it is.

`--no-ignore` turns all of that off, spelled as ripgrep and fd spell it. The
root itself is exempt from the name filter, so launching inside a directory
called `target` shows its contents instead of an empty screen. Files already in
the events database are added to the registry regardless: you chose them once,
so they stay findable even if git would hide them.

**Do not set `min_depth` on the builder.** It stops the crate applying ignore
rules to anything shallower, so an ignored directory at depth one is never
pruned and the whole of it is walked. Since the second pass starts at depth two,
that is exactly the shape this walker has: with `min_depth(2)`, `target` was
walked in full - 57,000 entries in this repository, enough on its own to pass
the threshold and drop the repository to showing only its top level. The depth
range is applied after the walk instead, which costs one extra `readdir` of the
root.

**Measured**, launching in three directories, before and after ignore files were
respected:

| | entries | walk | hit the limit |
|---|---|---|---|
| this repository, before | 21 | 57.9ms | yes |
| this repository, after | 53 | 17.0ms | no |
| `~/local-src`, before | 1,680 | 33.8ms | no |
| `~/local-src`, after | 1,283 | 33.8ms | no |
| `~`, before | 130 | 128.0ms | yes |
| `~`, after | 130 | 80.9ms | yes |

Fewer trees reaching the threshold is the point: a repository that degrades to a
top-level listing is one where search is worth least.

**Key points:**
- Streams the root's children immediately; holds everything deeper until the walk
  is known to be small enough to keep
- Sends both files and directories (with `is_dir` flag)
- Extracts mtime, atime, and file_size from the metadata the walk already had
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
- `timezone` - Multi-tier fallback: $TZ env var → /etc/localtime symlink → "UTC"

Why gather this: network context (home, office, cafe) may become an ML feature.

`running_processes` (the output of `ps`) and `shell_history` (the last ten
commands typed) used to be collected here too. Nothing ever read either. They
were 40MB of a 67MB database, and the second was a plain-text copy of what the
user had been doing, sitting in `~/.local/share`. Both are gone, and the columns
with them.

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

**One result list, not two.** `file_scores` is the result set: ranked order,
with each row's score and features at that row's position. There used to be a
parallel `Vec<FileId>` beside it holding the same order, and building a page
searched `file_scores` for a row whose `file_id` matched - once per row. Because
the two lists agreed, that search found row *k* after *k* comparisons, so the
cost grew with how far the user had scrolled rather than with the page size.

Measured over 8,000 results, the size a project under the walker's shallow-mode
threshold reaches:

| page | before | after |
|---|---|---|
| 0 | 0.019ms | 0.013ms |
| 30 | 0.380ms | 0.012ms |
| 60 | 0.761ms | 0.012ms |

The two lists disagreed in exactly one case: when ranking failed,
`filter_and_rank` filled the id list and left `file_scores` empty, and every row
then drew with a score of 0 and no features. That path now builds unscored
`FileScore` rows in the filter's own order, which says the same thing without a
second list to keep in step.

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

**One `stat` per path, on its own thread.** It used to be three syscalls each -
`exists()`, then `canonicalize()`, then `metadata()` - where the first and third
ask the same question, since a `stat` that succeeds *is* the existence check,
and the second answers one nothing asks: every writer of the events table stores
a path that is already canonical, because it comes from a registry entry the
walker canonicalised when it found it. Checked against the real database, of the
stored paths still on disk none differed from their canonical form by anything
but a trailing slash, and `Path` compares and hashes by component, so a trailing
slash could not have produced a second registry entry anyway.

The load also runs beside the ranker rather than after it. The two share
nothing: one reads the model file and the click history, the other reads the
path list and stats each path. The ranker stays on the worker thread, because a
LightGBM `Booster` holds raw pointers and is not `Send`; what crosses the
boundary is a `Vec<FileInfo>`, which is. Both open their own SQLite connection,
which WAL mode is happy with.

Measured against a copy of the real database, 177 historical paths, median of 7:

| | before | after |
|---|---|---|
| `ranker_init` | 4.44ms | 4.38ms |
| `load_historical_files` | 4.06ms | 2.37ms |
| `worker_state_new_total` | 8.56ms | 4.49ms |
| `first_query_complete` | 12.27ms | 8.63ms |

Roughly half of that is each change: with the syscall fix alone and the loads
still sequential, `worker_state_new_total` measured 6.79ms.

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
  columns of `idx_events_engagement`, so this became a range seek over one year
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

**File size is stored as a log, not a byte count.** `log_file_size` is
`log2(1 + size)`. A raw byte count is close to a unique id per file, and reads
as one: it says "the 47,312-byte one" where what it should say is "a small
config rather than a huge log". The log keeps the order of magnitude and drops
the pretence of precision, and it is the number that appears in the SHAP and
dependence plots, where a log axis is the readable one.

It does not change what the model predicts. A gradient-boosted tree splits on
order, and binning is quantile-based, so any strictly increasing transform of a
feature gives back the identical model - checked by training both ways on the
same 85k rows: same 71 trees, predictions equal to the last bit. The leak that
put raw size third by gain was fixed by the time split above, not here.

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
single feature is `log_file_size` at 0.25ms total, which is a `stat` syscall
doing real work.

**What kind of file you open, without naming a file.**
`extension_click_share` is the fraction of recent engagements that landed on
this file's extension. The click counts are per path and can only speak about
paths already clicked; this generalises to every file the user has never
touched, which on any given query is nearly all of them. It is a share rather
than a count so that it says the same thing on the first day and the thousandth
- a count climbs forever and the thresholds a tree learns early would rot.

**Files with no extension get no share**, and that detail is the whole feature.
`Makefile`, `LICENSE` and *every directory* land in the same empty bucket, and
directories are clicked constantly just to navigate, so the bucket's share
describes navigation rather than a kind of file. Counting it cost 1.0 point of
top-1; skipping it gained 2.3, and the feature went from 14th by gain to 6th.
The events table does not record whether a clicked path was a directory, which
is what would let the numerator be cleaned up instead - see the todo item about
recording `is_dir` on events.

**How long since, not how many.** `seconds_since_last_click` is
`ln(1 + seconds)` since the most recent engagement with this exact file, and
`seconds_since_last_click_parent_dir` the same for its directory. The count
windows can say a file was clicked today; only this can say it was clicked a
moment ago, and "a moment ago" is most of what makes a file the one you want
next.

Two details carry the design. It is **logged**, because the difference that
matters is order of magnitude: a minute against an hour is real, an hour against
an hour and a minute is not. And a file with no history reports a constant
`NEVER_CLICKED` of 19.57 - `ln(1 + ten years)` - rather than something outside
the range: the click index only ever holds 30 days, whose log is 14.77, so
"never" sits clear of every real value while staying on the same axis, which is
what lets one monotone split tell "no history" from "old history".

On this developer's data `seconds_since_last_click` came out as the largest
feature by gain - 28.5% of the total, ahead of `fuzzy_score` and
`filename_starts_with_query`.

**Directory visits are not clicks.** `visits_last_7_days` and
`visits_last_30_days` count `startup_visit` events - the zsh `chpwd` hook, via
`track-visit` - for the directory being ranked, and are 0 for files. They are
loaded by `Database::visits_since` into their own index, never mixed into the
click counts: a visit says "I work here", a click says "I opened this", and
adding them together would let an afternoon of navigation look like engagement
with files nobody opened.

This is the one signal a directory row had nothing to say about before.
`clicks_last_week_parent_dir` asks about the row's *container* - for a directory
`/a/b` it counts activity in `/a`, its siblings - so for the rows where "is this
a place I work" is the whole question, every existing feature was answering a
different one. The data was already being collected and read by nothing: 2,312
visits across 130 directories, against 1,165 clicks across 132 paths.

**`is_under_cwd` is one prefix check, on both sides.** `FeatureInputs` used to
carry an `is_from_walker` flag that the feature short-circuited on, because a
walked file is always under the current directory. True, but it meant inference
read a flag where training did a prefix check - two ways of computing one
feature, which is the shape train/serve skew takes. Every path in the registry
is canonical, so the check is exact and cheap. Verified by generating the
training CSV before and after: byte for byte identical.

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
- File properties: is_hidden, is_under_cwd, log_file_size
- Temporal: modified_last_24h, modified_age
- Directory features: clicks_last_week_parent_dir

### Module: `ranker.rs`

Hybrid ranking system that blends simple linear model with LightGBM model inference.

```rust
pub struct Ranker {
    model: Option<Booster>,
    clicks: ClickData,
    stats: Option<ModelStats>,
}

pub struct ClickData {
    clicks_by_file: HashMap<String, Vec<ClickEvent>>,
    clicks_by_parent_dir: HashMap<PathBuf, Vec<ClickEvent>>,
    clicks_by_query_and_file: HashMap<(String, String), Vec<ClickEvent>>,
}
```

**Query-keyed indexes are nested, and resolved once per query.**
`clicks_by_query_and_file` and `engagements_by_episode_query_and_file` are
`query -> path -> events`, not `(query, path) -> events`. A ranking pass has one
query and hundreds of files, so `rank_files` looks the query up once - that is
what `QueryClicks` holds - and each file is then a lookup by path. Under the flat
key, two features each built a `(String, String)` key for every file on every
keystroke: four allocations per file, 972 per query here.

**The fuzzy score is passed in, not recomputed.** `filter_and_rank` matches every
file against the query to decide whether it is a candidate at all, and then
`FuzzyScore::compute` used to build a fresh `SkimMatcherV2` and run the same
match again, per file, per keystroke. The score the filter already has now rides
along on `FileCandidate` and through `FeatureInputs`. Training has no filter, so
`features.rs` does the match itself, against the same string, with one matcher
shared across the whole pass rather than one per row. Checked by generating the
training CSV with the previous binary and this one from the same database: byte
for byte identical.

The one wrinkle is the empty query. The filter scores it `i64::MAX` - everything
matches - while the feature reports 0, because "no query" carries no match
signal. `FuzzyScore::compute` still short-circuits on an empty query before
looking at the number it was handed, so that stays true.

**`FileCandidate` borrows.** It is built fresh for every keystroke, one per
file, and owning its display name and path meant two allocations per file per
query - 486 of them on a 243-file query here. It holds `&'a str` and `&'a Path`
into the worker's registry instead, and nothing in it outlives the `rank_files`
call it was made for. The ranker keeps its own candidate type rather than taking
the worker's `FileInfo`: the worker knows about the ranker, and pointing that
the other way as well would tie the two together for no gain.

**Preloading clicks:** All click and scroll events from the last 30 days are loaded at startup into multiple HashMaps:
- `clicks_by_file`: Indexed by full file path
- `clicks_by_parent_dir`: Indexed by parent directory path
- `clicks_by_query_and_file`: Indexed by (query, full_path) tuple for query-specific click tracking

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
   - Trained on the full feature set with a ranking objective; its score is put
     through a logistic to land in [0, 1] for the blend
   - Requires training data (model file may not exist on first run)
   - More accurate but only useful with sufficient click history
   - If there is no model at all, ranking is 100% simple score and blending is skipped

3. **Blending:**
   - Final score: `w_simple * simple_score + w_ml * ml_score`
   - Weights come from a tanh ramp over `num_positive_examples`, with `k = 15` and `l = 2`:

```rust
let ml_weight = (1.0 + (num_positive_examples as f64 / k - l).tanh()) / 2.0;
let simple_weight = 1.0 - ml_weight;
```

| `num_positive_examples` | w_simple | w_ml |
|---|---|---|
| 0 | 0.982 | 0.018 |
| 15 | 0.881 | 0.119 |
| 30 | 0.500 | 0.500 |
| 45 | 0.119 | 0.881 |
| 60 | 0.018 | 0.982 |

The ramp crosses over at `k * l` = 30 positives and is effectively saturated by 60,
so the interesting range is narrow, and it is crossed once on a given installation:
the number only moves when a retrain writes new stats.

**The gate asks how much the model was trained on, not how busy the last month was.**

`num_positive_examples` is the count of clicked rows in the training data, read
from `model_stats.json` into `ModelStats` and held on the ranker as `stats`. It
answers the only question the blend needs answered - has this model seen enough
to be trusted - and it changes only when a retrain writes new stats.

It used to ramp over the number of clicks and scrolls in the last 30 days,
recomputed at every startup. That made a quiet month hand ranking back
to the simple model on an installation with years of history and a perfectly good
model behind it, and it counted scrolling through results as evidence for the
model. Staleness does need handling, but it is handled where it belongs: the
click *data* is already a rolling 30-day window, and the model's features carry
no file identity, so nothing in the model itself expires.

A model that loaded but whose stats did not gets 0, and so ~2% weight. That means
a working model is nearly ignored, so `load_stats` logs a warning when it happens;
`train.py` writes both files in the same run, so it should not.

Why blend at all: on a new installation the ML model either doesn't exist or has
nothing meaningful to learn from, so the simple model provides reasonable ranking
from recency and click counts, and the ramp hands over once the model has been
trained on enough clicks to beat it.

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

**Thread safety:** the ranker never leaves the worker thread, so nothing has to
assert anything about it. A LightGBM `Booster` holds raw pointers and is not
`Send`; what crosses a thread boundary is the `Vec<FileInfo>` the walker
produces, which is. There used to be a `SendRanker` wrapper with an
`unsafe impl Send` to move it between threads; it is gone. The `unsafe` that
remains is all in `tty_input.rs`, where `libc::poll`, `isatty` and `close` are
called on raw descriptors.
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
- Objective: `lambdarank`, grouped by `episode_id`
- Metric: NDCG at 1 and 5, with early stopping after 50 rounds without improvement
- `lambdarank_truncation_level`: 30, about a screenful
- `num_leaves`: 15, `learning_rate`: 0.1 - smaller and shallower than the
  LightGBM defaults

**Small trees, because they are better here and cheaper too.** Swept over three
seeds and three folds:

| | trees | top-1 | MRR |
|---|---|---|---|
| 31 leaves, lr 0.05 (the defaults) | 93 | 0.7754 | 0.8520 |
| 15 leaves, lr 0.1 | **67** | **0.7960** | **0.8656** |
| 15 leaves, lr 0.2 | 33 | 0.7942 | 0.8636 |
| 7 leaves, lr 0.1 | 83 | 0.7965 | 0.8674 |

Every smaller setting beat the default one on quality. That is what 1,200
positives look like: a model with room for 31-leaf trees uses that room to
memorise. And because predict is proportional to trees times depth, and predict
had become 85% of the cost of ranking a query, the better model is also the fast
one - `predict` went 1.99ms to 0.45ms and the whole filter-and-rank 2.33ms to
0.83ms.

`lr 0.2` shaves the tree count in half again for a difference in top-1 well
inside the seed noise, and is there if the millisecond is ever wanted back.

**The objective is a ranking one, because the question is a ranking one.**
psychic asks "of the files on screen, which is the one" - never "what is the
probability that this file gets clicked". lambdarank is trained on exactly that:
its gradient comes from swapping pairs *within* an episode, weighted by what the
swap does to NDCG, so a change that reorders nothing contributes nothing.

It was `lambdarank` originally, became `regression` in `ac1b74d` ("try using
regression instead of lambdarank") and later `binary`, with no measurement
recorded either way. Measured now, same features, same folds, only the objective
changing: top-1 0.7010 -> 0.7717, MRR 0.7993 -> 0.8422, all three folds up. AUC
falls, 0.9612 -> 0.9444, which is what should happen - AUC is pooled over rows
and is no longer what the model optimises.

Two things follow from the objective being a ranking one:

- **Rows have to be grouped.** LightGBM is handed group *sizes*, not group ids,
  so it reads episode boundaries off consecutive rows. `load_data` sorts by
  episode: they were nearly in order already, since `episode_id` is handed out
  in one pass over time-sorted events, but the accumulator flushes whatever
  impressions are still pending at the end out of a hash map, which is enough to
  break it. `group_sizes` asserts the ordering rather than trusting it.
- **The score is no longer a probability.** It comes out unbounded - about -6 to
  +6 on this developer's data - and `rank_files` puts it through a plain
  logistic before blending it with the simple score, which lives in `[0, 1]`.
  That is the same map the classification objective applied internally; it
  changes no ordering.

`class_weight: "balanced"` went with the old objective. It had never done
anything: it is a scikit-learn parameter, not a LightGBM one, and LightGBM was
ignoring it silently.

**The split is by time, not at random.** `time_split` puts the first 80% of
episodes in train, the next 10% in validation and the last 10% in test. It used
to be a `GroupShuffleSplit`, which held out a random fifth of episodes: an
episode from March 2 was then validated against a model that had trained on
March 3-30. Under that split any feature that identifies a file is rewarded for
knowing the future, memorisation scores as skill, and early stopping happily
keeps adding trees that memorise - the tell was raw `file_size_bytes`, nearly a
unique id per file, ranking third by gain. There is no timestamp column in the
CSV, but `episode_id` is handed out in a single pass over time-sorted events
(`features.rs`), so it is monotone in time and splitting on it splits on time.

Measured on this developer's own 84k-row CSV, the change moves the last 10% of
episodes from "seen during training" to genuinely held out: AUC on those
episodes 0.966 -> 0.938, top-1 0.690 -> 0.664, RMSE 0.110 -> 0.131. The old
numbers were the leak being scored, not quality that was lost. Early stopping
also settles sooner (177 rounds -> 96), which is the memorising trees no longer
paying off.

**The model that ships is refit on every row.** The split answers one question -
how many trees before this starts fitting noise - and it pays the last 20% of
the data to answer it. Shipping the validated fit would ship a model that has
never seen the most recent fortnight, which is the part most like what the user
is about to search for. So `refit_on_everything` regrows the model on all rows
for exactly the round count early stopping settled on, with no early stopping of
its own (nothing is held out to stop against), and that is what `save_model`
writes. `make_params` is shared by both fits, because a round count chosen under
one set of parameters means nothing under another. `model_stats.json` reports
`best_iteration` and takes its feature importances from the shipped model.

Evaluation still belongs to the validated fit: `create_visualizations` and the
printed metrics use it against the time-split test set, since the shipped model
has seen every row and cannot be scored on any of them. Measured one step
earlier - validated on the first 80%, refit on the first 90%, both scored on the
untouched last 10% - the extra rows are worth top-1 0.664 -> 0.698 and MRR
0.763 -> 0.777, with RMSE 0.131 -> 0.130 and AUC flat. Training takes roughly
twice as long (~3.5s -> ~6s), in a background thread.

**Rows are weighted by age.** `recency_weights` halves a row's weight every
`HALF_LIFE_DAYS`, measured from the newest row in the CSV rather than from now,
so a stale export is not uniformly discounted into noise. The `timestamp` column
that feeds this is metadata, not a feature: `csv_columns` emits it and
`prepare_features` drops it before building X.

This costs a little, and was shipped knowingly. Over three rolling-origin folds -
train on a growing prefix of episodes, early-stop on the next 10%, score the 10%
after that, 333 held-out episodes in all - the sweep is monotone in how much
decay is applied:

| weighting | effective rows (last fold) | AUC | top-1 | MRR |
|---|---|---|---|---|
| uniform | 45,073 / 45,073 | 0.9611 | 0.7032 | 0.8009 |
| half-life 365d | 44,219 | 0.9607 | 0.7035 | 0.7976 |
| **half-life 180d** | **41,294** | **0.9601** | **0.6849** | **0.7900** |
| half-life 120d | 36,526 | 0.9566 | 0.6790 | 0.7816 |
| half-life 60d | 20,242 | 0.9571 | 0.6711 | 0.7753 |
| half-life 30d | 7,730 | 0.9541 | 0.6398 | 0.7545 |

The gentler the decay the better it does, and the damage tracks the effective
sample size almost exactly. Two things explain it. Recency is already a feature -
`clicks_last_hour` through `clicks_last_30_days` - so decay adds no information
the model lacked and only removes rows. And with ~1.2k clicks in the entire
history, positives are the scarce resource; a 60-day half-life leaves 224
effective clicks of 1,162, because this developer's history is back-loaded (54%
of rows are 240+ days old).

180 days is the deliberate trade: 83% of the effective rows, ~2pt of top-1, in
exchange for insurance the metrics cannot show yet. A week of unusual activity
keeps a full vote forever under uniform weighting, and any future feature prone
to memorising specific files has its grip on stale rows loosened automatically.
Setting `HALF_LIFE_DAYS = 1000` restores uniform weighting.

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

"Unreadable" includes *a model from a build with a different feature set*, which
is what the first launch after adding a feature loads. `Ranker::new` compares the
booster's feature count against `FEATURE_REGISTRY.len()` and refuses a model that
does not fit. Without that check the model loaded happily and then failed inside
every `predict`, and `filter_and_rank`'s error path handed back the filter's own
order with no scores - worse than the cold-start path, which at least ranks on
the simple model.

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
- Episode tracking: `episode_queries` is every distinct query typed since the
  last engagement, so a click on the file the user reached by typing "tc", then
  "todo", then "todo-current" credits all three. It used to be an `Episode`
  struct in `episode.rs` - a `Vec<String>`, a `contains` check and a `to_json`,
  which is what it still is, three lines inside the thing that uses it.
- Event data formatting for database
- Temporal correctness (force flush before click/scroll events)

`Subsession.created_at` is an `Instant` rather than a wall-clock timestamp: it
answers one question, "has this query been on screen for 200ms", and a wall
clock can go backwards underneath that.

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
pub fn render_normal_mode(f: &mut Frame, app: &App) -> FrameLayout
pub fn render_history_mode(f: &mut Frame, ctx: HistoryRenderContext<'_>) -> PreviewPane
```

**Rendering is a pure function of `&App`.** It reads the app and writes nothing
back. What it works out from the geometry - and only the renderer knows the
geometry - comes back in `FrameLayout`: the preview pane's size, the visible list
height, the scroll offset the frame was drawn at, and the width of the path bar.
The main loop takes those after the frame is drawn.

It used to take a `NormalRenderContext` of about twenty fields copied out of
`App` and return a `RenderUpdates` of five copied back, and the only reason for
any of it was that render mutated two things as it drew: the preview, and the
marquee. Previews moved to their own thread; the marquee now advances in the
`Tick` handler, which is where an animation belongs - it should be driven by the
clock, not by how often the screen happens to be redrawn. `App::advance_marquee`
is the whole of it, and being off the draw path makes it testable, which it now
is. The renderer reports `path_bar_width` because the advance cannot work out
for itself how far there is to scroll.

Two duplicated pieces went with the context: `NormalRenderContext` carried its
own copy of `get_file_at_index`, and `App::update_scroll` carried a second copy
of `compute_scroll` for the case where the renderer had not supplied a scroll
position - a branch nothing could reach, since every caller arrives straight
from a frame.

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

**The syntaxes and theme come from `two-face`, which packages the set `bat`
ships.** Syntect's own defaults are Sublime Text's, and they have no TOML, no
TypeScript and no Dockerfile - `Cargo.toml` rendered as one flat colour. The
theme is `MonokaiExtended`, bat's default, which gives markdown headings some
weight where `base16-ocean.dark` renders them a grey barely distinct from body
text. Output is now byte-identical to bat's on the files checked. Costs about
0.6MB of binary and a millisecond or two of load.

**Is it actually faster, or only spared the process spawn?** Both, and the
second question is the interesting one. `psychic internal preview` exists to
answer it: it runs the generator with no UI, thread or channel in the way, and
reports the syntax load separately, because psychic pays that once at startup
while bat pays it on every invocation. bat's floor below is a one-line file
*with the same extension*, since bat loads grammars lazily and a `.txt` floor
would flatter it.

| file | lines | ours | bat | bat floor | bat's work | ours vs its work |
|---|---|---|---|---|---|---|
| Cargo.toml | 38 | 0.55ms | 8.15ms | 7.29ms | 0.87ms | 1.6x |
| src/render.rs | 80 | 1.77ms | 12.59ms | 9.10ms | 3.49ms | 2.0x |
| how-it-works.md | 80 | 1.46ms | 12.60ms | 10.04ms | 2.56ms | 1.8x |
| src/render.rs | 1283 | 33.95ms | 52.68ms | 9.58ms | 43.10ms | 1.3x |
| how-it-works.md | 1655 | 35.14ms | 62.56ms | 10.05ms | 52.51ms | 1.5x |

So the spawn is not the whole story. Discount bat's entire start-up and its
remaining work is still 1.3 to 2.0 times ours, because it formats and serialises
ANSI for a terminal and psychic builds ratatui spans in memory. The old code
then had to *parse* that ANSI back into spans, which is not in the bat column at
all. At the size that matters - a screenful, the common case - psychic's whole
operation costs less than bat's floor alone.

The syntax definitions take one to four milliseconds to deserialize, once, on
the preview thread while the walker is still running.

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

Now an event loop and application glue of about 900 lines, down from 2000+.

**Startup behavior:**
- Spawns a background thread to retrain the model using collected events
- Training runs asynchronously and doesn't block the UI
- Training output is appended to `~/.local/share/psychic/training.log`
- When retraining finishes, the worker does NOT reload the model. The list on
  screen must never reorder without user input; a reorder several seconds after
  launch, unprompted, is jarring (Spotlight does this and people hate it).
  Reordering during the initial fill-in is acceptable, later it is not. So the
  new model is picked up at the next moment the screen changes anyway: opening a
  file (on return from the editor) or entering a directory. Both go through
  `WorkerRequest::Reload` / `reload_model` in `search_worker.rs`, which reloads
  the click history along with the model in a single rerank.

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

**Wide terminals (≥120 columns):** Horizontal layout
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

**Narrow terminals (<120 columns):** Vertical stack layout
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
  shipped back on `WalkerDone`, `of it, rank` by the worker as the `query` line's
  `total_ms` and shipped back on `QueryUpdated`. The two the worker
  measures are passed through the response rather than re-measured on arrival,
  which would have made the pane read a channel hop slower than the log. Every
  latency in psychic - pane, log and `internal analyze-perf` - starts from the
  single `PROCESS_START` instant in `main.rs`.

  Two differences remain between the pane and `analyze-perf`, both by design:
  `analyze-perf` stops reading at `startup_complete`, so the `query` line it
  prints is the one from startup while the pane's `of it, rank` is the most
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
Why: Progressive disclosure - hide when not needed, expand for detailed debugging. Not available in narrow mode (<120 columns) where space is limited.

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
- File output: `<data dir>/app.log`, which follows `--data-dir`. The command
  line is parsed *before* logging is configured for that reason: the log used to
  be written to `~/.local/share/psychic` whatever was asked for, while
  `internal analyze-perf` and `print-log` read it from the data directory, so
  pointing psychic elsewhere split its log from the commands that read it.
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

### What the 2026-09-09 performance work is worth, end to end

Measured against `1d4d767`, the last commit before any of it. Both binaries run
from `$HOME` on a 40x120 pty, against their own copy of the same 19MB
`events.db`, with the same pinned `model.txt` - a launch retrains in the
background and would otherwise leave each version predicting with a differently
sized model. Medians; startup figures are 10 alternating trials each, keystroke
latency is 50 keystrokes each.

| | baseline | current | |
|---|---|---|---|
| keystroke -> redraw | 12.20ms | **2.45ms** | 4.98x |
| first full render | 29.92ms | **11.55ms** | 2.59x |
| - of it, the draw | 18.23ms | **3.08ms** | 5.91x |
| first results | 11.54ms | **8.28ms** | 1.39x |
| worker state ready | 8.50ms | **3.96ms** | 2.15x |
| load history | 4.92ms | **2.48ms** | 1.99x |
| load clicks | 0.56ms | **0.38ms** | 1.47x |
| filter+rank, steady state | 1.57ms | **1.05ms** | 1.50x |
| - of it, features | 0.61ms | **0.31ms** | 1.95x |
| query round trip, steady state | 1.61ms | **1.16ms** | 1.39x |
| walk complete | 68.9ms | 77.4ms | 1.12x *slower* |

Three things the raw numbers hide:

- **First results is better than 1.39x looks.** The baseline ranks 126 files at
  its first query; the current one ranks all 243, because the two-phase walk has
  already delivered the root's children. It is doing the complete job in less
  time, not the same job.
- **Per-query numbers must be compared in the steady state,** after the walk
  finishes and both hold the same 243 files. Comparing first queries compares
  126 files against 243 and reads as a regression.
- **The walk really is slower, and gitignore support is all of it.**
  Interleaved, 12 trials each: baseline 64.7ms, current 76.9ms, and the current
  binary with `--no-ignore` 61.9ms - slightly *faster* than the baseline, so the
  whole 12ms and a little more is reading and applying ignore rules. (An earlier
  block-ordered run put it at 7ms of 9ms; running all trials of one
  configuration before the next let a slow minute land on one of them. The walk
  reads the filesystem and is the noisiest thing measured here.)

  It costs nothing the user feels: first results are on screen at 7-8ms and the
  walk finishes at 77ms, so the extra 12ms lands in a window where the list is
  already up and usable.

  What it buys did *not* show up in either directory measured. From `$HOME` both
  settings index the same 243 files; in this repo it is 175 against 178. The
  walker's built-in floor already skips `target`, `node_modules`, `.git` and
  `.venv` by name, which is where the bulk of the noise lives, so `.gitignore`
  is left with whatever a project ignores beyond that - six paths in the largest
  repo on this machine. The reason to keep it is not the file count: it is that
  "what belongs to this project" should mean the same thing to psychic as it
  does to git, without psychic having to grow its own list of every build
  artifact anyone might name. The floor is a heuristic; the ignore file is the
  answer.

The baseline reproduces the profile recorded when this work was scoped, scaled
by about 0.6 - that session ran in a larger terminal. The shape is what matters
and it holds: the draw was 61% of the first full render here against 63% then,
and first results landed at 18% of walk-complete against 20% then.

**Re-measured after the ranking work**, 2026-09-10, same baseline, 8 trials:

| | before | after | |
|---|---|---|---|
| keystroke -> redraw | 9.91ms | **3.16ms** | 3.1x |
| first full render | 35.24ms | **13.71ms** | 2.6x |
| - of it, the draw | 23.42ms | **2.46ms** | 9.5x |
| worker state ready | 8.74ms | **5.91ms** | 1.5x |
| steady: features | 0.61ms | **0.24ms** | 2.5x |
| steady: predict | 0.74ms | 2.06ms | **2.8x slower** |
| steady: round trip | 1.73ms | 2.43ms | **1.4x slower** |
| walk complete | 62.9ms | 75.7ms | **1.2x slower** |

**The per-keystroke path got slower, and then got it back.** The ranking work in
September traded latency for quality: `lambdarank` settled at about 156 trees
where the classification objective stopped at 90, and predict is proportional to
trees. At that point predict was 2.06ms of a 2.43ms round trip - 85% of the cost
of ranking a query, where feature computation used to be the expensive half.

Shrinking the trees took it back and then some. With 15 leaves at `lr 0.1` the
model settles at ~70 trees, predict is 0.45ms, and the round trip is 0.91ms -
below the 1.73ms it was before any of this work, with better ranking than
either. The regression is gone; the seven points of top-1 the objective bought
are not.

One thing worth knowing before trying to win time back this way again.
`num_threads` on `predict_with_params` does nothing - measured on the real model
and real feature rows at 1, 2, 4 and 8 threads: 1.888, 1.896, 1.888, 1.889ms.
And synthetic feature values understate predict badly, because they take short
paths through the trees; the same benchmark on made-up numbers reported 0.96ms
against a true 1.89ms.

### The benchmark harness: `bench/`

```bash
./bench/run.py setup 1d4d767   # build that commit, stage a data dir per version
./bench/run.py startup 10      # startup timings, alternating, medians
./bench/run.py keystroke 50    # keystroke -> redraw
./bench/run.py walk 12         # walk time, with and without gitignore
```

`setup` builds the baseline in a git worktree under `/tmp/psychic-bench` and
gives each version its own copy of the real `events.db`, so the two runs cannot
interfere. Nothing reads or writes the real data directory except to copy out of
it. `harness.py` holds the pty plumbing and the log parsing; `run.py` is the four
commands on top.

Psychic is a TUI, so both halves of a measurement are awkward: it has to be
driven on a real terminal, and the numbers have to come back out of its own
`TIMING` lines rather than from wall-clock guesses outside the process. Five
things had to be got right, each of which produced a confident wrong answer
first:

- **Pin the model.** Every launch retrains in the background, and a retrain that
  finishes replaces `model.txt`. The baseline's `train.py` wrote a 53-tree model
  where the current one writes 84, and the resulting 1.8x difference in predict
  time looked exactly like a regression. Pinning one model for both closed it to
  1.10x.
- **Size the pty.** `script(1)` gives no control over geometry and hands out
  80x24, which nobody runs and which makes the baseline's synchronous `bat`
  spawn look four times cheaper than it is. The harness opens the pty itself and
  sets 40x120 with `TIOCSWINSZ`.
- **Keep stdin open.** An immediate EOF on the pty reads as a keypress, and
  psychic quits before it has finished starting up - so the first version of the
  harness measured nothing at all, silently.
- **Compare queries in the steady state.** At its first query the baseline ranks
  126 files and the current binary ranks all 243, because the two-phase walk has
  already delivered the root's children. Comparing those two numbers makes a 1.5x
  improvement read as a 2x regression. The last query of a run, after the walk,
  has both at 243.
- **Interleave, do not block.** Running all trials of one configuration before
  the next lets a slow patch on the machine land entirely on one of them. That
  inverted the walk result once already.

Both versions still redraw on a tick, about ten writes a second when idle, so
the keystroke measurement waits for a quiet moment before starting its clock:
that puts it just after a tick redraw, which makes the next write the one the
keystroke caused.

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
- `ignore` - Directory traversal that respects .gitignore (ripgrep's)
- `rusqlite` - SQLite (bundled feature for static linking)
- `lightgbm3` - LightGBM inference
- `anyhow` - Error handling
- `jiff` - Timestamps
- `syntect` + `two-face` - syntax highlighting in process, with bat's syntax and theme set
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
psychic internal preview <path> [--lines N] [--repeat N] [--show]

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
