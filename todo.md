## now

### 2026-09-08 review: remaining findings and suggested order

**Working notes for whoever picks these up.**

- *Design rule not written down anywhere else:* the visible result list must
  never reorder without user input. Reranking during the initial fill-in is
  fine; a reorder seconds later with no keypress is not (Spotlight does this
  and it is hated). Anything that changes ranking must be tied to a user
  action. This is why the retrained model is loaded on click/navigation, not
  when training finishes.
- *Workflow* (from llm.md): `just build` (release binary; the user's daily
  symlink points at it, never plain `cargo build`), `cargo test`,
  `cargo clippy`, then update how-it-works.md to describe the *current*
  state, not the history. Asserts for preconditions, not debug_asserts.
  Structs over tuples. Each todo item above ends with its own check.
- *Never test against the real data dir.* Make an isolated copy:
  ```sh
  S=/tmp/psychic-sandbox; mkdir -p $S/.local/share/psychic
  sqlite3 ~/.local/share/psychic/events.db ".backup $S/.local/share/psychic/events.db"
  cp ~/.local/share/psychic/{model.txt,model_stats.json} $S/.local/share/psychic/
  ```
  `.backup` is a consistent copy that does not touch the live file. Then run
  with `HOME=$S` so both the data dir and app.log land in the sandbox
  (the logger currently ignores --data-dir, see B7). Note `uv run train.py`
  under a fresh HOME builds a cold uv cache; fine, it is background.
- *Headless TUI run for timing:* the TUI needs a tty; `script` provides one
  and forwards piped stdin, and 0x03 is Ctrl-C = quit:
  ```sh
  ( sleep 2; printf '\x03' ) | HOME=$S EDITOR=true script -q /dev/null \
      ./target/release/psychic --no-preview >/dev/null 2>&1
  grep -o '"op":"[a-z_]*","ms":[0-9.]*' $S/.local/share/psychic/app.log
  ```
  Run from the directory you want walked. `psychic internal analyze-perf`
  prints the last TUI session's startup breakdown. The debug pane (Ctrl-O)
  shows the same numbers live.
- *Baseline to beat* (real session from `~`, 2026-09-08): first paint 3.3ms,
  first results 18.4ms, first full render 49.7ms (draw 31.3ms, bat 15.7ms),
  walk complete 92.6ms. From this 44-file repo in the sandbox: walk complete
  11ms, first results 13ms. Per keystroke filter+rank ~2ms for ~240 rows.
- *Real data shape* (for judging whether something matters): 83k events =
  80k impressions, 1.1k clicks, 2.1k startup_visits, 7 scrolls; 971
  sessions; 208 distinct engaged paths; model.txt 180 trees x 31 leaves;
  registry ~240 rows from `~` after shallow mode, ~170 historical.
- *Two invariants to keep while refactoring the worker:* `FileId` is an
  index into `file_registry`, held by `filtered_files`/`file_scores` and by
  pages the UI has, so never remove registry entries except in `change_cwd`
  where `filter_and_rank` is re-run immediately after; and every response
  to the UI carries the `query_id` of the request it answers, and the UI
  drops anything older than its current id.
- *Tests that look like coverage but are not:* `test_basic_feature_generation`
  and `test_ranker_basic` skip themselves when their fixture files are
  absent (always, in CI and locally). Do not count on them.

From a full read of the code plus measurements against the real data dir
(62MB events.db, 971 sessions, 80k impressions, 31MB app.log). The three
sections above came out of the same review. Everything below is still open.

**Suggested order.** Each step is independent; this is by what the user
feels, then by risk removed, then cleanup.

1. Two-phase walker (P1). Biggest startup win in the common `p` from `~` case.
2. Preview thread (P2). Removes the largest chunk of the first full render
   and the per-keystroke stalls.
3. Dropped-request bug (B1) and cwd-row bugs (B2). Silent, user-visible.
4. Model reload item (above), blend weight item (above), training item
   (above). Behavior of the ranking; do after the walker/preview work so
   perf numbers are stable to compare.
5. Database diet (P8) and logging cut (P4). Disk and per-keystroke syscalls.
6. Startup syscalls (P3), allocation-free lookups (P5), get_slice (P6),
   idle wakeups (P7). Small, safe, mechanical.
7. Bugs B3-B7, then simplifications S1-S8, then the docs pass (S9).

**Measured startup, last real session launched from `~`:**
first paint 3.3ms; first results 18.4ms; first full render 49.7ms, of which
the draw took 31.3ms because it spawned `bat` (15.7ms) synchronously; walk
complete 92.6ms. In an isolated data dir from this 44-file repo the walk
completes at 11ms, so the ~90ms is specific to the shallow-mode restart.
Filter+rank is ~2ms per keystroke and is not the problem.

#### Performance

- **P1. Two-phase walk.** DONE (2026-09-09). The walker now sends the root's
  own children first (one readdir, `min_depth(1).max_depth(1)`) followed by a
  new `WalkerMessage::ChildrenDone`, then collects everything deeper
  (`min_depth(2)`) and hands it over only if it comes in under the threshold.
  Past the threshold the deep pass is abandoned and the children stand alone,
  so a tree too big to index costs one walk instead of two and shows its
  children immediately instead of after both walks. `ChildrenDone` bypasses
  the worker's 200ms debounce the way `AllDone` does, without which publishing
  early would have bought nothing.
  Measured, launch to the file list appearing, median of 5: in `~` 80.9ms ->
  13.3ms; in this repo 13.7ms -> 12.3ms (small tree, never hit the threshold,
  so nothing to win). `walker_complete` itself barely moves (65-69ms either
  way): the depth-1 re-walk that was removed was always the cheap half.
  Two bugs fixed on the way, both found while rewriting:
  * An interrupted walk `try_recv`d the `ChangeCwd` that interrupted it and
    dropped it, then the outer loop blocked on `recv` for a command already
    delivered - so navigating during a long walk never walked the new
    directory. `walk_directory` now returns the command that stopped it.
  * `filter_entry` was applied to the root, so launching inside a directory
    called `target` (or `.git`, `node_modules`, `.venv`) showed an empty
    screen. Depth 0 is now exempt.
  Follow-up if a large-but-under-threshold project ever feels slow: stream the
  deep pass too, in batches, and add a message telling the worker to drop
  everything below depth 1 when the threshold is hit. Not done because it
  would make results appear and then vanish in exactly the `~` case this was
  about.
- **P2. Previews: off the UI thread and in process.** DONE (2026-09-09).
  New `preview.rs`: a thread that turns a path into styled text and keeps only
  its newest request, plus a `PreviewState` holding what is shown, what was
  asked for, and the scroll offset. The UI shows a preview only when the path
  it was generated for is the path selected now, so no file's contents ever
  sit under another's name. `render.rs` records the pane width; the main loop
  asks for the preview after the frame is drawn.
  `bat` and `eza` are gone: highlighting is `syntect` (the library bat is
  built on), a listing is a `read_dir`, and `ansi-to-tui` is dropped because
  nothing produces ANSI to parse any more. History mode uses the same thread,
  where it used to spawn `eza` on every frame including every tick, uncached.
  Measured, moving the selection one row: median 15.90ms -> 0.55ms, p90
  20.84 -> 2.18, max 23.58 -> 2.33. First full draw 31.6ms -> 4.2ms. The
  syntax set costs 3ms to load, once, on the preview thread. Binary grows
  10.0MB -> 12.7MB for the syntax definitions.
  Fixed along the way: **the binary-file display corruption** (todo: "display
  is broken if we scroll past a binary file"). The real fix is not sniffing
  but sanitising - `path_display::printable` turns tabs into spaces and every
  other control character into a dot, and every string that reaches a cell
  goes through it: file contents, directory entry names, file list rows, the
  path bar. Ratatui passes cell contents straight through, so an ESC in a file
  *or in a filename* was an instruction the terminal obeyed. The NUL sniff
  still names an obvious binary, but it only looks at the first 8KB and is no
  longer what keeps the display safe.
  Also: previews are capped at 5,000 lines and 4MB (the old code read whole
  files into memory as styled text), sliced to the visible window at draw time
  so a long preview is not cloned every frame, and scroll is clamped so it
  cannot walk off into the distance.
  Generation has **two states**, as the `bat` version did: a screenful while
  unscrolled, then the whole file in one pass on the first scroll, after which
  scrolling is free. The first cut generated whole files up front and a large
  markdown preview took 150ms; the second cut grew the budget as the user
  scrolled, which is worse still - syntect state means every pass restarts at
  line one, so a growing budget costs ~2x the work in a series of hiccups. Syntect is built with `oniguruma`, not `fancy-regex`: measured on
  markdown it is ~5x faster (median 11.81ms -> 2.53ms, worst 102.57 -> 25.79)
  and the binary is *smaller* (12.7MB -> 11.2MB). Syntaxes and theme come from
  `two-face` (bat's set): syntect's own defaults have no TOML, TypeScript or
  Dockerfile, and its themes render markdown headings as near-invisible grey.
  `psychic internal preview` times the generator with nothing around it; against
  bat we are 1.3-2.0x faster even discounting bat's entire start-up, and at a
  screenful our whole operation costs less than bat's floor alone.
- **P3. Startup syscalls and sequencing.** DONE (2026-09-09). One `stat` per
  historical path instead of `exists()` + `canonicalize()` + `metadata()`, and
  the history load now runs beside the ranker load rather than after it (the
  ranker stays put: `Booster` is not `Send`). Verified against the real
  database that no stored path differs from its canonical form by more than a
  trailing slash, which `Path` ignores. Measured, median of 7, same 177
  registry entries either way: `load_historical_files` 4.06ms -> 2.37ms,
  `worker_state_new_total` 8.56ms -> 4.49ms, `first_query_complete` 12.27ms ->
  8.63ms. About half from each change; with only the syscall fix the total was
  6.79ms. `get_file_metadata`/`FileMetadata` fell out as dead weight.
- **P4. Timing instrumentation: keep every number, cut the allocations and
  the line count.** `ranker.rs::compute_features_with_timing` allocates 15
  String keys and a hashmap per file per keystroke and collects a Vec of
  them; then each query emits ~23 log lines (8 op lines + 15 per-feature
  lines), and fern flushes per record, so each is a write syscall from the
  worker before the response goes out. `Instant::now` itself is tens of ns
  and is NOT the cost; per-use timing data stays. Fix: (a) per file, a
  `[Duration; N]` indexed by registry position (N = `FEATURE_REGISTRY.len()`),
  summed inside the rayon fold/reduce rather than collecting per-file maps;
  (b) one `TIMING` line per query, a single JSON object with filter_ms,
  simple_ms, features_ms, predict_ms, blend_ms, total_ms, count and a
  `per_feature: {name: total_ms}` map, and teach `analyze_perf.rs` the
  nested shape; (c) in `worker_thread_loop` send `QueryUpdated` first, then
  log, so the write syscall is after the user's results are on their way;
  (d) capture the session id once in the fern formatter closure in main.rs
  instead of `std::env::var` per line, which also removes the `unsafe
  set_var`. Consider a size cap/rotation for app.log (31MB now).
- **P5. Query-constant lookups allocate per file.** `ClicksForThisQuery`
  and `EngagementsInEpisodeWithQuery` build a `(String, String)` key per
  file. `FuzzyScore` constructs a new `SkimMatcherV2` per file and redoes
  the match `filter_and_rank` already did. `compute_simple_score` allocates
  the path string. Fix: key the two query maps as `query -> path -> events`
  and look the query up once per `rank_files`; pass `fuzzy_score` in via
  `FeatureInputs` (training computes it, inference reuses the filter's).
- **P6. `get_slice` is O(results) per row.** `search_worker.rs` ~line 708:
  `file_scores.iter().find(..)` for each of 128 rows. `filtered_files` is
  built from `file_scores` in the same order; index directly and delete
  `filtered_files`.
- **P7. Idle wakeups and input latency.** Three loops poll. The input one is
  DONE (2026-09-09); the other two are independent of it and still open.
  - *DONE: the input thread.* Rewritten as `src/tty_input.rs`: it waits with
    `libc::poll` on three descriptors - the terminal, a self-pipe for "stop
    reading", and a second self-pipe fed by a SIGWINCH handler - so it can
    block with no timeout and still be interrupted. `crossbeam` was used for
    nothing else and is gone. `pause()` now returns only once the thread has
    acknowledged it stopped, so the two 50ms sleeps in `input.rs` are gone,
    and it returns a guard whose Drop resumes the thread. The two
    near-identical `suspend_tui_*` functions are now one
    `suspend_tui_and_run`, which also pops the keyboard enhancement flags
    before suspending (they were pushed on every resume and never popped).
    The thread is shut down and joined explicitly in main before `drop(app)`,
    like the worker: it logs as it exits, and App's `log_receiver` field is
    declared before `input`, so leaving it to Drop closed the logging channel
    first and printed "Error performing logging" over the restored terminal
    (reproduced on 3 of 5 runs, 0 of 5 after).
    The reasoning is written up in how-it-works.md under "The input thread".
  - *Measured, old binary vs new, same pty, 30 trials:* keystroke to redraw
    median 6.87ms -> 2.00ms, mean 7.50 -> 1.74, p90 12.90 -> 3.17, max
    14.38 -> 3.64. Resize to redraw 4.8-13.1ms -> 1.0-2.2ms. The old numbers
    are the ~2ms of real work plus a uniform 0-10ms wait for the next poll.
  - *Correction to the original finding:* the idle CPU claim was wrong. Both
    binaries use 0.02s of CPU over 20s idle (`ps -o time`), i.e. the 100
    wakeups/second cost nothing measurable - each was a cheap `kevent` that
    found nothing. The wins here were latency and correctness (a design in
    which the eaten-keystroke bug cannot recur), not CPU. Do not expect the
    two items below to show up in a CPU measurement either; do them for
    simplicity, and because the tick one is a prerequisite for anything that
    wants the UI to be genuinely idle when idle.
  - *Still open - worker:* `recv_timeout(5ms)` in `worker_thread_loop` exists
    only because the walker has its own channel. Give the walker a clone of
    the worker request sender and add `WorkerRequest::Walker(WalkerMessage)`;
    the worker then blocks on one `recv()` with no timeout. Keep the
    FilesChanged debounce logic.
  - *Still open - tick:* send ticks only while something animates, i.e. the
    marquee path overflows the path bar. Note that the tick currently also
    serves as the fallback that makes a resize take effect (ratatui
    re-reads the terminal size on every `draw`); that fallback is no longer
    load-bearing now that the input thread wakes on SIGWINCH itself.
  - *Still open:* each redraw builds the debug pane text even when the pane
    is hidden, and deep-clones `PreviewManager` twice (render.rs
    `ctx.preview.clone()` then `text.clone()`); gate on visibility and pass
    `&mut`.
- **P8. Database diet.** DONE (2026-09-09). `running_processes` and
  `shell_history` are no longer collected and their columns are dropped;
  `Database::migrate` does it on any open and `VACUUM`s once (83ms) when it
  actually dropped something. The full index is replaced by a partial one on
  the engagement actions, plus a narrow index on `action` alone for the debug
  pane's histogram, which a partial index cannot answer. `load_clicks` states
  the index predicate redundantly, without which SQLite cannot prove the
  implication and silently scans; `db::plan_tests` runs EXPLAIN QUERY PLAN over
  every query psychic issues, including one pinning why that clause is needed.
  Impressions are one transaction rather than two dozen. Measured on a copy of
  the real database:

  | | before | after |
  |---|---|---|
  | file | 67.1 MB | 19.1 MB |
  | sessions table | 40.2 MB | 0.1 MB |
  | indexes on events | 9.9 MB | 2.0 MB |
  | `load_clicks_total` | 0.48ms | 0.47ms |
  | `load_historical_files` | 2.50ms | 1.93ms |
  | `internal summarize-events` | 14.99ms | 12.47ms |

  Nothing got slower. Still open from the original item: `session_id` is a u64
  stored as TEXT.
- **P9. Measure `num_threads=8` vs 1 in `predict_with_params`.** 180 trees,
  ~200 rows, 1.2-1.4ms; OpenMP fork/join likely exceeds the work.
- **P10. Optional: cache query-independent features per registry entry.**
  12 of 15 features do not depend on the query. Irrelevant at 200 files,
  ~25ms/keystroke at the 8000 the shallow threshold allows.

#### Bugs

- **B1-B4. DONE (2026-09-09).** One commit each.
  * **B1** `stop the worker throwing away requests it did not expect`.
    Draining kept the newest `UpdateQuery` and discarded anything else it
    found, so typing then pressing Enter on a directory while the worker was
    busy ate the `ChangeCwd`. Requests are queued and processed in order now,
    collapsing only *consecutive* query updates.
  * **B2** `give the current directory a row after navigating`. There is one
    `display_name_for(path, root)` and one `ensure_root_row()`, called by both
    `new` and `change_cwd`. A directory you had visited before is also
    reclassified from history to where-you-are. `change_cwd` canonicalises its
    root like `new` does.
  * **B3** `measure text in columns`. The debug pane's log truncation was a
    byte slice and panicked the UI on `end byte index 57 is not a char
    boundary; it is inside 'é'`. Same mistake in the cursor position
    (`query.len()`) and the file list's timestamp padding. `path_display` gained
    `display_width` and `truncate_to_width`; the test reproduces the original
    panic.
  * **B4** `notice when the search worker dies`. The main loop checks
    `worker_has_died()` after each event and exits with a message; the terminal
    is restored first. `Feature::compute` returns `f64` rather than a `Result`
    no implementation could fail, which removes the `.expect()` that sat inside
    the rayon loop.
- **B5. Train/serve skew.** Training computes `is_dir` by stat-ing today's
  filesystem (`features.rs` `full_path.is_dir()`, 80k syscalls per retrain),
  so a deleted directory trains as a file. Add an `is_dir` column to events,
  set from `DisplayFileInfo.is_dir` at log time, and read it back. Also
  impressions log the top 25 by rank, not the rows on screen
  (`num_results_to_log_as_impressions`); log exactly the visible rows using
  `visible_list_height`, so labels reflect what was seen. `is_from_walker`
  in `FeatureInputs` is redundant with `is_under_cwd`; drop it.
- **B6. Docs say LambdaRank; train.py is `objective: binary` + auc.**
  Episodes only drive the split. Pick one and make README/how-it-works say
  it. (Binary is fine; the episode machinery is then just grouping.)
- **B7. Small ones.** ~~Logger path ignores `--data-dir`~~ DONE: the CLI is
  parsed before logging is configured, so the log goes to the data directory
  the rest of the program uses. The session id is captured in the formatter
  rather than round-tripped through an environment variable, which removed the
  program's only `unsafe` block. Two `get_eza_flags` (preview.rs: <100 cols,
  three flags; render.rs: <80 cols, two flags). Layout breakpoint is 120 in
  code, 100 in docs. Suspend-for-editor uses a 50ms sleep as a race guard;
  have the input thread ack the pause on a channel. `test_basic_feature_generation`
  asserts a stale header and is skipped when test/events.db is absent;
  `test_ranker_basic` likewise; the `FileInfo` tests in search_worker.rs and
  the field tests in ui_state.rs assert values they just set. Delete or fix.

#### Simplifications

- **S1. Make render pure.** `NormalRenderContext` copies ~20 fields out of
  `App`, `RenderUpdates` copies 5 back, only because render mutates the
  preview and the marquee. Advance the marquee in the Tick handler, generate
  previews on the preview thread (P2), and render takes `&App`. Deletes the
  duplicate `get_file_at_index` and `compute_scroll`; the non-override
  branch of `App::update_scroll` is already dead.
- **S2. One response path in the worker.** Five arms each do "set id,
  mutate, filter_and_rank, get_page(0, 128), send QueryUpdated". One helper,
  and use `app::PAGE_SIZE` instead of the literal 128.
- **S3. Collapse parallel structs.** `FeatureClickIndexes` is `&ClickData`.
  `FileCandidate` duplicates `FileInfo` fields; rank over `&[&FileInfo]` (or
  store fuzzy score on a small wrapper). `Episode` is a Vec<String> with a
  contains check; inline into `Analytics`. `Subsession.created_at` is a
  jiff Timestamp used for a 200ms debounce; use `Instant`.
- **S4. One database open per thread.** DONE (2026-09-09), with the caveat
  that it bought no measurable speed. `Ranker::load_clicks` no longer opens a
  raw `Connection` with its own copy of the pragmas: the query moved to
  `Database::engagements_since`, beside the index it needs and the plan test
  that checks it. The worker holds one connection for its lifetime instead of
  opening one per reload and per hide. `App::new` reads the hidden prefixes
  from the connection it already has rather than `spawn` opening a second one
  on the same thread, which removes one open per launch (5 -> 4). Schema
  creation and migration now run once per database file per process rather
  than down every connection (`PREPARED`; `:memory:` excluded, since each such
  connection is its own database and caching it left the second one empty -
  eight tests caught that).
  Measured, median of 5, real database: `worker_state_new_total` 4.42ms ->
  4.57ms, `first_query_complete` 8.87ms -> 8.55ms, RSS 291.7MB -> 292.4MB.
  All noise: the schema work was ~0.2ms per open after the first, not the
  ~0.7ms `App::new`'s old log line suggested. Do this for the tidiness, not
  the speed.
  `lsof` handles on events.db went 3 -> 4 while *live connections stayed at
  2*. The extra one is a descriptor SQLite parks rather than closes when
  another connection holds a lock (sqlite3.c: "that would clear those
  locks"), because POSIX advisory locks are per-process. Verified it is not
  the worker's held connection by building a variant without it: still 4.
  Explained in how-it-works.md under "Why the same database is opened several
  times".
- **S5. Deduplicate terminal suspension.** `suspend_tui_for_editor` and
  `suspend_tui_and_run_shell` are identical except the Command;
  `cleanup_terminal` is a third copy of the teardown. One
  `with_tui_suspended(app, terminal, |..| Command)`.
- **S6. Walker cleanup.** DONE with P1: the buffer, the restart, the duplicated
  command-check loop and the unreachable deep-mode `MAX_FILES` guard are gone.
- **S7. `context.rs`** runs five `sh -c` pipelines per launch; nothing reads
  any of it. Keep `cwd`, delete the rest (see P8).
- **S8. `check_and_log_impressions`** builds the 25-row Vec on every event
  before checking `already_logged`; check first.
- **S9. Docs drift in how-it-works.md:** "5 threads" (there are 7+),
  "~600-line main.rs" (830), `unsafe impl Send` ranker (no longer), 100-col
  breakpoint, `get_eza_flags` in ui_state (does not exist), LambdaRank,
  `Analytics`/`App` method lists, and "Streams results" for the walker.

### 2026-09-08 feature ideas, ranked

What the 15 current features cover: match quality (fuzzy_score,
filename_starts_with_query); positive engagement counts (clicks 1h/24h/7d/30d,
clicks_for_this_query, engagements_in_episode_with_query,
clicks_last_week_parent_dir); file properties (modified_age,
modified_last_24h, log_file_size, is_dir, is_hidden, is_under_cwd). Top three
by gain, since the time split stopped rewarding memorised file sizes:
clicks_for_this_query, fuzzy_score, filename_starts_with_query.

Every feature below must be computed identically in `ranker.rs` (inference,
from the in-memory click maps) and `features.rs` (training, from the
Accumulator's fold over time-sorted events), so add the index to both
`ClickData` and `Accumulator`. Add to `feature_defs/implementations.rs` and
`registry.rs`; update `test_feature_computation`'s expected vector.

**Ranked.** Ordered by expected signal per unit of work.

1. **Directory visits.** 2,099 `startup_visit` rows from the zsh `chpwd`
   hook are never read by any feature; `load_clicks` filters to click and
   scroll. README claims visits teach directory preferences; nothing makes
   that true. Add `visits_by_dir: map<path, Vec<ts>>` loaded alongside
   clicks (add 'startup_visit' to the load query, into a separate map, do
   NOT count them as clicks or in total_clicks). Features
   `visits_last_7_days`, `visits_last_30_days` (0 for files). Strongest
   signal for `pd`, zero collection cost.
2. **Seconds since last click**, `ln(1 + now - last_click_ts)`, 
   large constant when never clicked; monotone decreasing. Same for parent
   dir. The windows count clicks but cannot tell 1 minute ago from 50.
   Uses existing `clicks_by_file` (take max timestamp).
3. **Query length** (`query.chars().count()`). Fuzzy scores scale with it
   and clicks_for_this_query only means anything past a few chars; today
   the tree infers it from fuzzy_score magnitude. Trivial.
4. **Collection change: record rank position on impressions.** Add
   `rank INTEGER` to events, set in `log_impressions` from the row's index.
   Cannot be backfilled. Enables position-aware weighting/features later
   (unclicked at #1 is far more negative than unclicked at #24).
5. **Collection change: record `is_dir` on events** (see B5). Also cannot
   be backfilled; closes the train/serve skew.
6. **Click-through rate per file** (preferred over a raw impression count:
   easier for the tree to use). Impressions are the training *labels* but
   no *feature* says how often this file was shown without a click, so the
   model has per-file memory of positives and none of negatives. Two
   columns: `impressions_last_30_days` (confidence) and a *smoothed* CTR,
   because raw clicks/impressions is 1.0 for one-shown-one-clicked and 0.5
   for 100-of-200, which is backwards:
   ```
   ctr = (clicks_30d + k * global_ctr) / (impressions_30d + k)      k = 5..10
   ```
   `global_ctr` = total clicks / total impressions in the same window
   (about 1.4% on today's data: 1,113 / 80k), computed once at load time
   and once per fold position in training. An unseen file sits at the
   global rate; an often-shown never-clicked file drifts to ~0. Optionally
   the same per (query, file). Training side: `impressions_by_file` (and
   running totals) in the Accumulator, counted before the current
   impression. Runtime: 30 days of impressions per path at startup, either
   `SELECT full_path, COUNT(*) ... WHERE action='impression' AND timestamp
   >= ? GROUP BY full_path` (needs its own partial index on impressions,
   or accept a scan of one month) or a small counts table maintained at
   write time. Measure the startup cost first. Rank position (item 4) will
   later let this become position-debiased.
7. **Clicks under this directory** (7d, 30d). A dir row's own clicks only
   count Enter on it; clicks on anything beneath it are the better signal.
   Build at load time by bumping every ancestor of each clicked path
   (bounded to, say, 8 levels). 0 for files.
8. **Clicks for this query in this directory.** Generalizes
   clicks_for_this_query to siblings: map keyed (query, parent_dir).
9. **Modified since last click** (binary: mtime > last click ts). "Something
   new here" for files you have opened before.
10. **Extension click share.** Fraction of the user's clicks (30d) on files
    with this extension; captures "never opens .lock/.png" without
    memorizing files. Map ext -> count at load time.
11. **Depth below cwd** (component count; large constant for historical
    files outside cwd). Shallow things get clicked more; helps `~`.
12. **Mentioned in recent shell commands** (binary): filename or its dir
    appears as a token in the last 10 shell commands of this session. To
    make it trainable and less of a privacy problem, change `context.rs` to
    store only extracted path-like tokens per session (split on whitespace,
    keep tokens containing '/' or '.', drop the rest), not the raw commands.
    Read the session's tokens at startup into a HashSet. Speculative; try
    after 1-6.
13. **Git status flags** (`is_git_modified`, `is_git_untracked`): strong for
    developers but needs a background `git status --porcelain` per repo at
    walk time and belongs with the "respect .gitignore" item. Later.

**Not worth it.** `ps` output: turning a process list into a per-file
signal needs lsof-style work per process, and the plausible signal ("is my
editor open on this project") is weak. Drop it (P8). Raw shell history as a
blob: no; see 12 for the salvageable part. Time-of-day / day-of-week: needs
a timezone and was removed for cost and skew reasons (see how-it-works).

**Hygiene while in there.** `is_from_walker` in `FeatureInputs` is
redundant with `is_under_cwd`; `modified_last_24h` is a threshold of
`modified_age` the tree can learn itself (harmless, but drop it if you want
a shorter vector). After adding features, rerun training with the time
split (see training item) and compare validation AUC and per-feature gain;
drop anything that does not move it.

distribution:

- full text search mode using rg.

- do something about the size of the events db
- if a historical file or dir no longer exists then filter it out
- audit the whole codebase for modularity. can we refactor extract something into a module, which can then be expect tested? right now its a big ball of very IO heavy code that makes it difficult to test. maybe the overall state logic and keypress logic? maybe the page caching logic? maybe the logic that when walker is finished it sends an AllDone message? maybe the logic that historical files in cwd still need to shown in filter view?
- when history is filtered, suppose number of items becomes less than selected index, then selected index should become 0 so the top item is automatically becomes selected.
- display is broken if we scroll past a binary file and it gets previewed
- if we hit up while file walker is still walking then it shows loading and we end up in some strange middle of the results. instead we should remember our scroll position as -1 and reevaluate that when results are updated.
- pick some good keybindings for going to top, and paging up and down the results
- watch the cwd + all historical files; if mtime changes then update. more generally, our internal file data structure must be kept up-to-date with the filesystem. Right now this works because the file list view polls the filesytem for file metadata every frame or something awful like that. But the fixes below will break that.
- until filewalker is done, don't bother sorting and calculating features? idk. or really, make sure we don't recalculate features? hmm. maybe we want to divide features into query-dependent and query independent?
- watch the cwd. if new files added then add them.
- maybe add a slight linear term?
- try fitting a linear or logistic regressor esp on modified time and num clicks and last time clicked
- hit enter to open Preview or whatever default thing is configured
- can we preview PDFs and images in the terminal?
- make sure that subnet and gateway are being logged properly. generally, look at the db and see what's up, is it missing important data?
- is_in_dotdir would need to be logged at query time i think. can't be done at feature gen time because what if FS changes
- maybe try random forests?


## notes on performance

### could maybe do click aggregation in SQL rather than in the rust code

```
SELECT
    full_path,
    COUNT(*) AS clicks_last_30_days
  FROM events
  WHERE action = 'click'
    AND timestamp >= strftime('%s','now') - 30 * 24 * 60 * 60
  GROUP BY full_path
  ORDER BY clicks_last_30_days DESC;`
```

And cache these in sqlite as a table. Would need to refresh this table at startup, and also lock the table to do the refresh. sqlite doesn't have materialized views. Or just have all this be a view, maybe it's still worth it, idk, haven't measured.

### 2025-10-20

Trying to make startup faster. Time to fully rendered initial screen.

Try commenting out preview. Actually add an argument to turn off preview, so we can always measure startup time without preview.

Added arguments to turn off various features to see if they were the culprit.

The real culprit: a 100ms per frame sleep/timeout. Fixed by refactoring to use a single event stream, and fast polling done on just the input events.

### 2025-10-10

On 5000 files, `update_filtered_files` takes 150-200ms. All of that comes from `rank_files`. The majority of THAT is from feature computation, not actually from running the model.

Update: with some optimizations, model is actually more expensive than computing features.

Feature computation can be sped up in a bunch of ways:
* Divide features into those that depend on the query and those that don't. The ones that don't depend on query don't need to be recomputed on query change.
* Two of them iterate over clicks_by_file. We can probably move parent dir click aggregation, and last 30 days click counting, into a precomputation that happens once at db load time and computes click counts by iterating over all clicks once. Then the features just do lookups? Idk how to do this elegantly though, where adding each feature doesn't become a big chore at db load time. Ah! Maybe each feature also gets to define its own Agg type that can do stuff at db load time? 

Helix file picker is a lot faster. Maybe implement a mode that doesn't sort or run the model or compute features, to make sure it's somewhat as fast.


## not now, maybe never
- use tracing subscriber crate so we can have nice spans of time and we can maybe visualize and optimize idk
