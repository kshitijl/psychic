## now

Ordered by what a user would feel, then by risk removed. Every item ends with
its own check. Read `llm.md` first: every change gets benchmarked against its
parent, and a ranking change gets `./bench/model.py compare --seeds 5` as well.

### 1. Click-through rate: built, measured, not shipped

Built in full - `impressions_last_30_days` and a smoothed
`click_through_rate`, indexed on both sides, four tests - and then not
shipped, because the startup cost is real and the quality gain is not.

The model *likes* it: `click_through_rate` came out **3rd of 22 by gain** at
14.7%, `impressions_last_30_days` 7th at 4.5%. But over four seeds a side:

| | before | after |
|---|---|---|
| AUC | 0.9421 | 0.9491 |
| top-1 | 0.7968 | 0.7999 |
| MRR | 0.8671 | 0.8664 |

top-1 +0.0031 against a standard error of 0.0034, and MRR flat. AUC is the
only clear movement, and AUC is a pooled metric we deliberately stopped
optimising when the objective became `lambdarank`.

The cost is not small. Impressions are 96% of the table and deliberately not
in the engagement index, so counting them is a scan:

| | before | after |
|---|---|---|
| load clicks | 0.62ms | **14.62ms** |
| worker state ready | 3.46ms | 17.40ms |
| first results | 6.71ms | **20.81ms** |

A partial index on `action='impression'` roughly halves the query, at 27.2MB
of database against 18.6MB - re-adding most of what P8 removed for the same
reason.

**What would change the answer**, in order of how much it would change it:

1. **Position debiasing.** The rate is measured over rows *this ranker chose
   to show*, which is a feedback loop: a file the model ranks low is shown
   less, so its rate stays low. `events.rank` started being collected on
   2026-09-10 and is the input for correcting that. Retry when there are a
   few months of it.
2. **A cheaper count.** A counts table maintained at write time, if the
   30-day window can be aged out correctly, or loading the counts off the
   critical path after the first query - at the price of the first query not
   having them.
3. **More data.** 1,173 clicks over 10k impressions in the window is a thin
   base for a per-file rate.

The code is in `git show` for the commit that reverted it; rebuilding from
that is an hour, and the measurement above is the thing worth keeping.

### 2. S8, and the tests that only look like tests

`check_and_log_impressions` builds the 25-row Vec on every event before checking
`already_logged`; check first.

`test_basic_feature_generation` returns early unless `test/events.db` exists and
`test_ranker_basic` unless `output.txt` does. Neither file is in the repo, so
both have always passed by doing nothing, and `test_basic_feature_generation`
additionally asserts a stale CSV header. Delete them or give them fixtures - the
trained-model test (`search_worker.rs`) is the model to copy: it builds its own
data, runs the real thing, and takes seven seconds.

### 4. Smaller, still open

- **P10.** Cache query-independent features per registry entry. 12 of 20 features
  do not depend on the query. Irrelevant at 244 files and 0.24ms; it would
  matter at the 8,000 the shallow-mode threshold allows. Revisit only if
  someone launches in a big-but-under-threshold tree and it feels slow.
- **Feature: modified since last click** (binary, `mtime > last_click_ts`).
  "Something changed here since you last looked." Cheap: both numbers are in
  memory already.
- **Feature: mentioned in recent shell commands.** Speculative, and needs
  `context.rs` to store extracted path-like tokens rather than raw commands -
  which is a privacy improvement in its own right. Do after the CTR work.
- **Feature: git status flags.** `is_git_modified`, `is_git_untracked`. Strong
  for developers, but needs a background `git status --porcelain` per repo at
  walk time. Belongs with the gitignore machinery, which now exists.

### 3. Check how fragile the September tuning is, and redo it in a year

Every choice made in September - objective, tree size, learning rate, half-life,
which features are in - was measured against one history at one moment: about a
year of one person's clicks, 1,200 of them. how-it-works.md has the table of
what was chosen and what it beat, under "Where these choices came from".

**The experiment that has not been run.** Take windows of the history - last
week, last month, last six months - and slide each along the timeline, redoing
the key comparisons inside each. A choice that wins in every window is a
property of the problem; one that only wins in the windows with a few hundred
clicks is a property of *this much* data. `bench/model.py` already takes a CSV
and a set of folds, and restricting to a window is a filter on `episode_id`, so
this is the same shape as the rolling-origin folds it already runs.

Most likely to be window-dependent, in order: `num_leaves` and `learning_rate`
(more data supports a bigger model), the 180-day half-life (this history is
back-loaded - 54% of rows are 240+ days old - so "old" means something unusual
here), and click-through rate, rejected partly for having too thin a base.

**And redo the tuning after another year of use.** By late 2027 the database
should hold several times the clicks it does now, which is the axis these
choices are most sensitive to.

## measurement discipline

Three things this repo learned the hard way in September. All three are in
`llm.md`; they are repeated here because they are what makes the numbers above
trustworthy.

- **One seed cannot see one feature.** The same feature set under eight seeds
  spans 0.027 of top-1. Anything smaller, measured once, is noise:
  `./bench/model.py compare --seeds 5`.
- **Early stopping moves underneath a comparison.** A feature that shifts the
  validation curve changes the tree count, and then two differently sized models
  are being compared. `./bench/model.py compare 120` pins it.
- **The timing harness has five ways to lie**, all of them documented in
  `bench/harness.py`: an unpinned model, an 80x24 pty, an immediate EOF on
  stdin, comparing first queries that rank different numbers of files, and
  running trials in blocks rather than interleaved. Read it before trusting a
  surprising result.

## done

Detail lives in the commit messages; this is the index.

**September 2026 performance push** (`1d4d767` to `8c0f34e`). Measured end to
end on 2026-09-10, current against the commit before any of it, from `$HOME` on
a 40x120 pty:

| | before | after | |
|---|---|---|---|
| keystroke -> redraw | 9.91ms | **3.16ms** | 3.1x |
| first full render | 35.24ms | **13.71ms** | 2.6x |
| - of it, the draw | 23.42ms | **2.46ms** | 9.5x |
| worker state ready | 8.74ms | **5.91ms** | 1.5x |
| - load history | 4.91ms | **2.41ms** | 2.0x |
| first paint | 1.89ms | **1.35ms** | 1.4x |
| steady: features | 0.61ms | **0.24ms** | 2.5x |
| first results | 11.75ms | 11.27ms | 1.0x, on 244 files vs 127 |
| steady: predict | 0.74ms | 2.06ms | **2.8x slower** |
| steady: round trip | 1.73ms | 2.43ms | **1.4x slower** |
| walk complete | 62.9ms | 75.7ms | **1.2x slower** |

The two regressions are both understood and both bought something. Predict is
the bigger, denser lambdarank model - item 1 above. The walk is gitignore
support: the same binary with `--no-ignore` walks in 62.8ms, so the whole
difference is reading and applying ignore rules, and it lands after the results
are already on screen.

- **P1** two-phase walk. **P2** previews on their own thread, in process, with
  `bat`'s syntax set and no `eza`. **P3** one `stat` per historical path, and
  history loaded beside the model. **P4** one TIMING line per query, written
  after the results are sent. **P5** per-query work resolved once, not per file.
  **P6** a page of results is a slice, not a search. **P8** database diet:
  62MB -> 17MB, partial index, one transaction for impressions.
- **B1** the worker no longer drops requests it did not expect. **B2** the
  current directory keeps its row after navigating. **B3** text measured in
  columns, so a non-ASCII path cannot panic the UI. **B4** the main loop
  notices when the worker dies. **B6** void: the docs said LambdaRank, the code
  had drifted to `binary`, and the code came back (`764ab2c`).
- **S1** render reads `&App` and writes nothing back. **S2** one response path
  in the worker. **S3** four parallel structs collapsed. **S4** one database
  open per thread. **S5** one function hands the terminal back. **S6** walker
  cleanup, with P1. **S9** docs drift, fixed 2026-09-10.
- **Model reload** merged into one `Reload` request, and taken on directory
  change as well as on file open.

**Training and ranking** (`dd5af53` to `12eb383`):

- Time-based split, refit on everything, recency weights at a 180-day half-life,
  `log_file_size`, blend weight gated on training positives rather than
  last-30-day activity.
- **`lambdarank`, grouped by episode** (`764ab2c`) - the largest single ranking
  change measured: top-1 0.7010 -> 0.7717, MRR 0.7993 -> 0.8422, all three folds
  up. It had been `binary` since `ac1b74d`, "try using regression instead of
  lambdarank", with no measurement recorded.
- Features shipped: **directory visits**, **seconds since last click** (the
  largest feature by gain, 28.5%), **extension click share**. Re-measured over
  five seeds, the three together are worth +0.0098 top-1, sd of the mean 0.0040.
- **Collection:** `events.rank` records where each impression sat in the list.
- **A real bug found by benchmarking:** a model from a build with a different
  feature set loaded happily and then failed inside every predict, dropping
  ranking to unscored filter order. `Ranker::new` now checks the feature count.

## tried, measured, rejected

Kept because the measurements cost real time and the reasoning generalises.

- **`query_length`.** Under the old pooled objective it cost 3.9 points of top-1
  with rounds fixed, in all three folds. The interactions were learned - 75
  splits, none at the root, 143 splits beneath them - they just did not
  transfer: trained on the first 70%, top-1 rose on seen data and fell 0.041 on
  the future. Retried under lambdarank: the harm is gone (+0.0005) and no gain
  is left either.
- **Three directory generalisations**, all negative: visits inherited by the
  files inside a directory, clicks under a directory (the model would not use it
  at all - two folds came out bit-identical), and per-query clicks by directory.
  Taking a signal that works per file and spreading it over a directory does not
  transfer here, whatever the signal. The directory features that *do* work
  describe something the user did to that directory itself.
- **Depth below cwd.** -0.010 top-1. `is_under_cwd` already carries the useful
  half, and `fuzzy_score` already leans against long paths.

Three of these five were rejected on deltas inside the seed noise (-0.010 to
-0.016). They were not shown to hurt, only shown not to help enough to see, and
they stay out on parsimony - each costs compute on every keystroke and none read
more than 1.4% of gain. Anyone revisiting them should start from
`compare --seeds`, not from those numbers.

## not worth it

- **`ps` output as a signal.** Turning a process list into a per-file signal
  needs something like `lsof` per process; the plausible signal, "is my editor
  open on this project", is weak. Collection was removed in `03ca743`.
- **Raw shell history.** A plain-text copy of what the user has been typing,
  living in `~/.local`. The valuable part - where they `cd` - is already
  captured as `startup_visit` and is now the `visits_last_*` features.
- **Time of day / day of week.** Plausible, but it is an episode-level
  attribute, which is the shape that overfits here: see `query_length`.

## not now, maybe never
- use tracing subscriber crate so we can have nice spans of time and we can maybe visualize and optimize idk
