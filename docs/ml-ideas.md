# Small ML ideas beyond the ranking model

Written 2026-11-06, from a brainstorm: can classifiers or other tiny ML methods
do something useful in psychic besides the main ranking model? Nothing here is
built or measured yet. Ideas the user liked most are marked **liked**; the rest
are recorded so the reasoning isn't lost.

Every idea uses data the events table already collects. The rules in `llm.md`
apply as usual: anything that changes what the model sees gets
`./bench/model.py compare --seeds 5`, and anything on the startup path or the
keystroke path gets `./bench/run.py`.

## Ideas

### Best-first walk (liked)

Past `SHALLOW_MODE_THRESHOLD` the walker abandons the descent and keeps only
the root's children, so which files are indexed in a large tree like `~` is up
to traversal order.

Instead, walk best-first: a priority queue of directories, scored by something
like Naive Bayes over directory-name tokens (`src`, `docs` high; `dist`,
`vendor`, `__pycache__`, `Library` low), learned from which subtrees across the
whole history ever had something clicked. Launching in `~` would then index the
8,000 entries most likely to matter rather than the first 8,000 found.

- This changes which files can be found at all (recall), not their order, so
  it does not conflict with "the list must never reorder without user input".
- The scores can be computed at retrain time in the background and loaded as a
  small table, so startup pays a file read, not a computation.
- The table is as old as the last retrain while it is being used. That's fine
  here, because it isn't a ranking feature, so there is no train/serve skew.

### Learned aliases (liked)

The ranker only ranks what `SkimMatcherV2` lets through. `episode_queries`
records which queries came before each click. Episodes where a query did not
fuzzy-match the file eventually clicked are typos ("srach") or private
nicknames ("notes" for `todo-current.md`). Count those (query, file) pairs, and
once a pair repeats, let that file through the filter for that query.

**First step:** one read-only SQL pass over the clicks, on a copy of
`events.db`, to count how often a click's episode queries fail to match it. If
it is rare, drop the idea. Cost per keystroke would be one hash lookup per
query.

### Jump confidence

`pd xy` always takes the top row, and a wrong jump is worse than opening the
TUI. A calibrator (logistic or isotonic regression) would predict the
probability that the top row is right, from a few inputs: the score gap between
rows 1 and 2, the number of matches, query length, and whether the top row
already has `clicks_for_this_query`. Below a threshold, `pd xy` opens the TUI
with the query typed in, which it already does when nothing matches.

- It needs a model of its own because lambdarank scores only order the rows.
  The logistic in `rank_files` doesn't make them a probability.
- Labels: replay TUI episodes, scoring each query with rolling-origin models,
  and record whether the clicked row was ranked first. `jump` events add real
  labels: a jump followed within seconds by another `pd` or a `cd` somewhere
  else was probably wrong.
- It runs in nanoseconds and is a pure function, so it can be tested
  expect-style.

### Ghost-text query completion

A prefix trie of past final queries, weighted by recency and by whether they
led to a click. Typing `to` shows `do-current` dimmed, and some key accepts it
(Tab and → are taken today). It never reorders the list.

### Next-directory prediction

Consecutive `startup_visit` rows are a sequence of directories, so a table of
how often the user goes from directory A to directory B (a Markov chain) could
answer a bare `pd` or seed the empty-query view. Used as a ranking feature
instead, it's a ranking change and needs the full `compare --seeds` treatment.

### Position bias

The todo's click-through-rate item names position debiasing as the thing that
would make CTR worth retrying, and `events.rank` has been collected since
2026-09-10.

A click model says P(click) = P(the user looked at row r) × P(the file is what
they wanted). If the first factor were known per row, an unclicked row 18 could
count as weak evidence and an unclicked row 1 as strong. The textbook way to
estimate it is to find the same file shown at different rows in different
episodes, assume its relevance was the same each time, and read the drop in
click rate as the drop in looking.

**That assumption is the problem, and the user called it "a huge huge deal".**
Psychic's queries are one or two letters on the way to something, so the same
file under `n` and under `s` is two different intents. Under `s` the user was
probably after `search_worker.rs`, and `notes.md` was never relevant at any
row. Pooling those would mistake the ranker's quality for how people look at
rows. Ways around it, cheapest first:

1. **Count only the rows above the click (liked).** Someone scanning top-down
   who clicks row 4 looked at rows 1-3 and passed them over. Rows below the
   click are unknown. Keep the rows above as negatives and drop or down-weight
   the rows below. This assumes nothing about relevance being the same across
   episodes, only that people read top-down, which in a list you type into is
   close to true. It is a training-data change: measure it with
   `compare --seeds 5`. It may cost something, because rows below the click
   still carry real "this file lost to that one" pairs.
2. **Same query only.** Count a file as "seen at different rows" only when the
   query was identical. Retraining on every launch gives natural variation in
   rows for the same query. Much closer to the same intent, but thin.
   **Before building:** count how many (query, file) pairs appear at two or more
   different rows. If it's a handful, stop.
3. **Within one episode.** Intent is fixed while typing toward one file, and
   each keystroke reshuffles the rows. But not clicking the target at row 3
   under `to` may mean the user saw it and kept typing, because typing is
   cheaper than arrowing down. That's a third reason for no click, which the
   two-factor model has no place for.
4. **Randomize slightly.** Now and then swap two adjacent rows on a query. Same
   query, same moment, same intent, so any difference in click rate is only
   looking. This doesn't break the no-unprompted-rerank rule (the order is set
   when the user types), but it sometimes shows a slightly worse order. That
   is a product decision.

Consequence for CTR: debiasing it with a curve built from mixed intents just
swaps one bias for another. A per-(query, file) CTR avoids mixing intents but
is thin. CTR stays rejected until there is a lot more data.

If CTR ever comes back, computing it at retrain time in the background and
loading a small file avoids the 14ms startup scan that rejected it, at the cost
of being one retrain old.

## Something to build before most of these: an online metric

Impressions record their row, so the row of every real click can be read
straight from the database. That gives MRR and top-1 from actual use, alongside
the offline folds. Putting it in the debug pane or `summarize-events` would show
whether any change above helps in practice. It would also warn if the trained
model ever does worse than the simple fallback.

## Considered and not worth it

- **Embeddings (file2vec) from files clicked together.** About 1,200 clicks is
  too little to train them.
- **A learned blend gate in place of the tanh ramp.** An installation crosses
  it once.
- **Predicting the next selected row so its preview is ready.** "The next row
  down" is already the answer, and a preview takes half a millisecond.
- **Time-of-day intent.** The episode-level shape that overfit as `query_length`;
  already under "not worth it" in `todo.md`.
