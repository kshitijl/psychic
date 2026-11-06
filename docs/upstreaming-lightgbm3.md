# Upstreaming the training pieces to lightgbm3

Status: **parked** (2026-11-06). Nothing has been forked, pushed or posted.
Pick this up when there's time.

## Why

psychic trains in-process (`src/training/`). The safe wrapper crate
`lightgbm3` can't do two things training needs, so `src/training/lgbm.rs`
calls `lightgbm3-sys` directly, and that file holds all of the `unsafe` code:

1. **Query groups**, which `lambdarank` needs. lightgbm3 has `set_weights` but no `set_group`.
2. **Step-by-step training with a validation set**: create a booster, add
   validation data, `update_one_iter`, `eval`. lightgbm3 1.0.11 has
   `train_with_valid`, but its early stopping doesn't behave like Python's
   `lgb.train`. It looks only at the first metric, breaks on `is_finished`,
   rolls back, and doesn't expose `best_iteration`. Parity with `train.py`
   depends on matching Python exactly; see `src/training/mod.rs`, `EarlyStopping`.

A fork isn't an option for psychic itself, because crates.io rejects git
dependencies. Either upstream accepts these, or `lgbm.rs` stays.

## Upstream (checked 2026-11-06)

- Repo: `Mottl/lightgbm3-rs`. It's active; 1.0.11 shipped on 2026-09-08 and bundles LightGBM 4.7.0.
- No policy against AI-assisted contributions, and no history of junk PRs.
  #21 and #22 have gone unanswered since June and October, so expect slow review.
- CI runs `cargo build --features=openmp --features=polars`,
  `cargo test --features=polars`, and `cargo fmt -- --check`.

**Process decision:** I (the human) write all of the PR text myself. Claude
drafts code and walks me through it. Claude doesn't fork, push or open anything
without my explicit go-ahead.

## PR 1: `Dataset::set_group` (drafted)

The draft is `docs/upstreaming-lightgbm3-set-group.patch`, made against upstream
commit `3567b86` ("Updates LightGBM to 4.7.0"). It touches only `src/dataset.rs`, +97/−1.
To apply it:

```bash
git clone https://github.com/Mottl/lightgbm3-rs && cd lightgbm3-rs
git checkout -b dataset-set-group 3567b86
git apply /path/to/psychic/docs/upstreaming-lightgbm3-set-group.patch
cargo fmt -- --check && cargo test --features=polars
```

Last run: fmt was clean; 27 unit tests and 14 doc tests passed; clippy found nothing on the new lines.

What it does, line by line:

- **Import:** adds `C_API_DTYPE_INT32`, the tag that tells LightGBM "this array is int32".
- **Doc comment and runnable example:** a group is a run of consecutive rows;
  `[3, 2]` means rows 0–2 form one group and rows 3–4 another; the sizes must add up to the row count.
- **`set_group(&mut self, group: &[i32]) -> Result<()>`:** has the same shape as
  `set_weights`. It takes `i32` because LightGBM only accepts int32 groups,
  the same reason `set_weights` takes `f32`. It sums the sizes in `i64` so the
  sum can't overflow, returns an error if the total isn't the row count, and then
  calls `LGBM_DatasetSetField(handle, "group", ptr, len, INT32)`. LightGBM copies the array.
- **Tests:** all use the same 5-row fixture as the `set_weights` tests.
  - `set_group`: `[3, 2]` is accepted.
  - `set_group_wrong_total`: `[3, 1]` and `[3, 3]` are rejected.
  - `lambdarank_needs_groups`: training lambdarank without groups fails with
    LightGBM's exact message, "Ranking tasks require query information"
    (asserting the message means the test can't pass for the wrong reason),
    and training with `set_group(&[3, 2])` succeeds.

### Open decision: negative group sizes

`set_group(&[6, -1])` is **accepted** today, because the total is still 5, and
LightGBM doesn't reject it when the field is set. The group boundaries would be
0, 6, 5: they run backwards and point past the end of the data, so training on
that reads garbage or crashes.

- **Option A (recommended):** reject it, and add `assert!(dataset.set_group(&[6, -1]).is_err())` to `set_group_wrong_total`:
  ```rust
  if group.iter().any(|&size| size < 0) {
      return Err(Error::new("group sizes must not be negative".to_string()));
  }
  ```
- **Option B:** leave it, which matches `set_weights` (it doesn't check for negative
  weights either), and mention the gap in the PR text.

I'd go with A: a negative weight is just a strange number, but a negative group size
makes LightGBM read memory it shouldn't.

### Remaining steps

1. Decide on negative sizes, and update the patch to match.
2. Commit in the clone (decide how to attribute it).
3. Fork on GitHub, push the branch, and write the PR text myself.

## PR 2: step-by-step training API (not started)

Expose `Booster::new(train, params)`, `add_valid`, `update_one_iter`, `eval(i)` and
`save_model_to_string` as safe methods, so callers can write their own early
stopping. `src/training/lgbm.rs` is the reference implementation. Its
lifetime-bound `Booster<'data>` keeps the datasets alive for as long as LightGBM
holds pointers into them. This is a bigger API conversation than PR 1, so it's
better to wait and see how PR 1 is received.

If both land in a release, `lgbm.rs` can be deleted and `lightgbm3-sys` dropped
from `Cargo.toml`. Run `./bench/train_parity.py` afterwards; it should still print
"identical".
