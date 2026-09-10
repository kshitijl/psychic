Read how-it-works.md to understand the code and decisions.

Use asserts liberally throughout the code. For any function, consider documenting its preconditions in the form of asserts. Put these at the top of the function. They should be asserts, not debug_asserts, so they're run in prod and we can find bugs.

If a function returns a tuple, strongly consider defining and returning a struct instead. This way we give a name to each field.

## Module Design Philosophy

Follow John Ousterhout's "deep modules" philosophy:
* **Small interface, deep implementation** - Each module should have a minimal public API that hides substantial complexity inside
* **Separation of concerns** - Think hard about the API between different components. A minimal API should be exposed.
* **Information hiding** - The work of each component should be defined, and functions for that should live in the code for that component. They should not be public.
* **Example of a good deep module:** `preview.rs` (if extracted) would expose just `render()`, `scroll()`, `reset_scroll()` but hide all the bat/eza execution, caching, ANSI parsing, width adaptation logic inside.

See `refactor.md` for planned module extractions following this philosophy.

Read src/main.rs to understand the entrypoint. Read train.py to understand how the model is made.

## Development workflow

After adding implementing a feature or fixing a bug:
* run `just build`. You can see the justfile to understand what this does
* run `cargo test`
* also `cargo clippy`.
* add new tests for the feature just added, if possible
* **update how-it-works.md so that it reflects current state, in the same
  commit.** Not "later", not a follow-up: a commit that changes behaviour and
  leaves the document describing the old behaviour has made the document
  actively misleading, which is worse than a gap. If the change is genuinely
  invisible there - a test, a benchmark script, a comment - say so in the commit
  message rather than leaving the reader to wonder whether it was forgotten.
* **benchmark the change against the commit before it, and put the numbers in
  the commit message.**

## Benchmarking every change

Every commit gets benchmarked against its parent, and the numbers go in the
commit message. Not just the ones meant to be faster.

```bash
./bench/run.py setup HEAD      # build the parent commit as the baseline
# ... make the change, just build, cargo test, cargo clippy ...
./bench/run.py startup 6
./bench/run.py keystroke 30
```

Put the numbers in the commit message **as a table**, not as a run of
`a -> b; c -> d` prose - that is unreadable at the width a commit message wants:

```
                        before   after
first paint               1.50    1.33
steady: filter+rank       1.00    0.92
keystroke -> redraw       3.14    2.92
```

Why every commit and not just the performance ones:

* **A regression is only cheap to fix while you remember what you changed.**
  Found six commits later it is an afternoon of bisecting; found in the commit
  that caused it, it is usually obvious.
* **A performance claim with no measurement behind it is a guess.** Several
  changes in this repo that were "obviously" faster were not, and one that
  looked like a 1.8x regression turned out to be a benchmarking mistake. Write
  down what was measured, not what was expected.
* **The numbers accumulate into a history.** `git log` becomes the record of how
  the tool got faster and where it got slower, which is the only way to answer
  "when did startup double" without re-deriving it.

## Adding or changing a ranking feature

A feature changes what the model predicts, so the performance benchmark is not
enough on its own. Retrain and report what happened to ranking quality, as a
table, in the commit message:

```
                      before   after
  AUC                  0.938   0.941
  top-1                0.664   0.681
  MRR                  0.763   0.771
  RMSE                 0.131   0.129
```

Measured the way `bench/` measures speed: rolling-origin folds over the episode
timeline, so the model is always predicting the future from the past. One
feature per commit - two at once and neither number means anything, because a
gain and a loss cancel and both look like noise.

**Average over seeds, and know the floor.** Bagging and feature sampling are
random, and adding a column changes which subsets each tree sees - the same
perturbation a different seed causes. Measured on this data with
`./bench/model.py seeds 8`: one feature set, eight seeds, top-1 ranges 0.7827 to
0.8094. **A spread of 0.027, sd 0.010.** Most single features are worth less
than that, so a one-seed comparison cannot see them.

    ./bench/model.py compare --seeds 5     # means of 5 seeds, both sides

Five seeds put the standard error of the mean near 0.004, which is enough to
resolve a one-point change. **Where the decision is close, run eight.** A
feature measured at four seeds once read top-1 +0.0056, MRR +0.0031 and AUC
+0.0010 - positive on every metric, and wrong: at eight seeds it was -0.0008.
The failure mode is not a wild number, it is a plausible one.

Anything measured at one seed and smaller than about 0.03 top-1 is not evidence,
whichever way it points.

Gain is the steadier number at this data size: it aggregates thousands of
splits, where top-1 rests on ~110 episodes a fold. A feature that reads 20%+ of
total gain is being used heavily whatever the held-out delta says that day.

Report the feature's own importance too (gain, from `model_stats.json`), and say
where the data came from: a feature computed from rows the collection has never
written is a feature that will read zero until enough time passes.

The performance benchmark still applies. A feature is computed for every
candidate on every keystroke, so it is exactly the kind of change that can cost
a millisecond without anyone noticing.

A commit that touches only documentation has nothing to measure; say so rather
than running the harness for form's sake.

**Know the noise floor before reading anything into a number.** Benchmarked
against itself, the same binary lands within 1-3% on the big figures (walk
complete, worker state, first full render) and within about 15% on the
sub-millisecond ones (first paint, load clicks, per-feature times). A 10% move
on a 0.3ms number is noise; a 10% move on walk complete is real. When a result
matters, run it again.

Slower is often the right call - a bug fix, a feature, a refactor that makes the
code honest. Gitignore support costs 12ms of walk time and is worth keeping.
The rule is not "never regress", it is **never regress without knowing**: if a
change costs something, the commit message says how much and why it is worth it.
Read `bench/harness.py` before trusting a surprising result; five different
mistakes there each produced a confident wrong answer first.

Do not run `cargo build`. I have a symlink to the RELEASE binary under target. That's what I use every day, and that's what I test as a user. It must reflect the current latest code. You must run `just build`, which will build a release binary and also other tasks that I need done in order to read and understand the code.

You MUST ensure that `just build`, `cargo test` and `cargo clippy` produce clean output! No warnings or broken tests! No documentation link warnings!

## Version control

This is a single developer repository. Work on `main` and commit there directly.
Do not create a branch or open a pull request unless I ask for one.

## Running the binary

Write tests instead of trying to run the binary. It is a TUI application. It doesn't make sense for you, as an LLM, to test the binary by running it. It only makes sense when testing things like feature generation.

When running the binary, always use `cargo run --release -- <args>` instead of running the binary directly. This prevents running an outdated binary.

## Debug, error and log output

Do not eprintln! This is a terminal UI app. Printing to stderr messes up the UI. Instead, use the standard logging functions. This will go in the log file.

## Testing Guidelines

**Prefer expect-style tests:** Tests should be written in an "expect test" style where the expected output is explicitly written in the test code itself, not computed or hidden. This makes it easy to inspect the expected behavior at a glance.

Good example:
```rust
#[test]
fn test_truncate_path_simple() {
    let result = truncate_path("a/b/c/d/e.txt", 15);
    assert_eq!(result, "a/.../d/e.txt", "Should truncate middle components");
}
```

Bad example:
```rust
#[test]
fn test_truncate_path_simple() {
    let result = truncate_path("a/b/c/d/e.txt", 15);
    let expected = compute_expected_truncation(...); // Expected value is hidden
    assert_eq!(result, expected);
}
```

**Design for testability:** When writing functionality, prefer designs that enable expect tests:
* Extract pure functions that take inputs and return outputs (no IO, no global state)
* If a feature involves IO or complex state, refactor to separate the pure logic from the IO
* Consider whether the core logic can be tested in isolation with simple inputs and explicit expected outputs

**IO-free functions:** Try to write features as functions in such a way that as much as possible of the functionality can be tested using `cargo test`. So, try to keep functions and functionality free of IO.

