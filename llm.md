Read how-it-works.md to understand the code and decisions.

Use asserts liberally throughout the code. For any function, consider documenting its preconditions in the form of asserts. Put these at the top of the function. They should be asserts, not debug_asserts, so they're run in prod and we can find bugs.

If a function returns a tuple, strongly consider defining and returning a struct instead. This way we give a name to each field.

Separation of concerns: think hard about the API between different components. A minimal API should be exposed. The work of each component should be defined, and functions for that should live in the code for that component. They should not be public.

Read src/main.rs to understand the entrypoint. Read train.py to understand how the model is made.

After adding a fair amount of new code to implement a feature or fix a bug:
* run `cargo build --release`
* run `cargo test`
* also `cargo clippy`.
* add new tests for the feature just added, if possible
* update how-it-works.md so that it reflects current state.

When running the binary, always use `cargo run --release -- <args>` instead of running the binary directly. This prevents running an outdated binary.

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

