Read how-it-works.md to understand the code and decisions.

Use asserts liberally throughout the code. For any function, consider documenting its preconditions in the form of asserts. Put these at the top of the function.

If a function returns a tuple, strongly consider defining and returning a struct instead. This way we give a name to each field.

Read src/main.rs to understand the entrypoint. Read train.py to understand how the model is made.

It should reflect the current state of the world, and also be a dev log. It should have caveats, gotchas, decisions, and bugs fixed.

After adding a fair amount of new code to implement a feature or fix a bug:
* run `cargo build --release`
* run `cargo test`
* also `cargo clippy`.
* add new tests for the feature just added, if possible
* update how-it-works.md so that it reflects both current state and has a log of things done. You don't need to do this for every little change, but when you consider a piece of work "done".

When running the binary, always use `cargo run --release -- <args>` instead of running the binary directly. This prevents running an outdated binary.

Try to write features as functions in such a way that as much as possible of the functionality can be tested using `cargo test`. So, try to keep functions and functionality free of IO.

