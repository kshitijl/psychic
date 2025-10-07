Read how-it-works.md to understand the basics. It should reflect the current state of the world, and also be a dev log. It should have caveats, gotchas, decisions, and bugs fixed.

After adding a fair amount of new code to implement a feature or fix a bug:
* run `cargo build --release`
* run `cargo rest`
* also `cargo clippy`.
* add new tests for the feature just added, if possible
* update how-it-works.md so that it reflects both current state and has a log of things done. You don't need to do this for every little change, but when you consider a piece of work "done".

Try to write features as functions in such a way that as much as possible of the functionality can be tested using `cargo test`. So, try to keep functions and functionality free of IO.

