# Making psychic easier to install: what's left

Written 2026-11-06, after v0.1.0 shipped with Homebrew (`kshitijl/tap`), a
`curl | sh` installer and Mac/Linux binaries, and the crate name `psychic-cli`
was claimed on crates.io. How releases work is in `docs/how-it-works.md`
("Building and releasing"); the procedure is in `llm.md` ("Releasing").
Most useful first.

1. **A real install on a machine that has never seen psychic.** Each piece was
   tested separately: the CI binaries in Docker containers, the curl installer
   into a throwaway HOME, and brew with `--dry-run` plus `fetch`. Nobody has
   done a full `brew install` and first run with no data. Check that the first
   run before any clicks is usable, and that shell integration works. A
   friend's laptop, a second macOS user, or a Linux box would do.

2. **bash and fish shell integration.** Only zsh is supported
   (`shell/psychic.zsh`). A large share of the people who'd want a `cd`
   replacement use bash or fish; zoxide supports both. This probably matters
   more to new users than anything below.

3. **A static musl Linux build** (`x86_64-unknown-linux-musl`,
   `aarch64-unknown-linux-musl`), for Alpine and for older distros: Ubuntu
   20.04 and Debian 11 are below the glibc 2.34 the current binaries need. Do
   it when someone asks. Expect friction building LightGBM's C++ against musl
   (static libstdc++). Test it the same way, with the PR artifacts in
   containers.

4. **homebrew-core**, so plain `brew install psychic` works. It needs notability
   (roughly 75+ stars, or 30+ forks or watchers), and the formula must build
   from source, which needs Rust and CMake 3.28+ as build dependencies.
   Until then, the tap is the normal route.

5. **Release notes.** Add a `CHANGELOG.md`. dist uses each version's section
   as the GitHub release body, so upgraders see what changed.

6. **Uninstall instructions and where data lives** in the README:
   `brew uninstall psychic`, or delete `~/.local/bin/psychic`, plus
   `~/.local/share/psychic` for the click database and model. A tool that
   records clicks should say how to wipe them. `psychic init --reverse` now
   prints exactly this, so the README can point at it; the README itself is
   the user's to edit.

7. **The Homebrew formula's style warnings.** dist's generated formula has no
   `test do` block and one line is too long; `brew style` prints both on
   every release. They don't break anything, but homebrew-core would require a
   test (e.g. `system bin/"psychic", "--version"`).

8. **A `psychic uninstall` command**, maybe, instead of (or on top of) the
   uninstall hints `psychic init --reverse` prints today. It would work out how
   psychic was installed (brew, the curl installer's `~/.local/bin`, cargo)
   and uninstall itself that way, take the shell integration out of
   `~/.zshrc`, and then deal with the history in the data directory: on a TTY,
   ask whether to delete it; without one, leave it alone and print that it was
   not deleted and where it is. Noted 2026-11-08, after `psychic init` landed.

Decided against, for now:

- **macOS signing and notarization.** It costs $99/yr. brew and `curl | sh`
  don't set the quarantine flag, so only browser downloads hit Gatekeeper.
- **A self-updater** (dist's `install-updater`). `brew upgrade` and re-running
  the curl installer already cover upgrades.
- **Windows.** psychic uses Unix-only APIs (`poll`, ttys, signals).
