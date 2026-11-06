## psychic

Read my post about it on tech website: [klauria.tech/psychic](https://klauria.tech/psychic/).

What if we put Spotlight in the terminal and made it fast? 

`psychic` is a terminal-based file browser with ML-powered ranking, built in Rust using ratatui. The results it shows you update as you type. It learns from your behavior (clicks, scrolls) to improve search relevance over time. Inspired by `fzf` and `zoxide`, `psychic` helps you navigate the filesystem and open the files you need, *fast*.

## Installation

### With homebrew on Mac or Linux

```sh
brew install kshitijl/tap/psychic
```

on Apple Silicon, Intel Macs, and for Linuxbrew users on x86_64 and Arm.

### curl installer on Mac or Linux

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/kshitijl/psychic/releases/latest/download/psychic-installer.sh | sh
```

This installs the binary to `~/.local/bin` and adds it to your PATH. Run it again to upgrade.

### Shell integration

For shell integration, put this in your `.zshrc`:

```sh
eval "$(psychic zsh)"
```

### Linux requirements

glibc 2.34 or newer, which means Ubuntu 22.04+, Debian 12+, Fedora 41+, RHEL 9+.

### Using cargo

You'll need the rust toolchain, CMake 3.28+ a C++ compiler and libclang. On macOS that's Xcode's command line tools (`xcode-select --install`); on Debian/Ubuntu, `build-essential clang libclang-dev`.

Then,

```sh
cargo install psychic-cli
```

### From source

Same requirements as above.

Clone this repo, then run

```sh
cargo install --path .
```

## DEMO!

[![asciicast](https://asciinema.org/a/752725.svg)](https://asciinema.org/a/752725)

## What kind of ML does it use?

`psychic` trains a tiny gradient-boosted tree model using the excellent LightGBM library. The model lives on your computer, using data collected from your actions within psychic. It makes no API calls, and your data never leaves your computer. No transformers, no deep learning, and definitely no LLMs. All that stuff would be, at least in 2025, too slow for this use-case.

## Okay seriously what is this, how does it work

`psychic` shows you all files and directories that are under the current working directory where you opened it, plus every file or directory you've interacted with while using psychic. If you set up the shell integration it also includes directories you `cd` into. Whenever you do some searches or click on anything, that gets recorded in a sqlite database on your filesystem. Every time you open `psychic`, it computes a bunch of features from your click data and trains a LightGBM model, in the background, inside the `psychic` binary itself: no Python, nothing else to install.

This LightGBM model tries to predict, for every file/directory, given the current query, how likely you are to click it. `psychic` sorts the files and directories by that score.

That's kind of it. The rest is details.

## Ok let's hear some details

Well. We need this to be fast, like under 100ms startup ideally, and we want the UI to update instantly as you type. All design decisions follow from these requirements. We are here because we want performance, otherwise we would click around in GUIs or use Spotlight or painfully type in `cd this` and `cd that` and tab complete. We need speed. We have a need for speed. That is why we are here so everything in psychic turns toward, sometimes bends, sometimes *breaks* in service of this goal. 

So the filesystem walking, the ML inference and the model training all happen on background threads. Threads pass data around in chunks, using channels. The UI thread doesn't get all search results, just the ones for the current page and, sometimes, the next and previous ones. It caches those pages and requests new pages when the user gets close to scrolling to the edge.

The model is written to disk so that next time you open the app, it uses the model trained last time.

The sqlite database has some indexes to support the most common queries. There is one particularly expensive but also important feature that we mostly pre-compute at the time of writing to the db: "which queries were tried before the user clicked on this file?" The actions a user does before clicking is called, in this code, an *episode*. The episode queries help make psychic feel magical: it can sometimes guess the file you want based on the very first letter you type, even if that letter is buried somewhere in the middle of the filename, because in previous episodes, when you typed that letter, you eventually clicked on that file. The episode queries are written in the same db event row as the click even though they can be derived from the other data we write, because doing this lets us avoid a linear scan through all that data at startup time.

The preview pane is generated in-process, on its own thread. Syntax highlighting is `syntect`, the same library `bat` is built on, and a directory listing is a `read_dir`. It used to shell out to `bat` and `eza` and parse the ANSI they printed back, which cost a process spawn per preview: moving the selection took 16ms, which needs to happen before redraw. Now it's half a millisecond, because the redraw never waits for the preview at all - it is generated on a thread and appears when it is ready. If you don't scroll a file, only what the pane shows is highlighted. On your first scroll it highlights the rest of the file in one pass.

Syscalls are expensive but of course this whole program is syscalls. We try to minimize syscalls. No syscalls in the main render loop; we carefully make sure that we ask for file metadata at the time we walk the filesystem and then carry that around everywhere else.

We keep only the last 1 year and up to 2000 paths of history. More doesn't seem to improve results, and unlimited growth of the history would slow down startup.

## Supported systems

Prebuilt binaries: macOS 11+ (Apple Silicon and Intel), and Linux on x86_64 or ARM with glibc 2.34+: Ubuntu 22.04+, Debian 12+, Fedora 41+. Ubuntu 20.04 and Alpine don't currently work. Windows is not supported.

The shell integration works with zsh. It might work with bash with a little bit of work.

## How to use it

If you're using the shell integration (which you should), type

* `p` to explore both files and directories
* `pd` to explore just directories. Hit enter to `cd` into the selected directory
* `pc` to explore just things under the cwd, at any depth

Otherwise, type `psychic` to open the TUI. By default, it will explore both files and directories.

* Start typing to filter the list of files.
* Hit enter on a file to open it in `$EDITOR`, falling back to `$VISUAL` and then to `vi`.
* Hit enter on a directory to navigate into it.
* `Tab` and `Shift-Tab` to cycle through different filters: all items, under the cwd (any depth), direct children of the cwd, directories only, files only.
* `Left/Right` to go back and forward in history.
* `Alt-Up` to navigate to the parent directory.
* `Ctrl-J` to get dropped in a shell in the current directory.
* `Ctrl-O` will open a debug pane showing you the values of each ML feature for the currently selected file or directory.
* `--no-ignore` shows files that `.gitignore` would hide. By default psychic respects `.gitignore`, `.ignore` and your global gitignore, which is mostly why searches stay fast: a build directory is usually the largest thing in a project and never the thing you are looking for. Dotfiles are still shown, because `.zshrc` is worth finding.
* `Ctrl-G` (or `F1`) shows a help screen with every key and command. The list above is the short version; the help screen is generated from the keybinding registry, so it is never out of date. There is a reminder of this in the bottom right of the search box.

## LLM use disclosure

This was my first project where I used LLMs to generate a substantial portion of the code. That being said, I stand behind the quality of this code. The design and architecture are my own creation. I exercised substantial creative control in how the tests and types are set up, which constrain the rest of the code. You can read `llm.md` and `docs/how-it-works.md` to see how the LLMs were instructed, but they are only a small part of the story. Every single commit is here because I decided that change was necessary, and many initial drafts were rejected before getting to the commits you see in this repository. 

At least as that term was defined in Karpathy's original tweet, this is not a vibe-coded project.

## License

MIT
