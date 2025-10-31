## psychic

What if we put Spotlight in the terminal and made it fast? 

`psychic` is a terminal-based file browser with ML-powered ranking, built in Rust using ratatui. The results it shows you update as you type. It learns from your behavior (clicks, scrolls) to improve search relevance over time. Inspired by `fzf` and `zoxide`, `psychic` helps you navigate the filesystem and open the files you need, *fast*.

## DEMO!

[![asciicast](https://asciinema.org/a/752725.svg)](https://asciinema.org/a/752725)

## What kind of ML does it use?

`psychic` trains a tiny gradient-boosted tree model using the excellent LightGBM library. The model lives on your computer, using data collected from your actions within psychic. It makes no API calls, and your data never leaves your computer. No transformers, no deep learning, and definitely no LLMs. All that stuff would be, at least in 2025, too slow for this use-case.

## Okay seriously what is this, how does it work

`psychic` shows you all files and directories that are under the current working directory where you opened it, plus every file or directory you've ever clicked on (hit enter on or used your mouse to scroll) using psychic. Whenever you do some searches or click on anything, that gets recorded in a sqlite database on your filesystem. Every time you open `psychic`, it computes a bunch of features from your click data and trains a LightGBM model.

This LightGBM model tries to predict, for every file/directory, given the current query, how likely you are to click it. `psychic` sorts the files and directories by that score.

That's kind of it. The rest is details.

## Ok let's hear some details

Well. We need this to be fast, like under 100ms startup ideally, and we want the UI to update instantly as you type. All design decisions follow from these requirements. We are here because we want performance, otherwise we would click around in GUIs or use Spotlight or painfully type in `cd this` and `cd that` and tab complete. We need speed. We have a need for speed. That is why we are here so everything in psychic turns toward, sometimes bends, sometimes *breaks* in service of this goal. 

So the filesystem walking, the ML inference and the model training all happen on background threads. Threads pass data around in chunks, using channels. The UI thread doesn't get all search results, just the ones for the current page and, sometimes, the next and previous ones. It caches those pages and requests new pages when the user gets close to scrolling to the edge.

The model is written to disk so that next time you open the app, it uses the model trained last time.

The sqlite database has some indexes to support the most common queries. There is one particularly expensive but also important feature that we mostly pre-compute at the time of writing to the db: "which queries were tried before the user clicked on this file?" The actions a user does before clicking is called, in this code, an *episode*. The episode queries help make psychic feel magical: it can sometimes guess the file you want based on the very first letter you type, even if that letter is buried somewhere in the middle of the filename, because in previous episodes, when you typed that letter, you eventually clicked on that file. The episode queries are written in the same db event row as the click even though they can be derived from the other data we write, because doing this lets us avoid a linear scan through all that data at startup time.

We use `bat` to show you a preview of the currently selected file, but some files are huge so we don't want to generate a preview of the whole thing. So we generate a preview of only the first bit of the file and cache that in memory. If you scroll, we ask `bat` to generate the rest and cache *that*.

Syscalls are expensive but of course this whole program is syscalls. We try to minimize syscalls. No syscalls in the main render loop; we carefully make sure that we ask for file metadata at the time we walk the filesystem and then carry that around everywhere else.

## Supported systems

I run this on macOS and Ubuntu. It probably works on other Linux systems; the closer to Debian the more likely it is to work. I haven't tried running it on Windows.

The shell integration works with zsh. It might work with bash with a little bit of work.

## Installation

I'm working on making this part better, but for now there are no pre-built packages in any manager.

You'll need the rust toolchain, `uv` and a recent version of CMake installed. Then,

`cargo install --path .`

For preview, you'll also need `bat` and `eza` installed.

For shell integration, put this in your `.zshrc`:

`eval "$(psychic zsh)"`

## How to use it

If you're using the shell integration (which you should), type

* `p` to explore both files and directories
* `pd` to explore just directories. Hit enter to `cd` into the selected directory
* `pc` to explore just children of the cwd

Otherwise, type `psychic` to open the TUI. By default, it will explore both files and directories.

* Start typing to filter the list of files.
* Hit enter on a file to open it in my favorite editor, Helix (note to self: respect the user's `$EDITOR` environment variable).
* Hit enter on a directory to navigate into it.
* `Tab` and `Shift-Tab` to cycle through different filters: all files, just directories, just items under the current cwd, just direct descendants of the cwd.
* `Alt-Left/Right` to go back and forward in history.
* `Alt-Up` to navigate to the parent directory.
* `Ctrl-J` to get dropped in a shell in the current directory.
* `Ctrl-O` will open a debug pane showing you the values of each ML feature for the currently selected file or directory.
