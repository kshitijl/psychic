## psychic

`psychic` is a terminal-based file browser with ML-powered ranking, built in Rust using ratatui. It learns from your behavior (clicks, scrolls) to improve search relevance over time. Inspired by `fzf` and `zoxide`, `psychic` helps you navigate the filesystem and open the files you need, *fast*.

[![asciicast](https://asciinema.org/a/752725.svg)](https://asciinema.org/a/752725)

## What kind of ML does it use?

`psychic` trains a tiny gradient-boosted tree model using the excellent LightGBM library. The model lives on your computer, using data collected from your actions within psychic. It makes no API calls, and your data never leaves your computer. No transformers, deep learning, and definitely no LLMs. All that stuff would be, at least in 2025, too slow for this use-case.

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
