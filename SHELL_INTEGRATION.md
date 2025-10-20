# Shell Integration

Psychic supports shell integration, allowing you to quickly navigate directories using the `p` command.

## Installation

### Zsh

Add this line to your `~/.zshrc`:

```zsh
eval "$(psychic zsh)"
```

Then reload your shell or run `source ~/.zshrc`.

## Usage

### `p` - Browse all files and directories

Simply type `p` in your terminal to launch psychic. When you press **Ctrl-J**, psychic will exit and cd your shell into the current directory.

```bash
$ p
# (psychic opens, navigate to a directory)
# Press Ctrl-J
$ pwd
/path/to/selected/directory
```

### `pd` - Browse directories only

Type `pd` to launch psychic with the directories-only filter pre-enabled. Great for quickly navigating to a directory.

```bash
$ pd
# (psychic opens showing only directories)
# Press Ctrl-J
$ pwd
/path/to/selected/directory
```

## How it works

The `p` function:
1. Runs `psychic --shell-integration` in a command substitution `$(...)`
2. Psychic writes its TUI to `/dev/tty` (the terminal device directly)
3. When you press Ctrl-J, psychic prints the directory path to stdout and exits
4. The shell function captures stdout and runs `cd` to that directory

This approach (using `/dev/tty` for the TUI and stdout for the result) allows the TUI to display properly while still capturing the output via command substitution. This is exactly how zoxide's interactive mode works.

## Comparison to normal mode

- **Without shell integration** (`psychic`): Ctrl-J spawns a new shell in the selected directory
- **With shell integration** (`p` or `psychic --shell-integration`): Ctrl-J prints the path and exits, allowing your shell to cd

## Other shells

Currently only zsh is supported. Bash and fish support coming soon. The same approach should work - just need to create the equivalent shell functions.
