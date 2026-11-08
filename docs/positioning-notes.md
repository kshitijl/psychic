# Positioning notes: who switches, and why

Notes from a brainstorm on 2026-11-08, written down for the user to consider
later. Nothing here is decided. The README is the user's to write; this is raw
material for it.

## Why switch from zoxide + fzf?

The core user already has both: `zoxide` (`z foo`) to jump to directories by
frecency, and `fzf` (Ctrl-T, `vim $(fzf)`, `**<Tab>`) to pick files. They will
judge psychic against that pair in the first five minutes. The honest case:

1. **One tool for files and directories.** zoxide only knows directories;
   fzf only knows what is under the current one, and nothing about what you
   used. psychic ranks both, from anywhere: files you opened last week in
   another repo are one query away without `cd`ing there first.
2. **It learns from what you typed, not just where you went.** zoxide's score
   is visit count weighted by recency, the same for every query. psychic
   records the queries typed on the way to each pick (`episode_queries`), so
   a letter that once led to a file pulls that file up next time - even a
   letter buried in the middle of its name. That is the "it guessed from one
   letter" moment, and neither tool can do it.
3. **Ranking is a model of you, trained on your machine.** Recency, frequency,
   file type habits, the directory you are in, how a file matches the query:
   a LightGBM model weighs them from your own clicks. No network, no account.
4. **A preview while you choose.** Syntax-highlighted file contents and
   directory listings as you move, generated in-process, so it costs no
   latency.
5. **It starts from what you already have.** The first launch imports zoxide's
   database, shell history, editor recents and more (`psychic import`), so
   switching does not mean starting from zero - your zoxide history comes along.
6. **Speed is the design constraint.** Under ~10ms to first results; the walk,
   ranking and training all run off the UI thread.

What they give up, today:

- **bash and fish.** zoxide and fzf work everywhere. (Being fixed.)
- **Composability.** fzf is a filter: anything in, a line out, used inside
  other commands. psychic is mostly a destination - you open or `cd`. See the
  next section.
- **Maturity and ubiquity.** fzf is on every machine and in every dotfiles
  repo, and plugins assume it. psychic replaces part of that, not all of it.
- **Cold start.** Until about 30 clicks the model barely counts and the simple
  model ranks. The import is what has to carry the first week.

Probably the right framing: not "replace fzf", but "replace `z` and Ctrl-T
with one thing that learns", and keep fzf for piping arbitrary lists.

## Dropping a path into a command

fzf users compose: `vim $(fzf)`, `git add $(fzf)`, Ctrl-T to insert a path at
the cursor. psychic's answer so far is `pd`, which is exactly that pattern for
one command (`cd`): the TUI prints the pick, the shell function uses it.

Built on 2026-11-08: `--on-file-click=print-to-stdout` (with the existing
`--on-dir-click=print-to-stdout`), so `vim "$(psychic --on-file-click=print-to-stdout)"`
works, and a Ctrl-T binding in zsh, bash and fish that inserts the pick at the
cursor.

Still open, for the user to think about:

- **A short name for "pick and print".** The flags are long for typing by
  hand. Something like `psychic pick` (both clicks print, exit on the first),
  so `git add $(psychic pick)` reads well. How much of psychic should be a
  filter versus a place?
- **Multi-select.** fzf's Ctrl-T takes several paths (Tab to mark). psychic
  has no marking.
- **Relative or absolute.** The Ctrl-T widgets make a pick under `$PWD`
  relative and leave the rest absolute. `$(...)` gets absolute paths, which are
  always correct but long on the command line.
- **Should picks made this way count as clicks?** They do now: a pick is a
  pick. But "the file I passed to `git add`" may be a different signal from
  "the file I opened".
- **Pre-filtering by the command.** `vim **<Tab>` could start the TUI with a
  filter suited to the command (files for `vim`, dirs for `cd`).
