# Ctrl-R in psychic? Notes, not a plan

From a discussion on 2026-11-15. Nothing is decided. The user's next step:
"I'll try atuin and mcfly first before doing this."

The question: fzf's Ctrl-R (`fzf-history-widget`) lists `fc -rl 1` newest
first and fuzzy-filters it. Could psychic do better, and should it?

## The case for it

- **It is psychic's core idea.** psychic learns which item you want for a
  query in a context. Commands are another kind of item, and context predicts
  them better than it predicts files: the current directory, the git repo and
  the previous command all say a lot about the next one. fzf's Ctrl-R uses
  none of that, only recency and the fuzzy match.
- **Most of the machinery exists.** The TUI, the sqlite event log, LightGBM
  training, per-shell hooks and insert-at-cursor widgets. The query-to-pick
  learning (`episode_queries`) carries over directly: "dep" learns which
  deploy command you mean in this repo.
- **Lots of labels.** People press Ctrl-R much more often than they pick
  files, so the model would get far more training data.

## Why to be careful

- **It has been done.** mcfly (Rust) ranks Ctrl-R with a small neural net on
  cwd, exit status, recency and past selections, which is this exact idea. It
  never became widely used. atuin did, and mostly for its sync, its filter
  modes (this directory, this session, global) and its UI, not for better
  ranking. Ranking alone may not be the reason people switch.
- **It needs new data.** zsh history (with `EXTENDED_HISTORY`) records the
  time, but not the cwd or the exit status. To get those, a preexec/precmd hook
  has to log every command, which is a bigger change than today's chpwd hook.
  Old history would have no cwd. It could be guessed by replaying `cd`s, or
  imported from atuin, which records cwd per command.
- **Commands contain secrets.** Tokens and passwords pasted into commands
  would be copied into a second database. That needs a deliberate answer:
  honor `HISTIGNORE`/`HIST_IGNORE_SPACE`, give the database the same file
  permissions as the history file, and offer a purge.
- **The ranking problem is different.** Half the time you want the command
  you ran 30 seconds ago, and recency already handles that. The model must win
  on the other half without moving the obvious answer. Commands also need
  dedup and handling of near-duplicates that differ only in arguments.
- **Focus.** `positioning-notes.md` frames psychic as "replace `z` and Ctrl-T
  with one thing that learns". Adding Ctrl-R means competing with atuin too.

## Experiment to run before building anything

Replay `~/.zsh_history` offline. For each command, pretend the user typed its
first 1-3 characters, and record where the command actually run ranks under:

1. fzf's ordering: newest first, filtered by the fuzzy match.
2. A simple model on recency, frequency, the previous command, and a cwd
   guessed from earlier `cd`s.

Compare mean reciprocal rank. If the model does not clearly beat fzf, drop
the feature. If it does, those numbers are the pitch. A few hours of work.

## Open questions for the user

1. **A separate mode, or one list?** A Ctrl-R mode with its own candidates and
   model, or commands mixed into the file and directory results. The
   suggestion was a separate mode.
2. **What do atuin and mcfly get right or wrong?** If the user tries them and
   stops using them, the reason matters more than anything above.
3. **Is this core to psychic, or a side experiment?** That decides how much of
   the hook and privacy work is justified.
