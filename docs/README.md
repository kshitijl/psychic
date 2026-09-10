# psychic docs

- **[how-it-works.md](how-it-works.md)** - what the code does and why. The
  module map, the thread architecture, the ranking features, the training
  pipeline, and the two benchmark harnesses. Kept current in the same commit as
  the change it describes; if it disagrees with the code, the doc is the bug.
- **[todo.md](todo.md)** - open work, and a record of what was tried, measured
  and rejected. The rejected list is the more useful half.
- **[archive/](archive/)** - dated material kept for provenance, not for
  reading as current state.

`../llm.md` is the working agreement - benchmarking rules, testing style, the
deep-modules philosophy - and lives at the root because that is where agents
look for it. `../README.md` is the user-facing introduction.
