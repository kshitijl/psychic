#!/usr/bin/env python3
"""Benchmark psychic against an older commit of itself.

    ./bench/run.py setup 1d4d767     # build that commit and stage two data dirs
    ./bench/run.py startup 10        # startup timings, alternating, medians
    ./bench/run.py keystroke 50      # keystroke -> redraw
    ./bench/run.py walk 5            # walk time, with and without gitignore

`setup` puts everything under /tmp/psychic-bench: a worktree at the baseline
commit, a release build of it, and a copy of the real events.db per version so
the two runs cannot interfere. The current binary is whatever `just build` last
produced. Nothing here touches the real data directory except to copy out of it.
"""

import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (  # noqa: E402
    COLS, DEFAULT_DATA_DIR, REPO, ROWS, WORK, Version,
    keystroke_latencies, run_once, summarize,
)


def versions():
    return {
        "baseline": Version("baseline", WORK / "old-src/target/release/psychic",
                            WORK / "data-old"),
        "current": Version("current", REPO / "target/release/psychic",
                           WORK / "data-new"),
    }


def setup(ref):
    """Build `ref` as the baseline and stage a data directory per version.

    Re-pointing an existing worktree rather than insisting on a fresh one is
    what makes "benchmark every commit against its parent" bearable: the
    rebuild is incremental, so moving the baseline forward one commit costs
    seconds rather than the forty a clean build takes.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    src = WORK / "old-src"
    # Resolve in the repo, not the worktree: "HEAD" there means whatever the
    # last benchmark left checked out, which is how a baseline silently stops
    # moving forward.
    resolved = subprocess.run(["git", "rev-parse", "--short", ref], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    if src.exists():
        # Discard whatever the last run left behind. Experiments edit this tree
        # directly - stripping features to measure them, say - and a dirty tree
        # makes every later checkout fail.
        subprocess.run(["git", "reset", "--hard", "--quiet"], cwd=src, check=True)
        subprocess.run(["git", "clean", "-fdq"], cwd=src, check=True)
        subprocess.run(["git", "checkout", "--detach", resolved], cwd=src, check=True)
    else:
        subprocess.run(["git", "worktree", "add", str(src), resolved], cwd=REPO, check=True)
    subprocess.run(["cargo", "build", "--release"], cwd=src, check=True)

    # Both versions start from a byte-identical database every time. Left to
    # accumulate, the two dirs drift - one run left a 20MB write-ahead log
    # against the other's 4.4MB, and the same binary measured against itself
    # came out 1.6x apart on first paint, entirely from the cost of opening it.
    for version in versions().values():
        if version.data_dir.exists():
            subprocess.run(["rm", "-rf", str(version.data_dir)], check=True)
        version.data_dir.mkdir(parents=True, exist_ok=True)
        # .backup rather than cp: it folds the source's write-ahead log into one
        # clean file, so neither copy starts with recovery work to do.
        subprocess.run(["sqlite3", str(DEFAULT_DATA_DIR / "events.db"),
                        f".backup '{version.data_dir}/events.db'"], check=True)
        subprocess.run(["cp", str(DEFAULT_DATA_DIR / "feature_schema.json"),
                        str(version.data_dir / "feature_schema.json")], check=True)

    # Commits before `03ca743` still write the two session columns that commit
    # dropped, and INSERT fails without them. Harmless to add back.
    subprocess.run(
        ["sqlite3", str(WORK / "data-old/events.db"),
         "ALTER TABLE sessions ADD COLUMN shell_history TEXT NOT NULL DEFAULT '';"
         "ALTER TABLE sessions ADD COLUMN running_processes TEXT NOT NULL DEFAULT '';"],
        capture_output=True)
    # Each version trains its own model, with its own feature set. Sharing one
    # would mean benchmarking a build against a model it cannot load.
    for version in versions().values():
        print(f"training {version.name}'s model...", flush=True)
        subprocess.run([str(version.binary), "retrain", "--data-dir", str(version.data_dir)],
                       check=True, capture_output=True)
        for name in ("model.txt", "model_stats.json"):
            subprocess.run(["cp", str(version.data_dir / name), str(version.pinned(name))],
                           check=True)

    print(f"staged {WORK}: baseline at {resolved} ({ref}), data dirs, a model each")


ROWS_TO_REPORT = [
    ("first_render", "first paint"),
    ("first_query_complete", "first results"),
    ("query_count", "  files ranked there"),
    ("first_full_render_complete", "first full render"),
    ("first_full_render_draw_time", "  of it, the draw"),
    ("worker_state_new_total", "worker state ready"),
    ("load_historical_files", "  load history"),
    ("load_clicks_total", "  load clicks"),
    ("ranker_init", "  ranker init"),
    ("walker_complete", "walk complete"),
    ("last_query_total", "steady: filter+rank"),
    ("last_query_features", "steady:   features"),
    ("last_query_predict", "steady:   predict"),
    ("last_round_trip", "steady: round trip"),
    ("last_query_count", "steady: files ranked"),
]


def startup(trials):
    vs = versions()
    results = {name: [] for name in vs}
    for version in vs.values():
        run_once(version)                     # warm up: schema, indexes, cache
    for i in range(trials):
        # Alternate who goes first, so a slow moment on the machine cannot land
        # on one version more often than the other.
        order = list(vs) if i % 2 == 0 else list(vs)[::-1]
        for name in order:
            results[name].append(run_once(vs[name]))
        print(f"trial {i + 1}/{trials}", flush=True)

    print(f"\n{ROWS}x{COLS} pty, from $HOME, {trials} trials each, 1 warmup discarded")
    print(f"{'':<24}{'baseline':>10}{'current':>10}{'':>14}")
    print("-" * 58)
    for key, label in ROWS_TO_REPORT:
        old = summarize([t[key] for t in results["baseline"] if key in t])
        new = summarize([t[key] for t in results["current"] if key in t])
        if not old or not new:
            print(f"{label:<24}{'(not logged by both)':>34}")
            continue
        o, n = old["median"], new["median"]
        if key.endswith("_count"):
            # A count, not a duration: more files ranked is better, so a ratio
            # here would read backwards.
            change = "files"
        elif n and o >= n:
            change = f"{o / n:.2f}x"
        else:
            change = f"{n / o:.2f}x slower" if o else ""
        print(f"{label:<24}{o:>10.2f}{n:>10.2f}{change:>14}")
    (WORK / "startup_results.json").write_text(json.dumps(results, indent=1))


def keystroke(trials):
    print(f"keystroke -> redraw, {ROWS}x{COLS} pty, {trials} keystrokes each")
    medians = {}
    for name, version in versions().items():
        stats = summarize(keystroke_latencies(version, trials))
        medians[name] = stats["median"]
        print(f"{name:<10}n={stats['n']:<4} median {stats['median']:6.2f}  "
              f"mean {stats['mean']:6.2f}  p90 {stats['p90']:6.2f}  "
              f"min {stats['min']:6.2f}  max {stats['max']:6.2f}")
    print(f"\n{medians['baseline']:.2f}ms -> {medians['current']:.2f}ms "
          f"({medians['baseline'] / medians['current']:.2f}x)")


def walk(trials):
    """Walk time, and how much of the difference is gitignore support.

    The three configurations are interleaved rather than run in blocks. The walk
    reads the filesystem, so it is the noisiest thing measured here, and running
    all of one config before the next lets a slow minute land entirely on one of
    them - which it did, and inverted the result.
    """
    vs = versions()
    runs = [
        ("baseline", vs["baseline"], ()),
        ("current", vs["current"], ()),
        ("current --no-ignore", vs["current"], ("--no-ignore",)),
    ]
    values = {label: [] for label, _, _ in runs}
    for i in range(trials):
        for label, version, args in runs:
            timings = run_once(version, extra_args=args)
            if "walker_complete" in timings:
                values[label].append(timings["walker_complete"])
        print(f"trial {i + 1}/{trials}", flush=True)

    print()
    for label, _, _ in runs:
        stats = summarize(values[label])
        print(f"{label:<22} median {stats['median']:6.1f}ms  "
              f"(n={stats['n']}, {stats['min']:.1f}-{stats['max']:.1f})")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    command = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if command == "setup":
        setup(arg or "HEAD~1")
    elif command == "startup":
        startup(int(arg or 10))
    elif command == "keystroke":
        keystroke(int(arg or 50))
    elif command == "walk":
        walk(int(arg or 5))
    else:
        print(__doc__)


main()
