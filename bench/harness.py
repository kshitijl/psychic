#!/usr/bin/env python3
"""Shared plumbing for the psychic benchmarks.

Everything here exists because psychic is a TUI: it has to be driven on a real
terminal, and the numbers have to come back out of its own TIMING log lines
rather than from wall-clock guesses outside the process.

Three things this handles that a naive harness gets wrong:

* **The window size matters.** Geometry decides how much the first full render
  has to draw, which is exactly what the preview work changed, so the pty is
  opened here and sized with TIOCSWINSZ. `script(1)` gives no control over it
  and hands out 80x24, a size nobody runs.
* **stdin must stay open.** An immediate EOF on the pty reads as a keypress and
  psychic quits before it has finished starting up.
* **The model must be pinned.** Every launch retrains in the background, and a
  retrain that finishes replaces model.txt. Two versions whose train.py differ
  end up predicting with differently sized models, which reads as a change in
  predict time that has nothing to do with the code being measured.
"""

import fcntl
import json
import os
import pty
import re
import select
import signal
import statistics
import struct
import subprocess
import termios
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = Path("/tmp/psychic-bench")
HOME = Path.home()
DEFAULT_DATA_DIR = HOME / ".local/share/psychic"

ROWS, COLS = 40, 120


class Version:
    """One binary under test, with its own copy of the data directory."""

    def __init__(self, name, binary, data_dir):
        self.name = name
        self.binary = Path(binary)
        self.data_dir = Path(data_dir)

    def logs(self):
        """Where this binary writes app.log.

        Commits before `9667bd3` wrote it to the default data directory whatever
        --data-dir said, so a baseline from before that logs somewhere else.
        Both candidates are watched and whichever grew is the one that was used.
        """
        return [self.data_dir / "app.log", DEFAULT_DATA_DIR / "app.log"]

    def pin_model(self):
        for name in ("model.txt", "model_stats.json"):
            src = WORK / f"pinned-{name}"
            if src.exists():
                subprocess.run(["cp", str(src), str(self.data_dir / name)], check=True)


def set_size(fd, rows=ROWS, cols=COLS):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def spawn(version, extra_args=(), rows=ROWS, cols=COLS, cwd=HOME):
    """Start psychic from `cwd` on a fresh pty. Returns (pid, fd).

    The directory matters: from $HOME the walk is large and hits the size
    threshold, while inside a project it finishes and the ignore rules decide
    what is in the list.
    """
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(str(cwd))
        os.environ["TERM"] = "xterm-256color"
        args = [str(version.binary), "--data-dir", str(version.data_dir), *extra_args]
        os.execv(str(version.binary), args)
    set_size(fd, rows, cols)
    return pid, fd


def stop(pid, fd, data_dir):
    """Hang the child up, kill it, and reap it without blocking forever."""
    try:
        os.close(fd)
    except OSError:
        pass
    os.kill(pid, signal.SIGKILL)
    for _ in range(50):
        try:
            if os.waitpid(pid, os.WNOHANG)[0]:
                break
        except ChildProcessError:
            break
        time.sleep(0.05)
    # A retrain may have got as far as spawning uv; it outlives its parent.
    subprocess.run(["pkill", "-f", f"{data_dir}/train.py"], capture_output=True)


def read_until_quiet(fd, idle=0.12, limit=1.5):
    """Read until the child has written nothing for `idle` seconds."""
    last = time.perf_counter()
    end = last + limit
    while time.perf_counter() < end:
        readable, _, _ = select.select([fd], [], [], 0.02)
        if readable:
            try:
                if not os.read(fd, 1 << 16):
                    return
            except OSError:
                return
            last = time.perf_counter()
        elif time.perf_counter() - last > idle:
            return


def drain_for(fd, seconds):
    """Keep reading for `seconds`, so the child never blocks on a full pty."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        readable, _, _ = select.select([fd], [], [], 0.05)
        if readable:
            try:
                os.read(fd, 1 << 16)
            except OSError:
                return


def parse_timings(text):
    """The TIMING lines of one run, as a flat dict of op -> milliseconds.

    Startup ops keep their first value. Queries are kept twice: the first is
    part of startup, and the last is the steady state, after the walk has
    finished and every version holds the same files. Comparing first queries
    compares different numbers of files and reads as a regression.
    """
    events = {}
    for line in text.splitlines():
        match = re.search(r"TIMING (\{.*\})", line)
        if not match:
            continue
        try:
            event = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        op = event.get("op")
        if op == "query":                       # current: one line per query
            events.setdefault("query_total", event["total_ms"])
            events.setdefault("query_features", event["features_ms"])
            events.setdefault("query_predict", event["predict_ms"])
            events.setdefault("query_count", event["count"])
            events["last_query_total"] = event["total_ms"]
            events["last_query_features"] = event["features_ms"]
            events["last_query_predict"] = event["predict_ms"]
            events["last_query_count"] = event["count"]
        elif op == "filter_and_rank_total":     # baseline: one op line each
            events.setdefault("query_total", event["ms"])
            events["last_query_total"] = event["ms"]
        elif op == "ml_compute_features":
            events.setdefault("query_features", event["ms"])
            events["last_query_features"] = event["ms"]
        elif op == "ml_predict":
            events.setdefault("query_predict", event["ms"])
            events["last_query_predict"] = event["ms"]
        elif op == "filter_files":
            events.setdefault("query_count", event["count"])
            events["last_query_count"] = event["count"]
        elif op == "query_round_trip":
            events.setdefault("query_round_trip", event["ms"])
            events["last_round_trip"] = event["ms"]
        elif op and "ms" in event:
            events.setdefault(op, event["ms"])
    return events


def run_once(version, seconds=4.0, extra_args=(), cwd=HOME):
    """Launch, let it start up and walk, kill it, and return its timings."""
    version.pin_model()
    offsets = {}
    for log in version.logs():
        log.parent.mkdir(parents=True, exist_ok=True)
        log.touch()
        offsets[log] = log.stat().st_size

    pid, fd = spawn(version, extra_args, cwd=cwd)
    drain_for(fd, seconds)
    stop(pid, fd, version.data_dir)

    # Whichever log grew is the one this binary writes to.
    appended = ""
    for log, offset in offsets.items():
        with open(log, "rb") as f:
            f.seek(offset)
            chunk = f.read().decode("utf-8", "replace")
        if len(chunk) > len(appended):
            appended = chunk
    return parse_timings(appended)


def keystroke_latencies(version, trials, settle=3.0, cwd=HOME):
    """Time from writing one byte to the first byte of the redraw, in ms.

    Both versions redraw on a tick - about ten writes a second when idle - so
    each measurement waits for a quiet moment first. That starts the clock just
    after a tick redraw, which makes the next write the one the keystroke caused.
    """
    version.pin_model()
    pid, fd = spawn(version, cwd=cwd)
    time.sleep(settle)                       # startup, the walk, the retrain
    read_until_quiet(fd, idle=0.5, limit=8.0)

    latencies = []
    # Type a word and rub it out, over and over: every keystroke changes the
    # query, so every one has to filter, rank and redraw.
    keys = list(b"psychic") + [0x7F] * 7
    for i in range(trials):
        read_until_quiet(fd)
        start = time.perf_counter()
        os.write(fd, bytes([keys[i % len(keys)]]))
        if not select.select([fd], [], [], 2.0)[0]:
            continue
        try:
            os.read(fd, 1 << 16)
        except OSError:
            break
        latencies.append((time.perf_counter() - start) * 1000.0)
        time.sleep(0.05)

    stop(pid, fd, version.data_dir)
    return latencies


def summarize(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": values[min(len(values) - 1, int(0.9 * len(values)))],
        "min": values[0],
        "max": values[-1],
    }
