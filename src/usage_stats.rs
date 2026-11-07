//! `psychic stats`: how much work psychic saves you, measured from the events
//! database.
//!
//! Two pure functions: `compute` turns the logged events into a
//! `UsageReport`, and `render` turns that into text. The command in `main.rs`
//! does the reading and the printing. See "psychic stats" in
//! docs/how-it-works.md for what each figure means and what it does not.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::Path;

use crate::db::{Launch, UsageAction, UsageEvent};

/// A gap longer than this before a pick is left out of the timing figures:
/// psychic was sitting open, not being searched.
const MAX_PICK_SECONDS: i64 = 10 * 60;

#[derive(Debug, PartialEq)]
pub struct UsageReport {
    /// First and last logged event. `None` when nothing has been logged.
    pub span: Option<Span>,
    /// TUI launches that showed anything.
    pub launches: usize,
    /// Rows picked from the list with Enter.
    pub picks: usize,
    /// `pd xy` jumps, which pick without showing a list.
    pub jumps: usize,
    /// Distinct paths picked.
    pub destinations: usize,
    pub typing: Typing,
    pub ranking: Ranking,
    pub timing: Timing,
    pub effort: Effort,
}

#[derive(Debug, PartialEq)]
pub struct Span {
    pub first: i64,
    pub last: i64,
}

#[derive(Debug, PartialEq)]
pub struct Typing {
    /// Characters in the query at each pick, summed over every pick.
    pub chars: usize,
    /// Picks made with an empty query. Arrow keys may have been pressed.
    pub typed_nothing: usize,
    /// Picks made with an empty query on the top row: Enter was the only key.
    pub enter_only: usize,
}

#[derive(Debug, PartialEq)]
pub struct Ranking {
    /// Picks whose position in the list is known.
    pub known: usize,
    pub top_1: usize,
    pub top_3: usize,
}

/// Time from the launch, or from the previous pick in the same launch, to a
/// pick. Timestamps are whole seconds, so a recorded gap of `n` means the real
/// one was under `n + 1`.
#[derive(Debug, PartialEq)]
pub struct Timing {
    /// Picks with a gap to measure, under `MAX_PICK_SECONDS`.
    pub timed: usize,
    /// Picks left out: a gap over `MAX_PICK_SECONDS`, or a clock that went
    /// backwards.
    pub left_out: usize,
    /// Of the recorded gaps, so whole seconds or halfway between two.
    pub median_seconds: Option<f64>,
    /// Recorded as at most 1s, so under 2s.
    pub under_2s: usize,
    /// Recorded as at most 3s, so under 4s.
    pub under_4s: usize,
}

/// Characters typed against the length of the path picked, written relative
/// to where psychic was launched: what `cd` would have needed.
#[derive(Debug, PartialEq)]
pub struct Effort {
    /// Picks whose launch directory is known.
    pub picks: usize,
    pub path_chars: usize,
    pub typed_chars: usize,
}

/// Measure psychic from its logged events.
///
/// A pick's position is found from the impressions logged for the same query
/// in the same launch just before it: the TUI logs what is on screen before
/// acting on Enter. Impressions logged before the `rank` column existed have
/// no position recorded, and there the order they were logged in stands in
/// for it. That is the order they were on screen; where both exist they
/// agree.
pub fn compute(events: &[UsageEvent], launches: &[Launch]) -> UsageReport {
    assert!(
        events.windows(2).all(|pair| pair[0].id < pair[1].id),
        "events must be in the order they were logged"
    );

    let launch_by_id: HashMap<&str, &Launch> = launches
        .iter()
        .map(|launch| (launch.session_id.as_str(), launch))
        .collect();

    // What was on screen, per (launch, subsession, query), in display order.
    let mut shown: HashMap<ListKey, Vec<ShownRow>> = HashMap::new();
    let mut launched: HashSet<&str> = HashSet::new();
    let mut previous_pick_at: HashMap<&str, i64> = HashMap::new();
    let mut destinations: HashSet<&str> = HashSet::new();
    let mut gaps: Vec<i64> = Vec::new();

    let mut report = UsageReport {
        span: None,
        launches: 0,
        picks: 0,
        jumps: 0,
        destinations: 0,
        typing: Typing {
            chars: 0,
            typed_nothing: 0,
            enter_only: 0,
        },
        ranking: Ranking {
            known: 0,
            top_1: 0,
            top_3: 0,
        },
        timing: Timing {
            timed: 0,
            left_out: 0,
            median_seconds: None,
            under_2s: 0,
            under_4s: 0,
        },
        effort: Effort {
            picks: 0,
            path_chars: 0,
            typed_chars: 0,
        },
    };

    for event in events {
        report.span = Some(match report.span {
            None => Span {
                first: event.timestamp,
                last: event.timestamp,
            },
            Some(Span { first, last }) => Span {
                first: first.min(event.timestamp),
                last: last.max(event.timestamp),
            },
        });

        let session = event.session_id.as_str();
        let key = ListKey {
            session,
            subsession: event.subsession_id,
            query: &event.query,
        };
        match event.action {
            UsageAction::Impression { rank } => {
                launched.insert(session);
                shown.entry(key).or_default().push(ShownRow {
                    path: &event.full_path,
                    rank,
                });
            }
            UsageAction::Jump => report.jumps += 1,
            UsageAction::Click => {
                launched.insert(session);
                report.picks += 1;
                destinations.insert(&event.full_path);

                let typed = event.query.chars().count();
                report.typing.chars += typed;

                let rank = shown
                    .get(&key)
                    .and_then(|on_screen| position_of(on_screen, &event.full_path));
                if typed == 0 {
                    report.typing.typed_nothing += 1;
                    if rank == Some(1) {
                        report.typing.enter_only += 1;
                    }
                }
                if let Some(rank) = rank {
                    report.ranking.known += 1;
                    report.ranking.top_1 += usize::from(rank == 1);
                    report.ranking.top_3 += usize::from(rank <= 3);
                }

                let launch = launch_by_id.get(session);
                let started_at = previous_pick_at
                    .get(session)
                    .copied()
                    .or(launch.map(|launch| launch.created_at));
                if let Some(started_at) = started_at {
                    let gap = event.timestamp - started_at;
                    if (0..MAX_PICK_SECONDS).contains(&gap) {
                        gaps.push(gap);
                    } else {
                        report.timing.left_out += 1;
                    }
                }
                previous_pick_at.insert(session, event.timestamp);

                if let Some(launch) = launch {
                    report.effort.picks += 1;
                    report.effort.typed_chars += typed;
                    report.effort.path_chars +=
                        relative_path_chars(Path::new(&launch.cwd), Path::new(&event.full_path));
                }
            }
        }
    }

    report.launches = launched.len();
    report.destinations = destinations.len();
    report.timing.timed = gaps.len();
    report.timing.under_2s = gaps.iter().filter(|&&gap| gap <= 1).count();
    report.timing.under_4s = gaps.iter().filter(|&&gap| gap <= 3).count();
    report.timing.median_seconds = median(&mut gaps);
    report
}

/// One list as the user saw it: a query, in one subsession of one launch.
#[derive(PartialEq, Eq, Hash)]
struct ListKey<'a> {
    session: &'a str,
    subsession: i64,
    query: &'a str,
}

/// One impression in that list.
struct ShownRow<'a> {
    path: &'a str,
    /// `None` on rows logged before the column existed.
    rank: Option<u32>,
}

/// Where `path` sat in a list of impressions, counting from 1: the rank the
/// latest impression of it recorded, or else the order it first appeared in.
fn position_of(on_screen: &[ShownRow], path: &str) -> Option<u32> {
    let mut distinct: Vec<&str> = Vec::new();
    let mut position = None;
    for &ShownRow { path: shown, rank } in on_screen {
        if !distinct.contains(&shown) {
            distinct.push(shown);
        }
        if shown == path {
            let in_order = distinct
                .iter()
                .position(|&p| p == path)
                .map(|i| i as u32 + 1);
            position = rank.or(in_order);
        }
    }
    position
}

/// Length of `to` written relative to `from`, as `cd` would take it.
fn relative_path_chars(from: &Path, to: &Path) -> usize {
    let from: Vec<_> = from.components().collect();
    let to: Vec<_> = to.components().collect();
    let common = from.iter().zip(&to).take_while(|(a, b)| a == b).count();

    let parts: Vec<String> = std::iter::repeat_n("..".to_string(), from.len() - common)
        .chain(
            to[common..]
                .iter()
                .map(|part| part.as_os_str().to_string_lossy().into_owned()),
        )
        .collect();
    if parts.is_empty() {
        return 1; // "."
    }
    parts.iter().map(|part| part.chars().count()).sum::<usize>() + parts.len() - 1
}

fn median(values: &mut [i64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[middle] as f64
    } else {
        (values[middle - 1] + values[middle]) as f64 / 2.0
    })
}

/// The report as `psychic stats` prints it. `date` formats a timestamp.
pub fn render(report: &UsageReport, date: impl Fn(i64) -> String) -> String {
    let mut out = String::new();
    let Some(span) = &report.span else {
        return "Nothing logged yet. Use psychic for a while and run this again.\n".to_string();
    };
    // Writing to a String cannot fail.
    let _ = writeln!(
        out,
        "psychic, {} to {}\n",
        date(span.first),
        date(span.last)
    );
    let _ = writeln!(
        out,
        "{} launches, {} picks, {} different places picked.",
        report.launches, report.picks, report.destinations
    );
    if report.jumps > 0 {
        let _ = writeln!(
            out,
            "Plus {} jumps with `pd`, which skip the list.",
            report.jumps
        );
    }
    if report.picks == 0 {
        return out;
    }

    let picks = report.picks;
    let _ = writeln!(out, "\nTyping");
    let _ = writeln!(
        out,
        "  {:.1} characters typed per pick, on average",
        report.typing.chars as f64 / picks as f64
    );
    let _ = writeln!(
        out,
        "  {} of picks: no characters typed",
        percent(report.typing.typed_nothing, picks)
    );
    let _ = writeln!(
        out,
        "  {} of picks: opened psychic and pressed Enter, nothing else",
        percent(report.typing.enter_only, picks)
    );

    let ranking = &report.ranking;
    if ranking.known > 0 {
        let _ = writeln!(
            out,
            "\nRanking ({} picks whose position is known)",
            ranking.known
        );
        let _ = writeln!(
            out,
            "  {} were the top result",
            percent(ranking.top_1, ranking.known)
        );
        let _ = writeln!(
            out,
            "  {} were in the top 3",
            percent(ranking.top_3, ranking.known)
        );
    }

    let timing = &report.timing;
    if let Some(median) = timing.median_seconds {
        let _ = writeln!(
            out,
            "\nSpeed, from opening psychic or the previous pick ({} picks)",
            timing.timed
        );
        let _ = writeln!(out, "  median {median}s");
        let _ = writeln!(out, "  {} under 2s", percent(timing.under_2s, timing.timed));
        let _ = writeln!(out, "  {} under 4s", percent(timing.under_4s, timing.timed));
        let _ = writeln!(out, "  timestamps are whole seconds");
        if timing.left_out > 0 {
            let _ = writeln!(
                out,
                "  left out: {} picks after psychic sat open over 10 minutes",
                timing.left_out
            );
        }
    }

    let effort = &report.effort;
    if effort.picks > 0 && effort.path_chars > 0 {
        let _ = writeln!(out, "\nEffort ({} picks)", effort.picks);
        let _ = writeln!(
            out,
            "  paths picked were {:.0} characters, written from where psychic was opened",
            effort.path_chars as f64 / effort.picks as f64
        );
        let _ = writeln!(
            out,
            "  you typed {} of that",
            percent(effort.typed_chars, effort.path_chars)
        );
    }
    out
}

fn percent(part: usize, whole: usize) -> String {
    assert!(whole > 0, "a percentage of nothing");
    format!("{:.0}%", 100.0 * part as f64 / whole as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch(session: &str, cwd: &str, created_at: i64) -> Launch {
        Launch {
            session_id: session.to_string(),
            cwd: cwd.to_string(),
            created_at,
        }
    }

    /// Builds events in log order, numbering them as the database would.
    struct Log(Vec<UsageEvent>);

    impl Log {
        fn push(
            &mut self,
            at: i64,
            session: &str,
            sub: i64,
            query: &str,
            path: &str,
            action: UsageAction,
        ) {
            let id = self.0.len() as i64 + 1;
            self.0.push(UsageEvent {
                id,
                timestamp: at,
                query: query.to_string(),
                full_path: path.to_string(),
                subsession_id: sub,
                session_id: session.to_string(),
                action,
            });
        }

        /// A screenful, in display order, with or without recorded ranks.
        fn shown(
            &mut self,
            at: i64,
            session: &str,
            sub: i64,
            query: &str,
            paths: &[&str],
            ranked: bool,
        ) {
            for (i, path) in paths.iter().enumerate() {
                let rank = ranked.then_some(i as u32 + 1);
                self.push(
                    at,
                    session,
                    sub,
                    query,
                    path,
                    UsageAction::Impression { rank },
                );
            }
        }

        fn pick(&mut self, at: i64, session: &str, sub: i64, query: &str, path: &str) {
            self.push(at, session, sub, query, path, UsageAction::Click);
        }
    }

    /// Four launches covering every case the report counts.
    fn history() -> (Vec<UsageEvent>, Vec<Launch>) {
        let mut log = Log(Vec::new());

        // Open, Enter: the top row with nothing typed, a second later.
        log.shown(101, "a", 0, "", &["/home/u/src", "/home/u/notes"], true);
        log.pick(101, "a", 0, "", "/home/u/src");

        // Typed "n" and went down one row, 3 seconds in...
        log.shown(
            201,
            "b",
            1,
            "n",
            &["/home/u/notes/todo.md", "/home/u/src/main.rs"],
            true,
        );
        log.pick(203, "b", 1, "n", "/home/u/src/main.rs");
        // ...then, from a row logged before ranks were recorded, the second
        // row with nothing typed, 7 seconds after the previous pick.
        log.shown(
            205,
            "b",
            2,
            "",
            &["/home/u/x", "/home/u/notes/todo.md"],
            false,
        );
        log.pick(210, "b", 2, "", "/home/u/notes/todo.md");

        // Opened and closed without a pick.
        log.shown(300, "c", 0, "", &["/home/u/src"], true);

        // Left open for over 10 minutes, then the top row.
        log.shown(400, "d", 0, "s", &["/home/u/src"], true);
        log.pick(4400, "d", 0, "s", "/home/u/src");

        log.push(5000, "e", 0, "x", "/home/u/x", UsageAction::Jump);

        let launches = vec![
            launch("a", "/home/u", 100),
            launch("b", "/home/u/src", 200),
            launch("c", "/home/u", 300),
            launch("d", "/home/u", 400),
        ];
        (log.0, launches)
    }

    #[test]
    fn test_compute_counts_every_kind_of_pick() {
        let (events, launches) = history();
        assert_eq!(
            compute(&events, &launches),
            UsageReport {
                span: Some(Span {
                    first: 101,
                    last: 5000
                }),
                launches: 4,
                picks: 4,
                jumps: 1,
                destinations: 3,
                typing: Typing {
                    chars: 2,
                    typed_nothing: 2,
                    enter_only: 1,
                },
                ranking: Ranking {
                    known: 4,
                    top_1: 2,
                    top_3: 4,
                },
                timing: Timing {
                    timed: 3,
                    left_out: 1,
                    median_seconds: Some(3.0),
                    under_2s: 1,
                    under_4s: 2,
                },
                effort: Effort {
                    picks: 4,
                    // "src", "main.rs", "../notes/todo.md", "src"
                    path_chars: 3 + 7 + 16 + 3,
                    typed_chars: 2,
                },
            }
        );
    }

    #[test]
    fn test_a_pick_takes_the_latest_rank_shown_for_its_query() {
        let rows = |rows: &[(&'static str, Option<u32>)]| -> Vec<ShownRow<'static>> {
            rows.iter()
                .map(|&(path, rank)| ShownRow { path, rank })
                .collect()
        };
        // The same query logged twice as the list moved under it.
        assert_eq!(
            position_of(
                &rows(&[("/a", Some(1)), ("/b", Some(2)), ("/b", Some(1))]),
                "/b"
            ),
            Some(1)
        );
        // Unranked rows count in the order they first appeared.
        assert_eq!(
            position_of(
                &rows(&[("/a", None), ("/b", None), ("/a", None), ("/c", None)]),
                "/c"
            ),
            Some(3)
        );
        assert_eq!(position_of(&rows(&[("/a", Some(1))]), "/z"), None);
    }

    #[test]
    fn test_relative_path_chars_is_what_cd_would_need() {
        let len = |from: &str, to: &str| relative_path_chars(Path::new(from), Path::new(to));
        assert_eq!(len("/home/u", "/home/u/src"), 3);
        assert_eq!(
            len("/home/u/src", "/home/u/notes/a.md"),
            "../notes/a.md".len()
        );
        assert_eq!(len("/home/u", "/home/u"), 1);
        assert_eq!(len("/home/u/a/b", "/home/u"), "../..".len());
    }

    #[test]
    fn test_median_of_an_even_count_is_halfway() {
        assert_eq!(median(&mut [4, 1, 3, 2]), Some(2.5));
        assert_eq!(median(&mut [5, 1, 3]), Some(3.0));
        assert_eq!(median(&mut []), None);
    }

    #[test]
    fn test_render() {
        let (events, launches) = history();
        let report = compute(&events, &launches);
        assert_eq!(
            render(&report, |t| format!("t{t}")),
            "\
psychic, t101 to t5000

4 launches, 4 picks, 3 different places picked.
Plus 1 jumps with `pd`, which skip the list.

Typing
  0.5 characters typed per pick, on average
  50% of picks: no characters typed
  25% of picks: opened psychic and pressed Enter, nothing else

Ranking (4 picks whose position is known)
  50% were the top result
  100% were in the top 3

Speed, from opening psychic or the previous pick (3 picks)
  median 3s
  33% under 2s
  67% under 4s
  timestamps are whole seconds
  left out: 1 picks after psychic sat open over 10 minutes

Effort (4 picks)
  paths picked were 7 characters, written from where psychic was opened
  you typed 7% of that
"
        );
    }

    #[test]
    fn test_render_with_nothing_logged() {
        assert_eq!(
            render(&compute(&[], &[]), |t| t.to_string()),
            "Nothing logged yet. Use psychic for a while and run this again.\n"
        );
    }
}
