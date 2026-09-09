use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Timing event from log file
///
/// Most lines are one op and one `ms`. The `query` op is the exception: ranking
/// emits a single line per query carrying its whole breakdown, rather than the
/// ~23 separate lines it used to write between the keystroke and the results.
#[derive(Debug, Deserialize)]
struct TimingEvent {
    op: String,
    #[serde(default)]
    ms: Option<f64>,
    #[serde(default)]
    avg_ms: Option<f64>,
    #[serde(default)]
    total_ms: Option<f64>,
    #[serde(default)]
    count: Option<usize>,
    #[serde(default)]
    filter_ms: Option<f64>,
    #[serde(default)]
    simple_ms: Option<f64>,
    #[serde(default)]
    features_ms: Option<f64>,
    #[serde(default)]
    predict_ms: Option<f64>,
    #[serde(default)]
    blend_ms: Option<f64>,
    /// Time in each feature's `compute`, totalled over the files in the query.
    #[serde(default)]
    per_feature: Option<BTreeMap<String, f64>>,
}

impl TimingEvent {
    /// The number this event is summarised by, for the totals list.
    ///
    /// A `query` line's headline is its `total_ms`; every other line's is `ms`.
    fn headline_ms(&self) -> Option<f64> {
        self.ms.or(if self.op == "query" {
            self.total_ms
        } else {
            None
        })
    }
}

/// Print a `query` line as its own indented block.
fn print_query(event: &TimingEvent, width: usize) {
    println!(
        "{:<8.2}{:<width$}({} files)",
        event.total_ms.unwrap_or(0.0),
        "query",
        event.count.unwrap_or(0),
        width = width
    );

    for (label, ms) in [
        ("filter", event.filter_ms),
        ("simple", event.simple_ms),
        ("features", event.features_ms),
        ("predict", event.predict_ms),
        ("blend", event.blend_ms),
    ] {
        if let Some(ms) = ms {
            println!("{:<8.2}  {}", ms, label);
        }
    }

    // Slowest features first: the reason this breakdown exists is to find the
    // one feature that costs more than everything around it.
    if let Some(per_feature) = &event.per_feature {
        let mut features: Vec<(&String, &f64)> = per_feature.iter().collect();
        features.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (name, ms) in features {
            println!("{:<8.3}    {}", ms, name);
        }
    }
}

/// Pull the session id out of a log line.
///
/// Format: `[2025-10-21 05:03:10 INFO psychic 3517040894769903083] message`
fn session_id_of(line: &str) -> Option<&str> {
    let end = line.find(']')?;
    let before_bracket = &line[..end];
    let start = before_bracket.rfind(' ')?;
    let candidate = &before_bracket[start + 1..];

    if !candidate.is_empty() && candidate.chars().all(|c| c.is_ascii_digit()) {
        Some(candidate)
    } else {
        None
    }
}

/// Analyze performance timings from the log file
pub fn analyze_perf(log_path: &Path) -> Result<()> {
    // Read the log file
    let file =
        File::open(log_path).context(format!("Failed to open log file at {:?}", log_path))?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

    // Extract the latest session ID that actually started the TUI.
    //
    // Every psychic invocation logs under its own session id, including CLI
    // subcommands like `retrain` and `generate-features`, which emit no startup
    // timings at all. Taking the last session id in the file therefore produced an
    // empty report whenever the most recent run was a CLI one. Anchor on
    // `first_render` instead: it is logged exactly once, only by the TUI.
    let session_id = lines
        .iter()
        .rev()
        .find(|line| line.contains("TIMING") && line.contains("\"first_render\""))
        .and_then(|line| session_id_of(line))
        .context(
            "No TUI session found in the log file. Startup timings are only recorded \
             when psychic runs as a TUI, not for CLI subcommands.",
        )?;

    println!("Latest session: {}", session_id);
    println!();
    println!("Timing breakdown (in milliseconds):");
    println!("====================================");

    // Parse timing events for this session (only startup sequence)
    // Stop collecting after we see "startup_complete" to avoid duplicate queries
    let mut timing_events = Vec::new();
    let mut seen_startup_complete = false;

    for line in &lines {
        if !line.contains(session_id) || !line.contains("TIMING") {
            continue;
        }

        // Extract JSON portion
        if let Some(start) = line.find('{')
            && let Some(end) = line.rfind('}')
        {
            let json = &line[start..=end];
            if let Ok(event) = serde_json::from_str::<TimingEvent>(json) {
                // Stop after startup complete to avoid showing subsequent queries
                let is_startup_complete = event.op == "startup_complete";
                timing_events.push(event);

                if is_startup_complete {
                    seen_startup_complete = true;
                    break;
                }
            }
        }
    }

    if !seen_startup_complete && !timing_events.is_empty() {
        // If we didn't see startup_complete, we might be looking at an incomplete session
        // Just show what we have
    }

    // Find the longest op name for alignment
    let max_op_len = timing_events.iter().map(|e| e.op.len()).max().unwrap_or(0);

    // Print timing breakdown with column alignment
    for event in &timing_events {
        if event.op == "query" {
            print_query(event, max_op_len + 2);
        } else if let Some(avg_ms) = event.avg_ms {
            // ML feature timing with average
            let total_ms = event.total_ms.unwrap_or(0.0);
            println!(
                "{:<8.2}{:<width$}(avg per file, total: {:.2}ms)",
                avg_ms,
                event.op,
                total_ms,
                width = max_op_len + 2
            );
        } else if let Some(ms) = event.ms {
            // Regular timing
            if let Some(count) = event.count {
                println!(
                    "{:<8.2}{:<width$}({} items)",
                    ms,
                    event.op,
                    count,
                    width = max_op_len + 2
                );
            } else {
                println!("{:<8.2}{}", ms, event.op);
            }
        }
    }

    // Print summary section
    println!();
    println!("Total startup time:");

    let key_ops = [
        "first_query_complete",
        "query",
        "worker_state_new_total",
        "load_clicks_total",
        "main_setup_total",
    ];

    let mut summary_events: Vec<&TimingEvent> = timing_events
        .iter()
        .filter(|e| {
            key_ops.contains(&e.op.as_str()) || e.op.contains("total") || e.op.contains("complete")
        })
        .collect();

    // Sort by ms descending
    summary_events.sort_by(|a, b| {
        let a_ms = a.headline_ms().unwrap_or(0.0);
        let b_ms = b.headline_ms().unwrap_or(0.0);
        b_ms.partial_cmp(&a_ms).unwrap_or(std::cmp::Ordering::Equal)
    });

    for event in summary_events {
        if let Some(ms) = event.headline_ms() {
            println!("\"{:.2}ms  {}\"", ms, event.op);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    //! The other half of the contract with `search_worker.rs`: whatever
    //! `QueryTimings::to_json` writes, this file has to be able to read.

    use super::*;

    /// Exactly what a query line looks like in app.log, minus the `TIMING `
    /// prefix. Pinned identically by `query_timing_tests` on the writing side.
    const QUERY_LINE: &str = r#"{"blend_ms":0.125,"count":171,"features_ms":0.75,"filter_ms":1.5,"op":"query","per_feature":{"clicks_for_this_query":0.0,"fuzzy_score":0.25,"log_file_size":0.5},"predict_ms":0.5,"simple_ms":0.25,"total_ms":3.25}"#;

    #[test]
    fn test_a_query_line_parses_into_its_breakdown() {
        let event: TimingEvent = serde_json::from_str(QUERY_LINE).expect("a query line parses");

        assert_eq!(event.op, "query");
        assert_eq!(event.total_ms, Some(3.25));
        assert_eq!(event.filter_ms, Some(1.5));
        assert_eq!(event.features_ms, Some(0.75));
        assert_eq!(event.count, Some(171));

        let per_feature = event.per_feature.expect("the per-feature map came through");
        assert_eq!(per_feature.get("log_file_size"), Some(&0.5));
        assert_eq!(per_feature.len(), 3);
    }

    #[test]
    fn test_a_query_is_summarised_by_its_total_not_its_missing_ms() {
        // Every other line carries `ms`; this one carries `total_ms`, and the
        // totals list ranks by whichever the line actually has.
        let query: TimingEvent = serde_json::from_str(QUERY_LINE).unwrap();
        assert_eq!(query.headline_ms(), Some(3.25));

        let ordinary: TimingEvent =
            serde_json::from_str(r#"{"op":"first_render","ms":3.3}"#).unwrap();
        assert_eq!(ordinary.headline_ms(), Some(3.3));

        // A line with neither is not something to rank.
        let feature: TimingEvent =
            serde_json::from_str(r#"{"op":"walker_started","count":12}"#).unwrap();
        assert_eq!(feature.headline_ms(), None);
    }

    #[test]
    fn test_a_report_reads_a_log_with_a_query_in_it() {
        let dir = std::env::temp_dir().join(format!("psychic-perf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let log_path = dir.join("app.log");
        std::fs::write(
            &log_path,
            format!(
                "[2026-09-09 10:00:00 INFO psychic 42] TIMING {{\"op\":\"first_render\",\"ms\":3.3}}\n\
                 [2026-09-09 10:00:00 INFO psychic 42] TIMING {}\n\
                 [2026-09-09 10:00:00 INFO psychic 42] TIMING {{\"op\":\"startup_complete\",\"ms\":49.7}}\n",
                QUERY_LINE
            ),
        )
        .unwrap();

        analyze_perf(&log_path).expect("a log with a query line reports cleanly");

        std::fs::remove_dir_all(&dir).ok();
    }
}
