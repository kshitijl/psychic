use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Timing event from log file
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
        if let Some(avg_ms) = event.avg_ms {
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
        "filter_and_rank_total",
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
        let a_ms = a.ms.unwrap_or(0.0);
        let b_ms = b.ms.unwrap_or(0.0);
        b_ms.partial_cmp(&a_ms).unwrap_or(std::cmp::Ordering::Equal)
    });

    for event in summary_events {
        if let Some(ms) = event.ms {
            println!("\"{:.2}ms  {}\"", ms, event.op);
        }
    }

    Ok(())
}
