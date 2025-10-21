use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};
use std::time::Instant;

use crate::app::App;
use crate::path_display::truncate_absolute_path;

/// Render the history navigation mode UI
pub fn render_history_mode(f: &mut Frame, app: &App) {
    // Split vertically: top for dir list + preview, bottom for input
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),    // Dir list + Preview
            Constraint::Length(3), // Search input at bottom
        ])
        .split(f.area());

    // Split horizontally: left for dir list, right for preview
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Directory list
            Constraint::Percentage(50), // Preview
        ])
        .split(main_chunks[0]);

    // Get filtered history (sorted chronologically as stored)
    let filtered_history = app.get_filtered_history();

    // Calculate list width (accounting for borders)
    let list_width = top_chunks[0].width.saturating_sub(2) as usize;

    // Build list items for directory history
    let items: Vec<ListItem> = filtered_history
        .iter()
        .enumerate()
        .map(|(idx, path)| {
            let rank = idx + 1;
            let rank_prefix = format!("{:2}. ", rank);
            let prefix_len = rank_prefix.len();
            let path_str = path.to_string_lossy();

            // Calculate available width for path (widget width - rank prefix - 1 for safety margin)
            let available_width = list_width.saturating_sub(prefix_len).saturating_sub(1);

            // Use truncate_absolute_path for good abbreviation
            let display_text = truncate_absolute_path(&path_str, available_width);

            let style = if idx == app.history_selected {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Cyan)
            };
            ListItem::new(Line::from(vec![
                Span::styled(rank_prefix, style),
                Span::styled(display_text, style),
            ]))
        })
        .collect();

    // Render directory list
    // Total includes all items in history display
    let total_dirs = app.history.items_for_display().len();
    let title = format!(
        "History (most recent at top) — {}/{}",
        filtered_history.len(),
        total_dirs
    );
    let list = List::new(items).block(
        Block::default().borders(Borders::ALL).title(Span::styled(
            title,
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )),
    );
    f.render_widget(list, top_chunks[0]);

    // Preview pane: show eza -al of selected directory
    let preview_text = if !filtered_history.is_empty()
        && app.history_selected < filtered_history.len()
    {
        let selected_dir = &filtered_history[app.history_selected];

        let preview_width = top_chunks[1].width;
        let extra_flags = app.ui_state.get_eza_flags(preview_width);

        let mut eza_cmd = std::process::Command::new("eza");
        eza_cmd.arg("-al").arg("--color=always");

        // Add extra flags if preview is narrow
        for flag in extra_flags.split_whitespace() {
            if !flag.is_empty() {
                eza_cmd.arg(flag);
            }
        }

        eza_cmd.arg(selected_dir);
        let eza_start = Instant::now();
        let eza_output = eza_cmd.output();
        log::info!(
            "TIMING {{\"op\":\"eza_history_preview\",\"ms\":{}}}",
            eza_start.elapsed().as_secs_f64() * 1000.0
        );

        match eza_output {
            Ok(output) => match ansi_to_tui::IntoText::into_text(&output.stdout) {
                Ok(text) => text,
                Err(_) => Text::from("[Unable to parse directory listing]"),
            },
            Err(_) => {
                // Fallback to ls if eza not available
                let ls_output = std::process::Command::new("ls")
                    .arg("-lah")
                    .arg(selected_dir)
                    .output();
                match ls_output {
                    Ok(output) => Text::from(String::from_utf8_lossy(&output.stdout).to_string()),
                    Err(_) => Text::from("[Unable to list directory]"),
                }
            }
        }
    } else {
        Text::from("No history available")
    };

    let preview_para = Paragraph::new(preview_text)
        .block(Block::default().borders(Borders::ALL).title("Preview"))
        .scroll((app.preview_scroll as u16, 0));
    f.render_widget(preview_para, top_chunks[1]);

    // Search input at bottom
    let input_area = main_chunks[1];
    let input_text = if app.query.is_empty() {
        "Filter history..."
    } else {
        &app.query
    };
    let input_para = Paragraph::new(input_text)
        .style(Style::default().fg(Color::Gray))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Search (Ctrl-H/Esc to exit)"),
        );
    f.render_widget(input_para, input_area);
}
