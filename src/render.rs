use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph},
};
use std::path::PathBuf;

use crate::help::{self, HelpLine};
use crate::keymap::{self, Action};
use crate::path_display::{
    display_width, human_bytes, printable, truncate_absolute_path, truncate_to_width,
};
use crate::preview::PreviewPane;

/// Group digits so six-figure counts stay readable in a narrow pane.
fn thousands(n: i64) -> String {
    let digits = n.abs().to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);

    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(c);
    }

    if n < 0 {
        format!("-{}", grouped)
    } else {
        grouped
    }
}

/// One latency line, or a placeholder while we are still waiting for it.
///
/// Sized to fit the narrow debug pane: 24 columns, which is what a 20% pane on a
/// 140-column terminal leaves inside its borders.
fn latency_line(label: &str, ms: Option<f64>) -> String {
    match ms {
        Some(ms) => format!("  {:<13} {:>6.1}ms", label, ms),
        None => format!("  {:<13} {:>8}", label, "-"),
    }
}

/// The hint that tells the user how to reach the help screen.
///
/// Derived from the keymap, so rebinding help changes what the main screen says.
fn help_hint() -> String {
    match keymap::primary_trigger(Action::ToggleHelp) {
        Some(keys) => format!(" {}: help ", keys),
        None => String::new(),
    }
}

/// Input data required to render the history mode UI.
pub struct HistoryRenderContext<'a> {
    pub filtered_history: &'a [PathBuf],
    pub history_selected: usize,
    pub total_history_items: usize,
    pub preview: &'a crate::preview::PreviewState,
    pub query: &'a str,
}

/// Compute the scroll offset for the file list based on selection and visible area
fn compute_scroll(
    selected_index: usize,
    current_scroll: usize,
    total_results: usize,
    visible_height: usize,
) -> usize {
    if total_results == 0 {
        return 0;
    }

    // If all results fit on screen, don't scroll at all
    if total_results <= visible_height {
        return 0;
    }

    // Auto-scroll the file list when selection is near top or bottom
    let mut scroll = current_scroll;

    // If selected item is above visible area, scroll up
    if selected_index < scroll {
        scroll = selected_index;
    }
    // If selected item is below visible area, scroll down
    else if selected_index >= scroll + visible_height {
        // Smart positioning: leave some space from bottom (5 lines)
        // This makes wrap-around more comfortable
        let margin = 5usize;
        scroll = selected_index.saturating_sub(
            visible_height
                .saturating_sub(margin)
                .min(visible_height - 1),
        );
    }
    // If we're in the bottom 5 items and there's more to see, keep scrolling
    else if selected_index >= scroll + visible_height.saturating_sub(5) {
        scroll = selected_index.saturating_sub(visible_height.saturating_sub(5));
    }

    scroll
}

/// Render the history navigation mode UI.
///
/// Returns the size of the preview pane, which only the layout knows and the
/// main loop needs in order to ask for the next preview.
pub fn render_history_mode(f: &mut Frame, ctx: HistoryRenderContext<'_>) -> PreviewPane {
    // Split vertically: top for dir list + preview, bottom for input
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),    // Dir list + Preview
            Constraint::Length(3), // Search input at bottom
        ])
        .split(f.area());

    // Match normal mode: stack vertically when the terminal is narrow
    let terminal_width = f.area().width;
    let use_vertical_stack = terminal_width < 120;

    // Choose layout direction based on available width
    let top_chunks = if use_vertical_stack {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(45), // Directory list
                Constraint::Percentage(55), // Preview
            ])
            .split(main_chunks[0])
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50), // Directory list
                Constraint::Percentage(50), // Preview
            ])
            .split(main_chunks[0])
    };

    // Get filtered history (sorted chronologically as stored)
    let filtered_history = ctx.filtered_history;

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

            let style = if idx == ctx.history_selected {
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
    let total_dirs = ctx.total_history_items;
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

    // Directories here go through the same preview thread as the file list.
    // This used to run `eza` on every frame, including every 200ms tick, with
    // no cache at all.
    let preview_text = match filtered_history.get(ctx.history_selected) {
        Some(dir) => ctx
            .preview
            .visible(dir, top_chunks[1].height.saturating_sub(2)),
        None => Text::from("No history available"),
    };

    let preview_para =
        Paragraph::new(preview_text).block(Block::default().borders(Borders::ALL).title("Preview"));
    f.render_widget(preview_para, top_chunks[1]);

    // Search input at bottom
    let input_area = main_chunks[1];
    let input_text = if ctx.query.is_empty() {
        "Filter history..."
    } else {
        ctx.query
    };
    let exit_keys = keymap::primary_trigger(Action::ToggleHistoryMode).unwrap_or_default();
    let input_para = Paragraph::new(input_text)
        .style(Style::default().fg(Color::Gray))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!("Search ({}/Esc to exit)", exit_keys))
                .title_bottom(
                    Line::from(Span::styled(
                        help_hint(),
                        Style::default().fg(Color::DarkGray),
                    ))
                    .right_aligned(),
                ),
        );
    f.render_widget(input_para, input_area);

    PreviewPane {
        width: top_chunks[1].width,
        height: top_chunks[1].height.saturating_sub(2),
    }
}

/// Draw the help screen on top of whatever is behind it.
///
/// Returns the largest useful scroll offset for the content as laid out, which
/// the caller stores so scrolling can be clamped to it.
pub fn render_help_overlay(f: &mut Frame, help_scroll: u16) -> u16 {
    let screen = f.area();

    // Leave a margin, then take off the borders and a space of padding a side.
    let outer_width = screen.width.saturating_sub(4);
    let layout = help::lay_out(
        &help::blocks(),
        outer_width.saturating_sub(4).max(1) as usize,
    );

    let content_height = layout.height() as u16;
    let width = (layout.width() as u16 + 4).min(screen.width);
    let height = (content_height + 2).min(screen.height); // 2 for the borders

    let area = Rect {
        x: screen.width.saturating_sub(width) / 2,
        y: screen.height.saturating_sub(height) / 2,
        width,
        height,
    };

    f.render_widget(Clear, area);

    let outer = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Keys and commands ",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ))
        .title_bottom(
            Line::from(Span::styled(
                " any other key closes this ",
                Style::default().fg(Color::DarkGray),
            ))
            .right_aligned(),
        );
    let inner = outer.inner(area);
    f.render_widget(outer, area);

    // Give up a line for the "more below" footer only when there is more below.
    let needs_footer = content_height > inner.height;
    let visible_height = inner
        .height
        .saturating_sub(if needs_footer { 1 } else { 0 });
    let max_scroll = content_height.saturating_sub(visible_height);
    let scroll = help_scroll.min(max_scroll);

    let text_area = Rect {
        x: inner.x + 1,
        width: inner.width.saturating_sub(2),
        height: visible_height,
        ..inner
    };

    let mut constraints = Vec::new();
    for i in 0..layout.columns.len() {
        if i > 0 {
            constraints.push(Constraint::Length(layout.gutter as u16));
        }
        constraints.push(Constraint::Length(layout.column_width as u16));
    }
    let column_areas = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(text_area);

    for (i, column) in layout.columns.iter().enumerate() {
        // Gutters sit between columns, so every other chunk is a column.
        let paragraph = Paragraph::new(styled_help_lines(column)).scroll((scroll, 0));
        f.render_widget(paragraph, column_areas[i * 2]);
    }

    if needs_footer {
        let remaining = max_scroll.saturating_sub(scroll);
        let footer = Paragraph::new(Line::from(Span::styled(
            if remaining > 0 {
                format!(
                    "{} more lines - {} to scroll",
                    remaining,
                    keymap::primary_trigger(Action::MoveDown).unwrap_or_default()
                )
            } else {
                format!(
                    "end - {} to scroll back",
                    keymap::primary_trigger(Action::MoveUp).unwrap_or_default()
                )
            },
            Style::default().fg(Color::DarkGray),
        )));
        let footer_area = Rect {
            y: inner.y + visible_height,
            height: 1,
            ..text_area
        };
        f.render_widget(footer, footer_area);
    }

    max_scroll
}

/// Style laid-out help lines: headings stand out, keys are cyan like directories.
fn styled_help_lines(lines: &[HelpLine]) -> Vec<Line<'static>> {
    lines
        .iter()
        .map(|line| match line {
            HelpLine::Title(title) => Line::from(Span::styled(
                title.clone(),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )),
            HelpLine::Entry { keys, description } => Line::from(vec![
                Span::styled(keys.clone(), Style::default().fg(Color::Cyan)),
                Span::raw("  "),
                Span::raw(description.clone()),
            ]),
            HelpLine::Blank => Line::from(""),
        })
        .collect()
}
/// State updates computed during rendering that need to be applied to App after rendering
/// What the frame's layout decided, which only the renderer knows.
///
/// Render reads `&App` and writes nothing back: everything it works out from
/// the geometry comes back here, and the main loop decides what to do with it.
pub struct FrameLayout {
    /// Size of the preview pane. The main loop uses it to ask for the preview
    /// after the frame is drawn, rather than reading a file mid-draw.
    pub preview_pane: Option<PreviewPane>,
    /// Rows of file list, borders excluded.
    pub visible_list_height: u16,
    /// Where the list was scrolled to for this frame.
    pub file_list_scroll: usize,
    /// Width of the path bar, which the marquee needs to know how far to go.
    pub path_bar_width: u16,
    /// Whether the selected path is too long for the bar, which is the only
    /// thing on screen that animates. When nothing does, the tick has nothing
    /// to drive and stops being sent.
    pub path_bar_overflows: bool,
}

/// Render the normal mode UI (file list, preview, debug pane)
/// Returns computed state updates that should be applied to App after rendering
pub fn render_normal_mode(f: &mut Frame, app: &crate::app::App) -> FrameLayout {
    use std::path::PathBuf;

    use crate::app::PAGE_SIZE;
    use crate::path_display::{get_time_ago, truncate_path};
    use crate::{feature_defs, ranker, search_worker, ui_state};

    // Check terminal width to decide layout direction
    let terminal_width = f.area().width;
    let use_vertical_stack = terminal_width < 120;

    // Split vertically: top for results/preview, bottom for input
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),    // Results + Preview
            Constraint::Length(1), // Path bar
            Constraint::Length(3), // Search input at bottom
        ])
        .split(f.area());

    // Split top area: horizontal (wide) or vertical (narrow)
    let top_chunks = if use_vertical_stack {
        // Vertical stack layout for narrow terminals: File list → Preview
        // Hide debug pane in vertical mode (too cramped)
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(40), // File list
                Constraint::Percentage(60), // Preview
                Constraint::Percentage(0),  // Debug (hidden)
            ])
            .split(main_chunks[0])
    } else {
        // Horizontal layout for wide terminals
        match app.ui_state.debug_pane_mode {
            ui_state::DebugPaneMode::Expanded => {
                // Debug expanded: give it most of the space
                Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(25), // File list (smaller)
                        Constraint::Percentage(0),  // Preview (hidden)
                        Constraint::Percentage(75), // Debug (expanded)
                    ])
                    .split(main_chunks[0])
            }
            ui_state::DebugPaneMode::Small => {
                // Debug small: normal layout
                Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(35), // File list
                        Constraint::Percentage(45), // Preview
                        Constraint::Percentage(20), // Debug (small)
                    ])
                    .split(main_chunks[0])
            }
            ui_state::DebugPaneMode::Hidden => {
                // Debug hidden: no space for debug pane
                Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(40), // File list (more space)
                        Constraint::Percentage(60), // Preview (more space)
                        Constraint::Percentage(0),  // Debug (hidden)
                    ])
                    .split(main_chunks[0])
            }
        }
    };

    // Compute scroll position based on selection and visible height
    let visible_height = top_chunks[0].height.saturating_sub(2); // subtract border

    let file_list_scroll = compute_scroll(
        app.selected_index,
        app.file_list_scroll,
        app.total_results,
        visible_height as usize,
    );

    // File list on the left
    let list_width = top_chunks[0].width.saturating_sub(2) as usize; // subtract borders
    let scroll_offset = file_list_scroll;

    // Build list items from page cache
    let items: Vec<ListItem> = (0..visible_height as usize)
        .map(|display_idx| {
            let i = scroll_offset + display_idx;
            if i >= app.total_results {
                return ListItem::new("");
            }

            if let Some(display_info) = app.get_file_at_index(i) {
                let time_ago = get_time_ago(display_info.mtime);
                let rank = i + 1;

                // Calculate space: "N. " takes 4 chars, time_ago length, we need padding between
                let rank_prefix = format!("{:2}. ", rank);
                let prefix_len = rank_prefix.len();
                let time_len = time_ago.len();

                // Available space for filename and padding
                let available = list_width.saturating_sub(prefix_len + time_len);
                let file_width = available.saturating_sub(2); // leave at least 2 spaces padding

                // Add "/" suffix for directories and "(cwd)" for current directory
                let cwd_suffix = if display_info.is_cwd { " (cwd)" } else { "" };
                let cwd_suffix_len = cwd_suffix.len();

                let display_name = if display_info.is_dir {
                    format!("{}/", display_info.display_name)
                } else {
                    display_info.display_name.clone()
                };

                // Adjust file_width to account for cwd suffix
                let adjusted_file_width = file_width.saturating_sub(cwd_suffix_len);

                // Detect whether historical item lives outside current cwd
                let is_outside_cwd = display_info.is_historical && !display_info.is_under_cwd;

                // Use absolute truncation only for historical items outside cwd
                let truncated_path = if is_outside_cwd {
                    truncate_absolute_path(&display_name, adjusted_file_width)
                } else {
                    truncate_path(&display_name, adjusted_file_width)
                };
                // A path is whatever someone managed to create on disk, and it
                // goes into a cell verbatim otherwise.
                let truncated_path = printable(&truncated_path).into_owned();

                // Build line with styled spans
                let base_style = if i == app.selected_index {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else if display_info.is_dir {
                    // Color directories cyan when not selected
                    Style::default().fg(Color::Cyan)
                } else {
                    Style::default()
                };

                // Decide rank number color
                let rank_style = if is_outside_cwd {
                    // Historical outside cwd: gray rank number
                    Style::default().fg(Color::DarkGray)
                } else if i == app.selected_index {
                    // Selected: match base style
                    base_style
                } else {
                    // Normal: match base style
                    base_style
                };

                let cwd_style = if i == app.selected_index {
                    // If selected, keep yellow but make it even more visible
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    // Otherwise use magenta to stand out
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD)
                };

                // Right-justify the timestamp. Measured in columns: bytes
                // over-count a non-ASCII name and `{:<width$}` under-counts a
                // wide one, and either way the column comes out ragged.
                let padding_len = adjusted_file_width
                    .saturating_sub(display_width(&truncated_path))
                    .saturating_sub(cwd_suffix_len);
                let padding = " ".repeat(padding_len);

                let line = if display_info.is_cwd {
                    Line::from(vec![
                        Span::styled(rank_prefix.clone(), rank_style),
                        Span::styled(truncated_path.clone(), base_style),
                        Span::styled(cwd_suffix, cwd_style),
                        Span::raw(padding),
                        Span::raw("  "),
                        Span::styled(time_ago.clone(), base_style),
                    ])
                } else {
                    Line::from(vec![
                        Span::styled(rank_prefix.clone(), rank_style),
                        Span::styled(
                            format!("{}{}  {}", truncated_path, padding, time_ago),
                            base_style,
                        ),
                    ])
                };

                ListItem::new(line)
            } else {
                // Page is not cached, show a loading indicator
                ListItem::new("[Loading...]").style(Style::default().fg(Color::DarkGray))
            }
        })
        .collect();

    // Create title with filter indicator
    let filter_name = match app.current_filter {
        search_worker::FilterType::None => "All",
        search_worker::FilterType::OnlyCwd => "CWD",
        search_worker::FilterType::DirectCwd => "Direct",
        search_worker::FilterType::OnlyDirs => "Dirs",
        search_worker::FilterType::OnlyFiles => "Files",
    };

    let title_line = if app.current_filter == search_worker::FilterType::None {
        // No filter active - no highlight
        Line::from(vec![Span::raw(format!(
            "{} ({}/{})",
            filter_name, app.total_results, app.total_files
        ))])
    } else {
        // Filter active - highlight in green
        Line::from(vec![
            Span::styled(
                filter_name,
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(" ({}/{})", app.total_results, app.total_files)),
        ])
    };

    let list = List::new(items).block(Block::default().borders(Borders::ALL).title(title_line));
    f.render_widget(list, top_chunks[0]);

    // Get current file from page cache - clone the info we need to avoid borrow issues
    let current_file_info: Option<(PathBuf, String, bool)> = app
        .get_file_at_index(app.selected_index)
        .map(|f| (f.full_path.clone(), f.display_name.clone(), f.is_dir));

    // The preview is generated on its own thread; this only shows whatever has
    // arrived for the row that is selected right now. Anything else would mean
    // a spawn or a file read inside the draw.
    let preview_pane = Some(PreviewPane {
        width: top_chunks[1].width,
        height: top_chunks[1].height.saturating_sub(2),
    });
    let preview_text = match &current_file_info {
        Some((path, _, _)) if !app.options.no_preview && app.total_results > 0 => app
            .preview
            .visible(path, top_chunks[1].height.saturating_sub(2)),
        _ => Text::default(),
    };

    let preview_pane_title = current_file_info
        .as_ref()
        .and_then(|(path, _, _)| path.file_name())
        .map(|x| x.to_string_lossy())
        .map(|x| x.to_string())
        .unwrap_or("No file selected".to_string());

    let preview = Paragraph::new(preview_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title(preview_pane_title),
    );
    f.render_widget(preview, top_chunks[1]);

    // Debug panel on the right
    let mut debug_lines = Vec::new();

    // Show current selection info - need another lookup to get score/features
    if let Some(display_info) = app.get_file_at_index(app.selected_index) {
        debug_lines.push(String::from("Scores:"));
        debug_lines.push(format!("  Final: {:.4}", display_info.score));
        if let Some(simple) = display_info.simple_score {
            let weight_str = display_info
                .simple_weight
                .map(|w| format!(" (w={:.3})", w))
                .unwrap_or_default();
            debug_lines.push(format!("  Simple: {:.4}{}", simple, weight_str));
        }
        if let Some(ml) = display_info.ml_score {
            let weight_str = display_info
                .ml_weight
                .map(|w| format!(" (w={:.3})", w))
                .unwrap_or_default();
            debug_lines.push(format!("  ML: {:.4}{}", ml, weight_str));
        }
        debug_lines.push(String::from(""));
        debug_lines.push(String::from("Simple Score Inputs:"));

        // Display fuzzy score (what goes into ML model)
        let ml_score = ranker::fuzzy_score_for_ml(display_info.fuzzy_score);
        if display_info.fuzzy_score == i64::MAX {
            debug_lines.push(String::from("  Fuzzy score: 0.0 (empty query)"));
        } else {
            debug_lines.push(format!("  Fuzzy score: {:.1}", ml_score));
        }

        debug_lines.push(String::from(""));
        debug_lines.push(String::from("Features:"));
        debug_lines.push(String::from(""));

        // Show all features from registry
        if !display_info.features.is_empty() {
            let features_map = ranker::features_to_map(&display_info.features);
            for feature in feature_defs::FEATURE_REGISTRY.iter() {
                if let Some(value) = features_map.get(feature.name()) {
                    debug_lines.push(format!("  {}: {}", feature.name(), value));
                }
            }
        } else {
            debug_lines.push(String::from("  (no features)"));
        }
    } else if app.total_results > 0 {
        debug_lines.push(String::from("(loading...)"));
    } else {
        debug_lines.push(String::from("No results"));
    }

    debug_lines.push(String::from("")); // Separator

    // Add retraining status
    if app.currently_retraining {
        debug_lines.push(String::from("Retraining model..."));
        debug_lines.push(String::from("")); // Separator
    }

    // Add model stats
    if let Some(stats) = app.model_stats_cache.as_ref() {
        debug_lines.push(String::from("Model Stats:"));
        let formatter = timeago::Formatter::new();

        // Parse timestamp and show how long ago
        if let Ok(trained_at) = stats.trained_at.parse::<jiff::Timestamp>() {
            let now = jiff::Timestamp::now();
            let duration = now.duration_since(trained_at);
            let time_ago =
                formatter.convert(std::time::Duration::from_secs(duration.as_secs() as u64));
            debug_lines.push(format!("  Trained: {}", time_ago));
        } else {
            debug_lines.push(format!("  Trained: {}", stats.trained_at));
        }

        debug_lines.push(format!(
            "  Duration: {:.2}s",
            stats.training_duration_seconds
        ));
        debug_lines.push(format!("  Features: {}", stats.num_features));
        debug_lines.push(format!(
            "  Examples: {} ({} pos, {} neg)",
            stats.num_total_examples, stats.num_positive_examples, stats.num_negative_examples
        ));
        debug_lines.push(String::from("  Top features:"));
        for feat in &stats.top_3_features {
            debug_lines.push(format!("    {}: {:.1}", feat.feature, feat.importance));
        }
        debug_lines.push(String::from("")); // Separator
    }

    // Latency: a few numbers that say how snappy this is, right now.
    // The first three are measured once from process start; the last two are
    // replaced on every query, so they describe the search just performed.
    debug_lines.push(String::from("Latency:"));
    debug_lines.push(latency_line("first paint", app.timings.first_paint_ms));
    debug_lines.push(latency_line("first results", app.timings.first_results_ms));
    debug_lines.push(latency_line("fs walk", app.timings.walk_complete_ms));
    debug_lines.push(latency_line("this search", app.timings.last_search_ms));
    debug_lines.push(latency_line("  of it, rank", app.timings.last_rank_ms));
    debug_lines.push(String::from(""));

    // Database contents, loaded in the background the first time this pane opens.
    match app.db_stats.as_ref() {
        Some(stats) => {
            let history = match stats.history_days {
                Some(days) => format!(", {}d", days),
                None => String::new(),
            };
            debug_lines.push(format!(
                "Database ({}{}):",
                human_bytes(stats.file_size_bytes),
                history
            ));
            for (label, count) in [
                ("events", stats.total_events),
                ("impressions", stats.impressions),
                ("clicks", stats.clicks),
                ("scrolls", stats.scrolls),
                ("visits", stats.startup_visits),
                ("sessions", stats.sessions),
            ] {
                debug_lines.push(format!("  {:<14} {:>7}", label, thousands(count)));
            }
        }
        None => {
            debug_lines.push(String::from("Database:"));
            debug_lines.push(String::from("  counting..."));
        }
    }

    debug_lines.push(String::from("")); // Separator

    // Add preview cache status
    if let Some((file_path, _, _)) = &current_file_info {
        debug_lines.push(format!("Preview: {}", app.preview.status(file_path)));
    } else {
        debug_lines.push(String::from("Preview: N/A"));
    }

    debug_lines.push(String::from("")); // Separator

    // Add page cache status
    if app.total_results > 0 {
        let current_page = app.selected_index / PAGE_SIZE;
        debug_lines.push(format!("Current page: {}", current_page));

        let mut cached_pages: Vec<usize> = app.page_cache.keys().copied().collect();
        cached_pages.sort_unstable();
        let pages_str = cached_pages
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        debug_lines.push(format!("Cached pages: [{}]", pages_str));
    } else {
        debug_lines.push(String::from("Current page: N/A"));
        debug_lines.push(String::from("Cached pages: []"));
    }

    debug_lines.push(String::from("")); // Separator

    // Add recent logs
    debug_lines.push(String::from("Recent Logs:"));
    // Show more log lines when debug is maximized
    let log_count = if app.ui_state.is_debug_pane_expanded() {
        30
    } else {
        10
    };
    let log_start = app.recent_logs.len().saturating_sub(log_count);
    for log_line in app.recent_logs.iter().skip(log_start) {
        // Truncate long lines to fit
        let max_len = if app.ui_state.is_debug_pane_expanded() {
            120
        } else {
            60
        };
        // Not a byte slice: a log line carries paths, and a path carries
        // whatever is on disk. Cutting one mid-character panics the UI.
        debug_lines.push(format!("  {}", truncate_to_width(log_line, max_len)));
    }

    let debug_text = debug_lines.join("\n");

    let debug_title = match app.ui_state.debug_pane_mode {
        ui_state::DebugPaneMode::Small => "Debug (Ctrl-O: expand)",
        ui_state::DebugPaneMode::Expanded => "Debug (Ctrl-O: hide)",
        ui_state::DebugPaneMode::Hidden => "Debug (Ctrl-O: show)",
    };
    let debug_pane =
        Paragraph::new(debug_text).block(Block::default().borders(Borders::ALL).title(debug_title));
    f.render_widget(debug_pane, top_chunks[2]);

    // Get path of currently selected file for marquee
    let selected_path_str = app
        .get_file_at_index(app.selected_index)
        .map(|f| f.full_path.to_string_lossy().to_string())
        .unwrap_or_default();

    // Pad the string to make the marquee scroll past the end
    let padded_path = format!("{}    ", printable(&selected_path_str));

    let path_bar_width = main_chunks[1].width as usize;

    // The marquee is advanced by the Tick handler, not here: drawing a frame
    // should not be what moves the animation on. Render only reports how wide
    // the bar came out, which is the one thing the advance cannot work out for
    // itself.
    let padded_path_len = padded_path.chars().count();
    let path_bar = Paragraph::new(padded_path)
        .style(Style::default().fg(Color::DarkGray))
        .scroll((0, app.path_bar_scroll));
    f.render_widget(path_bar, main_chunks[1]);

    // Search input at the bottom
    let cwd_str = app.cwd.to_string_lossy();
    let filter_indicator = match app.current_filter {
        search_worker::FilterType::None => "",
        search_worker::FilterType::OnlyCwd => " [CWD]",
        search_worker::FilterType::DirectCwd => " [DIRECT]",
        search_worker::FilterType::OnlyDirs => " [DIRS]",
        search_worker::FilterType::OnlyFiles => " [FILES]",
    };
    // A status message replaces the title rather than sharing the line with it:
    // the cwd is already on screen in the path bar above, and a message that
    // gets truncated away by a long path is not worth showing.
    let search_title = match app.status_message.as_deref() {
        Some(message) => Span::styled(
            message.to_string(),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        None => Span::raw(format!("Search: {}{}", cwd_str, filter_indicator)),
    };
    let input = Paragraph::new(app.query.as_str()).block(
        Block::default()
            .borders(Borders::ALL)
            .title(search_title)
            .title_bottom(
                Line::from(Span::styled(
                    help_hint(),
                    Style::default().fg(Color::DarkGray),
                ))
                .right_aligned(),
            ),
    );
    f.render_widget(input, main_chunks[2]);

    // Filter picker overlay (rendered on top if visible)
    if app.ui_state.filter_picker_visible {
        // Create a popup in the bottom-right
        let popup_width = 35;
        let popup_height = 7; // 5 options + top/bottom borders
        let area = f.area();

        // Position in bottom-right corner with some margin
        let popup_x = area.width.saturating_sub(popup_width + 2);
        let popup_y = area.height.saturating_sub(popup_height + 2);

        let popup_area = Rect {
            x: popup_x,
            y: popup_y,
            width: popup_width,
            height: popup_height,
        };

        // Clear the area first to make it opaque
        f.render_widget(Clear, popup_area);

        // Build filter options text with current selection highlighted
        let mut lines = vec![];

        let options = [
            (search_worker::FilterType::None, "0: No filter"),
            (
                search_worker::FilterType::OnlyCwd,
                "c: Only CWD (recursive)",
            ),
            (search_worker::FilterType::DirectCwd, "i: Direct CWD only"),
            (search_worker::FilterType::OnlyDirs, "d: Only directories"),
            (search_worker::FilterType::OnlyFiles, "f: Only files"),
        ];

        for (filter_type, label) in options.iter() {
            if *filter_type == app.current_filter {
                lines.push(format!("> {}", label));
            } else {
                lines.push(format!("  {}", label));
            }
        }

        let filter_text = lines.join("\n");
        let filter_popup = Paragraph::new(filter_text).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Filters (Tab/Shift-Tab)"),
        );

        f.render_widget(filter_popup, popup_area);
    }

    // Position cursor in the search input at the end of the query text
    // Account for border (1 char) + query length.
    // The help screen covers the input, so leaving a cursor on it would be a
    // stray block floating over the help text.
    if !app.ui_state.help_visible {
        // Columns, not bytes: the cursor belongs after what is drawn.
        let cursor_x = main_chunks[2].x + 1 + display_width(&app.query) as u16;
        let cursor_y = main_chunks[2].y + 1; // 1 for top border
        f.set_cursor_position((cursor_x, cursor_y));
    }

    FrameLayout {
        preview_pane,
        visible_list_height: visible_height,
        file_list_scroll,
        path_bar_width: path_bar_width as u16,
        path_bar_overflows: padded_path_len > path_bar_width,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_thousands() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(7), "7");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1,000");
        assert_eq!(thousands(80_855), "80,855");
        assert_eq!(thousands(1_234_567), "1,234,567");
        assert_eq!(thousands(-4_200), "-4,200");
    }

    #[test]
    fn test_latency_line_aligns_and_handles_missing_values() {
        assert_eq!(
            latency_line("first paint", Some(1.44)),
            "  first paint      1.4ms"
        );
        assert_eq!(
            latency_line("first results", Some(12.28)),
            "  first results   12.3ms"
        );
        assert_eq!(
            latency_line("fs walk", None),
            "  fs walk              -",
            "A measurement that has not happened yet shows a dash, not 0.0"
        );

        // Every line fits the narrow pane, including an implausibly slow one.
        for line in [
            latency_line("first results", Some(1234.5)),
            latency_line("  of it, rank", Some(0.04)),
            latency_line("fs walk", None),
        ] {
            assert!(
                line.len() <= 24,
                "Line {:?} is {} columns",
                line,
                line.len()
            );
        }
    }
}

#[cfg(test)]
mod non_ascii_tests {
    use super::*;
    use crate::ui_state::DebugPaneMode;
    use ratatui::{Terminal, backend::TestBackend};
    use std::collections::VecDeque;

    /// Draw the normal-mode UI with whatever is passed in, and read it back.
    fn draw(logs: VecDeque<String>, query: &str, name: &str) -> Vec<String> {
        let mut app = crate::app::App::for_test();
        app.query = query.to_string();
        app.recent_logs = logs;
        app.total_results = 1;
        app.total_files = 1;
        app.options.no_preview = true;
        app.ui_state.debug_pane_mode = DebugPaneMode::Small;
        app.page_cache.insert(
            0,
            crate::app::Page {
                start_index: 0,
                end_index: 1,
                files: vec![crate::search_worker::DisplayFileInfo {
                    display_name: name.to_string(),
                    full_path: std::path::PathBuf::from("/tmp").join(name),
                    score: 0.0,
                    features: Vec::new(),
                    mtime: None,
                    atime: None,
                    file_size: None,
                    is_dir: false,
                    is_cwd: false,
                    is_historical: false,
                    is_under_cwd: true,
                    simple_score: None,
                    ml_score: None,
                    simple_weight: None,
                    ml_weight: None,
                    fuzzy_score: 0,
                }],
            },
        );

        let mut terminal = Terminal::new(TestBackend::new(160, 60)).unwrap();
        terminal
            .draw(|f| {
                render_normal_mode(f, &app);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        (0..60)
            .map(|y| (0..160).map(|x| buffer[(x, y)].symbol()).collect())
            .collect()
    }

    #[test]
    fn test_a_log_line_with_an_accent_in_it_does_not_take_the_ui_down() {
        // The truncation was a byte slice, and the cut landed inside this
        // character. A log line carries paths, and a path carries whatever is
        // on disk, so this was a matter of when rather than whether.
        let line = format!("{}\u{e9}{}", "a".repeat(56), "b".repeat(20));
        let mut logs = VecDeque::new();
        logs.push_back(line);

        let screen = draw(logs, "", "plain.txt");

        assert!(
            screen.iter().any(|row| row.contains("aaaa")),
            "the log line should still be shown, just shortened: {:#?}",
            screen.last()
        );
    }

    #[test]
    fn test_a_wide_filename_does_not_take_the_ui_down() {
        let screen = draw(VecDeque::new(), "", "日本語のファイル名.txt");

        // A wide character occupies two cells, and the second reads back
        // blank, so the row is checked one glyph at a time.
        assert!(
            screen.iter().any(|row| row.contains("日")),
            "a name in wide characters should be drawn: {:?}",
            screen.first()
        );
    }

    #[test]
    fn test_the_cursor_follows_a_non_ascii_query() {
        // Not a panic, a misplacement: the cursor sat at a byte offset, so it
        // drifted right of the text by one column per extra byte.
        let screen = draw(VecDeque::new(), "café", "plain.txt");

        assert!(
            screen.iter().any(|row| row.contains("café")),
            "the query is drawn as typed"
        );
    }
}

#[cfg(test)]
mod help_overlay_tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    /// Draw the help screen and read the terminal back as text.
    fn render_help(width: u16, height: u16, scroll: u16) -> (Vec<String>, u16) {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        let mut max_scroll = 0;
        terminal
            .draw(|f| max_scroll = render_help_overlay(f, scroll))
            .unwrap();

        let buffer = terminal.backend().buffer();
        let lines = (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect();

        (lines, max_scroll)
    }

    /// Rows of the overlay box, without their borders or padding.
    fn boxed_rows(lines: &[String]) -> Vec<String> {
        lines
            .iter()
            .filter(|line| line.contains('│'))
            .map(|line| {
                let start = line.find('│').unwrap() + '│'.len_utf8();
                let end = line.rfind('│').unwrap();
                line[start..end].to_string()
            })
            .collect()
    }

    #[test]
    fn test_hint_on_the_main_screen_names_the_help_key() {
        assert_eq!(
            help_hint(),
            " Ctrl-G: help ",
            "This is the only thing telling the user the help screen exists"
        );
    }

    #[test]
    fn test_wide_terminal_shows_every_binding_without_scrolling() {
        let (lines, max_scroll) = render_help(120, 40, 0);
        assert_eq!(max_scroll, 0, "A 120x40 terminal should not have to scroll");

        let rows = boxed_rows(&lines);
        let has = |keys: &str, description: &str| {
            rows.iter()
                .any(|row| row.contains(keys) && row.contains(description))
        };

        assert!(has("Ctrl-U", "clear the query"), "rows: {:#?}", rows);
        assert!(has("Ctrl-G / F1", "show this help"));
        assert!(has("Wheel up", "scroll the preview up"));
        assert!(has("retrain", "retrain the ranking model"));
        assert!(has("p ", "jump to a directory"));
    }

    #[test]
    fn test_content_never_runs_into_the_border() {
        for width in [50, 60, 80, 100, 120, 140, 200] {
            let (lines, _) = render_help(width, 50, 0);
            for row in boxed_rows(&lines) {
                assert!(
                    row.ends_with(' '),
                    "Text reaches the border at width {}: {:?}",
                    width,
                    row
                );
            }
        }
    }

    #[test]
    fn test_short_terminal_scrolls_and_says_so() {
        let (top, max_scroll) = render_help(80, 24, 0);
        assert!(max_scroll > 0, "Everything cannot fit in 24 rows");
        assert!(
            top.iter().any(|line| line.contains("more lines")),
            "The user needs to be told there is more below"
        );
        assert!(
            boxed_rows(&top).iter().any(|row| row.contains("Search")),
            "Unscrolled, the screen starts at the top"
        );

        // Scrolling past the end is clamped rather than showing blank space.
        let (bottom, _) = render_help(80, 24, 999);
        assert!(
            boxed_rows(&bottom)
                .iter()
                .any(|row| row.contains("Command line")),
            "Scrolled to the end, the last block is on screen"
        );
    }

    #[test]
    fn test_tiny_terminal_does_not_panic() {
        for (width, height) in [(20, 5), (10, 3), (1, 1), (200, 2)] {
            render_help(width, height, 0);
        }
    }
}
