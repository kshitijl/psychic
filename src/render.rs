use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};
use std::{
    collections::{HashMap, VecDeque},
    path::{Path, PathBuf},
    time::Instant,
};

use crate::path_display::truncate_absolute_path;

/// Input data required to render the history mode UI.
pub struct HistoryRenderContext<'a> {
    pub filtered_history: &'a [PathBuf],
    pub history_selected: usize,
    pub total_history_items: usize,
    pub preview_scroll_position: u16,
    pub query: &'a str,
}

/// Input data required to render the normal mode UI.
pub struct NormalRenderContext<'a> {
    pub selected_index: usize,
    pub file_list_scroll: usize,
    pub total_results: usize,
    pub total_files: usize,
    pub current_filter: crate::search_worker::FilterType,
    pub no_preview: bool,
    pub preview: &'a crate::preview::PreviewManager,
    pub currently_retraining: bool,
    pub model_stats_cache: Option<&'a crate::ranker::ModelStats>,
    pub page_cache: &'a HashMap<usize, crate::app::Page>,
    pub ui_state: &'a crate::ui_state::UiState,
    pub recent_logs: &'a VecDeque<String>,
    pub last_path_bar_update: Instant,
    pub path_bar_scroll: u16,
    pub path_bar_scroll_direction: i8,
    pub cwd: &'a Path,
    pub query: &'a str,
}

impl<'a> NormalRenderContext<'a> {
    /// Look up a file in the paged cache by its global index.
    pub fn get_file_at_index(
        &self,
        index: usize,
    ) -> Option<&'a crate::search_worker::DisplayFileInfo> {
        if index >= self.total_results {
            return None;
        }

        let page_num = index / crate::app::PAGE_SIZE;
        let page = self.page_cache.get(&page_num)?;

        // Preconditions mirror App::get_file_at_index to keep invariants local.
        assert!(
            page.start_index < page.end_index,
            "Page has invalid range: [{}, {})",
            page.start_index,
            page.end_index
        );
        assert_eq!(
            page.end_index - page.start_index,
            page.files.len(),
            "Page length mismatch: expected {}, found {}",
            page.end_index - page.start_index,
            page.files.len()
        );

        assert!(
            index >= page.start_index && index < page.end_index,
            "Index {} outside page {} range [{}, {})",
            index,
            page_num,
            page.start_index,
            page.end_index
        );

        let rel_index = index - page.start_index;
        assert!(
            rel_index < page.files.len(),
            "Relative index {} out of bounds for page starting at {} with {} files",
            rel_index,
            page.start_index,
            page.files.len()
        );

        page.files.get(rel_index)
    }
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

/// Render the history navigation mode UI
pub fn render_history_mode(f: &mut Frame, ctx: HistoryRenderContext<'_>) {
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

    // Preview pane: show eza -al of selected directory
    let preview_text = if !filtered_history.is_empty()
        && ctx.history_selected < filtered_history.len()
    {
        let selected_dir = &filtered_history[ctx.history_selected];

        let preview_width = top_chunks[1].width;
        let extra_flags = get_eza_flags(preview_width);

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
        .scroll((ctx.preview_scroll_position, 0));
    f.render_widget(preview_para, top_chunks[1]);

    // Search input at bottom
    let input_area = main_chunks[1];
    let input_text = if ctx.query.is_empty() {
        "Filter history..."
    } else {
        ctx.query
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
/// State updates computed during rendering that need to be applied to App after rendering
pub struct RenderUpdates {
    pub file_list_scroll: Option<usize>,
    pub preview: Option<crate::preview::PreviewManager>,
    pub path_bar_scroll: Option<u16>,
    pub path_bar_scroll_direction: Option<i8>,
    pub last_path_bar_update: Option<std::time::Instant>,
}

impl RenderUpdates {
    pub fn new() -> Self {
        Self {
            file_list_scroll: None,
            preview: None,
            path_bar_scroll: None,
            path_bar_scroll_direction: None,
            last_path_bar_update: None,
        }
    }

    /// Apply the computed updates to the App
    pub fn apply_to(self, app: &mut crate::app::App) {
        if let Some(scroll) = self.file_list_scroll {
            app.file_list_scroll = scroll;
        }
        if let Some(preview) = self.preview {
            app.preview = preview;
        }
        if let Some(scroll) = self.path_bar_scroll {
            app.path_bar_scroll = scroll;
        }
        if let Some(dir) = self.path_bar_scroll_direction {
            app.path_bar_scroll_direction = dir;
        }
        if let Some(time) = self.last_path_bar_update {
            app.last_path_bar_update = time;
        }
    }
}

/// Render the normal mode UI (file list, preview, debug pane)
/// Returns computed state updates that should be applied to App after rendering
pub fn render_normal_mode(
    f: &mut Frame,
    ctx: NormalRenderContext<'_>,
    marquee_delay: std::time::Duration,
    marquee_speed: std::time::Duration,
) -> RenderUpdates {
    let mut updates = RenderUpdates::new();
    use ratatui::layout::Rect;
    use ratatui::widgets::Clear;
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
        match ctx.ui_state.debug_pane_mode {
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
        ctx.selected_index,
        ctx.file_list_scroll,
        ctx.total_results,
        visible_height as usize,
    );
    updates.file_list_scroll = Some(file_list_scroll);

    // File list on the left
    let list_width = top_chunks[0].width.saturating_sub(2) as usize; // subtract borders
    let scroll_offset = file_list_scroll;

    // Build list items from page cache
    let items: Vec<ListItem> = (0..visible_height as usize)
        .map(|display_idx| {
            let i = scroll_offset + display_idx;
            if i >= ctx.total_results {
                return ListItem::new("");
            }

            if let Some(display_info) = ctx.get_file_at_index(i) {
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

                // Build line with styled spans
                let base_style = if i == ctx.selected_index {
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
                } else if i == ctx.selected_index {
                    // Selected: match base style
                    base_style
                } else {
                    // Normal: match base style
                    base_style
                };

                let cwd_style = if i == ctx.selected_index {
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

                // Calculate padding to right-justify time
                let padding_len = adjusted_file_width.saturating_sub(truncated_path.len());
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
                            format!(
                                "{:<width$}  {}",
                                truncated_path,
                                time_ago,
                                width = adjusted_file_width
                            ),
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
    let filter_name = match ctx.current_filter {
        search_worker::FilterType::None => "All",
        search_worker::FilterType::OnlyCwd => "CWD",
        search_worker::FilterType::DirectCwd => "Direct",
        search_worker::FilterType::OnlyDirs => "Dirs",
        search_worker::FilterType::OnlyFiles => "Files",
    };

    let title_line = if ctx.current_filter == search_worker::FilterType::None {
        // No filter active - no highlight
        Line::from(vec![Span::raw(format!(
            "{} ({}/{})",
            filter_name, ctx.total_results, ctx.total_files
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
            Span::raw(format!(" ({}/{})", ctx.total_results, ctx.total_files)),
        ])
    };

    let list = List::new(items).block(Block::default().borders(Borders::ALL).title(title_line));
    f.render_widget(list, top_chunks[0]);

    // Get current file from page cache - clone the info we need to avoid borrow issues
    let current_file_info: Option<(PathBuf, String, bool)> = ctx
        .get_file_at_index(ctx.selected_index)
        .map(|f| (f.full_path.clone(), f.display_name.clone(), f.is_dir));
    let current_file_path = current_file_info
        .as_ref()
        .map(|(p, _, _)| p.to_string_lossy().to_string());
    let is_dir = current_file_info
        .as_ref()
        .map(|(_, _, d)| *d)
        .unwrap_or(false);

    // Preview on the right using bat/eza (with smart caching)
    let preview_text = if ctx.no_preview {
        // Skip preview generation when --no-preview is enabled
        Text::from("")
    } else if current_file_info.is_some() && ctx.total_results > 0 {
        if let Some(current_file_path) = &current_file_path {
            let preview_width = top_chunks[1].width;
            let preview_height = top_chunks[1].height.saturating_sub(2);

            // Use preview manager to render preview
            let mut preview_clone = ctx.preview.clone();
            let text = preview_clone.render(
                std::path::Path::new(current_file_path),
                is_dir,
                preview_width,
                preview_height,
            );
            updates.preview = Some(preview_clone);
            text
        } else {
            Text::from("[Loading preview...]")
        }
    } else {
        Text::from("")
    };

    let preview_pane_title = current_file_info
        .as_ref()
        .and_then(|(path, _, _)| path.file_name())
        .map(|x| x.to_string_lossy())
        .map(|x| x.to_string())
        .unwrap_or("No file selected".to_string());

    let preview = Paragraph::new(preview_text)
        .scroll((ctx.preview.scroll_position() as u16, 0))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(preview_pane_title),
        );
    f.render_widget(preview, top_chunks[1]);

    // Debug panel on the right
    let mut debug_lines = Vec::new();

    // Show current selection info - need another lookup to get score/features
    if let Some(display_info) = ctx.get_file_at_index(ctx.selected_index) {
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
    } else if ctx.total_results > 0 {
        debug_lines.push(String::from("(loading...)"));
    } else {
        debug_lines.push(String::from("No results"));
    }

    debug_lines.push(String::from("")); // Separator

    // Add retraining status
    if ctx.currently_retraining {
        debug_lines.push(String::from("Retraining model..."));
        debug_lines.push(String::from("")); // Separator
    }

    // Add model stats
    if let Some(stats) = ctx.model_stats_cache {
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

    // Add preview cache status
    if let Some((file_path, _, _)) = &current_file_info {
        let full_path_str = file_path.to_string_lossy().to_string();
        let cache_status = ctx.preview.status(&full_path_str);
        debug_lines.push(format!("Preview: {}", cache_status));
    } else {
        debug_lines.push(String::from("Preview: N/A"));
    }

    debug_lines.push(String::from("")); // Separator

    // Add page cache status
    if ctx.total_results > 0 {
        let current_page = ctx.selected_index / PAGE_SIZE;
        debug_lines.push(format!("Current page: {}", current_page));

        let mut cached_pages: Vec<usize> = ctx.page_cache.keys().copied().collect();
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
    let log_count = if ctx.ui_state.is_debug_pane_expanded() {
        30
    } else {
        10
    };
    let log_start = ctx.recent_logs.len().saturating_sub(log_count);
    for log_line in ctx.recent_logs.iter().skip(log_start) {
        // Truncate long lines to fit
        let max_len = if ctx.ui_state.is_debug_pane_expanded() {
            120
        } else {
            60
        };
        if log_line.len() > max_len {
            debug_lines.push(format!("  {}...", &log_line[..(max_len - 3)]));
        } else {
            debug_lines.push(format!("  {}", log_line));
        }
    }

    let debug_text = debug_lines.join("\n");

    let debug_title = match ctx.ui_state.debug_pane_mode {
        ui_state::DebugPaneMode::Small => "Debug (Ctrl-O: expand)",
        ui_state::DebugPaneMode::Expanded => "Debug (Ctrl-O: hide)",
        ui_state::DebugPaneMode::Hidden => "Debug (Ctrl-O: show)",
    };
    let debug_pane =
        Paragraph::new(debug_text).block(Block::default().borders(Borders::ALL).title(debug_title));
    f.render_widget(debug_pane, top_chunks[2]);

    // Get path of currently selected file for marquee
    let selected_path_str = ctx
        .get_file_at_index(ctx.selected_index)
        .map(|f| f.full_path.to_string_lossy().to_string())
        .unwrap_or_default();

    // Pad the string to make the marquee scroll past the end
    let padded_path = format!("{}    ", selected_path_str);

    let path_bar_width = main_chunks[1].width as usize;

    // Marquee animation logic
    let (path_bar_scroll, path_bar_scroll_direction, last_path_bar_update) =
        if padded_path.len() > path_bar_width {
            let now = Instant::now();
            let time_since_update = now.duration_since(ctx.last_path_bar_update);

            let max_scroll = padded_path.len().saturating_sub(path_bar_width) as u16;

            // Pause at the ends of the scroll
            let should_scroll = if ctx.path_bar_scroll == 0 || ctx.path_bar_scroll >= max_scroll {
                time_since_update > marquee_delay
            } else {
                time_since_update > marquee_speed
            };

            if should_scroll {
                let (new_scroll, new_direction) = if ctx.path_bar_scroll_direction == 1 {
                    if ctx.path_bar_scroll < max_scroll {
                        (ctx.path_bar_scroll + 1, ctx.path_bar_scroll_direction)
                    } else {
                        (ctx.path_bar_scroll, -1) // Change direction
                    }
                } else if ctx.path_bar_scroll > 0 {
                    (ctx.path_bar_scroll - 1, ctx.path_bar_scroll_direction)
                } else {
                    (ctx.path_bar_scroll, 1) // Change direction
                };
                (Some(new_scroll), Some(new_direction), Some(now))
            } else {
                (None, None, None)
            }
        } else {
            (None, None, None)
        };

    // Store marquee updates
    if let Some(scroll) = path_bar_scroll {
        updates.path_bar_scroll = Some(scroll);
    }
    if let Some(dir) = path_bar_scroll_direction {
        updates.path_bar_scroll_direction = Some(dir);
    }
    if let Some(time) = last_path_bar_update {
        updates.last_path_bar_update = Some(time);
    }

    // Path bar - use current context value for rendering since updates will be applied later
    let current_path_bar_scroll = path_bar_scroll.unwrap_or(ctx.path_bar_scroll);
    let path_bar = Paragraph::new(padded_path)
        .style(Style::default().fg(Color::DarkGray))
        .scroll((0, current_path_bar_scroll));
    f.render_widget(path_bar, main_chunks[1]);

    // Search input at the bottom
    let cwd_str = ctx.cwd.to_string_lossy();
    let filter_indicator = match ctx.current_filter {
        search_worker::FilterType::None => "",
        search_worker::FilterType::OnlyCwd => " [CWD]",
        search_worker::FilterType::DirectCwd => " [DIRECT]",
        search_worker::FilterType::OnlyDirs => " [DIRS]",
        search_worker::FilterType::OnlyFiles => " [FILES]",
    };
    let search_title = format!("Search: {}{}", cwd_str, filter_indicator);
    let input =
        Paragraph::new(ctx.query).block(Block::default().borders(Borders::ALL).title(search_title));
    f.render_widget(input, main_chunks[2]);

    // Filter picker overlay (rendered on top if visible)
    if ctx.ui_state.filter_picker_visible {
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
            if *filter_type == ctx.current_filter {
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
    // Account for border (1 char) + query length
    let cursor_x = main_chunks[2].x + 1 + ctx.query.len() as u16;
    let cursor_y = main_chunks[2].y + 1; // 1 for top border
    f.set_cursor_position((cursor_x, cursor_y));

    updates
}

/// Get eza command flags based on preview pane width
///
/// # Arguments
/// * `width` - Available width for the preview pane
///
/// # Returns
/// The flags to pass to eza (everything after "eza -al")
fn get_eza_flags(width: u16) -> &'static str {
    // If width is small, omit user and permissions to save space
    if width < 80 {
        " --no-user --no-permissions"
    } else {
        ""
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_get_eza_flags_wide_screen() {
        // Wide screens get full eza output
        assert_eq!(get_eza_flags(80), "", "80 width should use full eza");
        assert_eq!(get_eza_flags(100), "", "100 width should use full eza");
        assert_eq!(get_eza_flags(120), "", "120 width should use full eza");
    }

    #[test]
    fn test_get_eza_flags_narrow_screen() {
        // Narrow screens get compact eza output
        assert_eq!(
            get_eza_flags(79),
            " --no-user --no-permissions",
            "79 width should use compact eza"
        );
        assert_eq!(
            get_eza_flags(60),
            " --no-user --no-permissions",
            "60 width should use compact eza"
        );
        assert_eq!(
            get_eza_flags(40),
            " --no-user --no-permissions",
            "40 width should use compact eza"
        );
    }
}
