use ratatui::text::Text;
use std::path::Path;
use std::time::Instant;

/// Preview cache state - tracks whether we have a light or full preview cached
#[derive(Clone)]
enum CachedPreview {
    /// No preview cached
    None,
    /// Light preview cached (first N lines only) for a file
    Light { path: String, text: Text<'static> },
    /// Full preview cached (entire file) for a file
    Full { path: String, text: Text<'static> },
    /// Directory preview cached (eza output)
    Directory {
        path: String,
        text: Text<'static>,
        extra_flags: String,
    },
}

/// Manages preview generation and caching, along with scroll state
#[derive(Clone)]
pub struct PreviewManager {
    cache: CachedPreview,
    scroll: usize,
}

impl PreviewManager {
    pub fn new() -> Self {
        Self {
            cache: CachedPreview::None,
            scroll: 0,
        }
    }

    /// Render the preview for the given path
    /// Returns the preview text, handling caching automatically
    pub fn render(&mut self, path: &Path, is_dir: bool, width: u16, height: u16) -> Text<'static> {
        let path_str = path.to_string_lossy().to_string();
        let extra_flags = Self::get_eza_flags(width);

        if is_dir {
            // For directories, check cache with extra_flags match
            if let CachedPreview::Directory {
                path: cached_path,
                text,
                extra_flags: cached_flags,
            } = &self.cache
            {
                if cached_path == &path_str && cached_flags == &extra_flags {
                    // FAST PATH: Directory listing is cached with matching flags
                    return text.clone();
                }
            }

            // SLOW PATH: Generate directory listing
            let text = Self::generate_directory_preview(path, &extra_flags);
            self.cache = CachedPreview::Directory {
                path: path_str,
                text: text.clone(),
                extra_flags,
            };
            text
        } else {
            // For files, use bat with three-state caching
            match &self.cache {
                CachedPreview::Light {
                    path: cached_path,
                    text,
                } if cached_path == &path_str => {
                    // We have light cached
                    if self.scroll == 0 {
                        // FAST PATH: Light cached and unscrolled
                        return text.clone();
                    } else {
                        // User scrolled - need to upgrade to full
                        let full_text = Self::generate_full_file_preview(path);
                        self.cache = CachedPreview::Full {
                            path: path_str,
                            text: full_text.clone(),
                        };
                        return full_text;
                    }
                }
                CachedPreview::Full {
                    path: cached_path,
                    text,
                } if cached_path == &path_str => {
                    // FAST PATH: Full preview cached
                    return text.clone();
                }
                _ => {
                    // No cache - generate light preview
                    let text = Self::generate_light_file_preview(path, height);
                    self.cache = CachedPreview::Light {
                        path: path_str,
                        text: text.clone(),
                    };
                    text
                }
            }
        }
    }

    /// Scroll the preview by the given delta
    pub fn scroll(&mut self, delta: isize) {
        let new_scroll = (self.scroll as isize + delta).max(0);
        self.scroll = new_scroll as usize;
    }

    /// Get current scroll position
    pub fn scroll_position(&self) -> usize {
        self.scroll
    }

    /// Clear the cache (called when selection changes)
    pub fn clear(&mut self) {
        self.cache = CachedPreview::None;
        self.scroll = 0;
    }

    /// Get cache status for debugging
    pub fn status(&self, path: &str) -> &str {
        match &self.cache {
            CachedPreview::None => "Not cached",
            CachedPreview::Light { path: cached_path, .. } if cached_path == path => "Cached (light)",
            CachedPreview::Full { path: cached_path, .. } if cached_path == path => "Cached (full)",
            CachedPreview::Directory { path: cached_path, extra_flags, .. } if cached_path == path => {
                if extra_flags.is_empty() {
                    "Cached (dir)"
                } else {
                    "Cached (dir+flags)"
                }
            }
            _ => "Not cached",
        }
    }

    /// Get extra eza flags based on terminal width
    fn get_eza_flags(width: u16) -> String {
        if width < 100 {
            "--no-permissions --no-user --no-time".to_string()
        } else {
            String::new()
        }
    }

    /// Generate a directory preview using eza (or ls as fallback)
    fn generate_directory_preview(path: &Path, extra_flags: &str) -> Text<'static> {
        let mut eza_cmd = std::process::Command::new("eza");
        eza_cmd.arg("-al").arg("--color=always");

        // Add extra flags if preview is narrow
        for flag in extra_flags.split_whitespace() {
            if !flag.is_empty() {
                eza_cmd.arg(flag);
            }
        }

        eza_cmd.arg(path);
        let eza_start = Instant::now();
        let eza_output = eza_cmd.output();
        log::info!(
            "TIMING {{\"op\":\"eza_dir_preview\",\"ms\":{}}}",
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
                    .arg(path)
                    .output();
                match ls_output {
                    Ok(output) => {
                        Text::from(String::from_utf8_lossy(&output.stdout).to_string())
                    }
                    Err(_) => Text::from("[Unable to list directory]"),
                }
            }
        }
    }

    /// Generate a full file preview using bat (or plain text as fallback)
    fn generate_full_file_preview(path: &Path) -> Text<'static> {
        let bat_start = Instant::now();
        let bat_output = std::process::Command::new("bat")
            .arg("--color=always")
            .arg("--style=numbers")
            .arg("--paging=never")
            .arg(path)
            .output();
        log::info!(
            "TIMING {{\"op\":\"bat_full_preview\",\"ms\":{}}}",
            bat_start.elapsed().as_secs_f64() * 1000.0
        );

        match bat_output {
            Ok(output) => match ansi_to_tui::IntoText::into_text(&output.stdout) {
                Ok(text) => text,
                Err(_) => Text::from("[Unable to parse preview]"),
            },
            Err(_) => match std::fs::read_to_string(path) {
                Ok(content) => Text::from(content),
                Err(_) => Text::from("[Unable to preview file]"),
            },
        }
    }

    /// Generate a light (truncated) file preview using bat (or plain text as fallback)
    fn generate_light_file_preview(path: &Path, preview_height: u16) -> Text<'static> {
        let line_range = format!(":{}", preview_height);
        let bat_start = Instant::now();
        let bat_output = std::process::Command::new("bat")
            .arg("--color=always")
            .arg("--style=numbers")
            .arg("--line-range")
            .arg(&line_range)
            .arg(path)
            .output();
        log::info!(
            "TIMING {{\"op\":\"bat_light_preview\",\"ms\":{}}}",
            bat_start.elapsed().as_secs_f64() * 1000.0
        );

        match bat_output {
            Ok(output) => match ansi_to_tui::IntoText::into_text(&output.stdout) {
                Ok(text) => text,
                Err(_) => Text::from("[Unable to parse preview]"),
            },
            Err(_) => {
                // Fallback for light preview
                match std::fs::read_to_string(path) {
                    Ok(content) => Text::from(
                        content
                            .lines()
                            .take(preview_height as usize)
                            .collect::<Vec<_>>()
                            .join("\n"),
                    ),
                    Err(_) => Text::from("[Unable to preview file]"),
                }
            }
        }
    }
}
