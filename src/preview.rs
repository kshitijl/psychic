use ratatui::text::Text;
use std::path::Path;
use std::time::Instant;

/// Preview cache state - tracks whether we have a light or full preview cached
#[derive(Clone)]
pub enum PreviewCache {
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

impl PreviewCache {
    pub fn get_if_matches(&self, path: &str) -> Option<(Text<'static>, bool)> {
        match self {
            PreviewCache::None => None,
            PreviewCache::Light {
                path: cached_path,
                text,
            } if cached_path == path => Some((text.clone(), false)),
            PreviewCache::Full {
                path: cached_path,
                text,
            } if cached_path == path => Some((text.clone(), true)),
            PreviewCache::Directory { .. } => None, // Use get_dir_if_matches instead
            _ => None,
        }
    }

    pub fn get_dir_if_matches(&self, path: &str, extra_flags: &str) -> Option<Text<'static>> {
        match self {
            PreviewCache::Directory {
                path: cached_path,
                text,
                extra_flags: cached_flags,
            } if cached_path == path && cached_flags == extra_flags => Some(text.clone()),
            _ => None,
        }
    }
}

/// Generate a directory preview using eza (or ls as fallback)
pub fn generate_directory_preview(
    path: &Path,
    extra_flags: &str,
) -> Text<'static> {
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
        Ok(output) => {
            match ansi_to_tui::IntoText::into_text(&output.stdout) {
                Ok(text) => text,
                Err(_) => Text::from("[Unable to parse directory listing]"),
            }
        }
        Err(_) => {
            // Fallback to ls if eza not available
            let ls_output = std::process::Command::new("ls")
                .arg("-lah")
                .arg(path)
                .output();
            match ls_output {
                Ok(output) => Text::from(
                    String::from_utf8_lossy(&output.stdout).to_string(),
                ),
                Err(_) => Text::from("[Unable to list directory]"),
            }
        }
    }
}

/// Generate a full file preview using bat (or plain text as fallback)
pub fn generate_full_file_preview(path: &Path) -> Text<'static> {
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
        Ok(output) => {
            match ansi_to_tui::IntoText::into_text(&output.stdout) {
                Ok(text) => text,
                Err(_) => Text::from("[Unable to parse preview]"),
            }
        }
        Err(_) => match std::fs::read_to_string(path) {
            Ok(content) => Text::from(content),
            Err(_) => Text::from("[Unable to preview file]"),
        },
    }
}

/// Generate a light (truncated) file preview using bat (or plain text as fallback)
pub fn generate_light_file_preview(path: &Path, preview_height: u16) -> Text<'static> {
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
        Ok(output) => {
            match ansi_to_tui::IntoText::into_text(&output.stdout) {
                Ok(text) => text,
                Err(_) => Text::from("[Unable to parse preview]"),
            }
        }
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

/// Get or generate preview text with caching
/// Returns (preview_text, updated_cache)
pub fn get_preview_with_cache(
    cache: &PreviewCache,
    path: &str,
    is_dir: bool,
    preview_height: u16,
    preview_scroll: usize,
    extra_flags: &str,
) -> (Text<'static>, PreviewCache) {
    if is_dir {
        // For directories, check cache with extra_flags match
        if let Some(cached_text) = cache.get_dir_if_matches(path, extra_flags) {
            // FAST PATH: Directory listing is cached with matching flags
            (cached_text, cache.clone())
        } else {
            // SLOW PATH: Generate directory listing
            let text = generate_directory_preview(Path::new(path), extra_flags);
            let new_cache = PreviewCache::Directory {
                path: path.to_string(),
                text: text.clone(),
                extra_flags: extra_flags.to_string(),
            };
            (text, new_cache)
        }
    } else {
        // For files, use bat with three-state caching
        match cache.get_if_matches(path) {
            Some((cached_text, is_full)) => {
                // We have a cached preview
                if is_full || preview_scroll == 0 {
                    // FAST PATH: We have what we need (full, or light + unscrolled)
                    (cached_text, cache.clone())
                } else {
                    // We have light cached but user scrolled - need to upgrade to full
                    let text = generate_full_file_preview(Path::new(path));
                    let new_cache = PreviewCache::Full {
                        path: path.to_string(),
                        text: text.clone(),
                    };
                    (text, new_cache)
                }
            }
            None => {
                // No cache - generate light preview
                let text = generate_light_file_preview(Path::new(path), preview_height);
                let new_cache = PreviewCache::Light {
                    path: path.to_string(),
                    text: text.clone(),
                };
                (text, new_cache)
            }
        }
    }
}
