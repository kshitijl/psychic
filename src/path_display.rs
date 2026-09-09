use std::borrow::Cow;
use std::path::Path;
use std::time::Duration;
use unicode_width::UnicodeWidthStr;

/// How many terminal columns `text` occupies.
///
/// Not its length in bytes, which is what a non-ASCII path measures far more of
/// than it draws, and not its length in characters either: a CJK glyph or an
/// emoji is two columns wide. Ratatui lays out in columns, so anything that is
/// deciding what fits has to count the same way.
pub fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

/// Shorten `text` to at most `width` columns, marking the cut with an ellipsis.
///
/// Returns whole characters. Slicing a `String` by a byte offset panics the
/// moment the offset lands inside a multi-byte character, which for a log line
/// or a path is a matter of when, not whether.
pub fn truncate_to_width(text: &str, width: usize) -> Cow<'_, str> {
    if display_width(text) <= width {
        return Cow::Borrowed(text);
    }
    if width <= 1 {
        return Cow::Owned("…".repeat(width));
    }

    // One column for the ellipsis.
    let budget = width - 1;
    let mut out = String::with_capacity(text.len());
    let mut used = 0;

    for c in text.chars() {
        let w = display_width(c.encode_utf8(&mut [0u8; 4]));
        if used + w > budget {
            break;
        }
        out.push(c);
        used += w;
    }
    out.push('…');

    Cow::Owned(out)
}

/// What is safe to put in a terminal cell.
///
/// Control characters are not just ugly, they are *instructions*: an ESC in a
/// file, or in a filename, starts an escape sequence that the terminal obeys,
/// and a carriage return or backspace moves the cursor out from under whatever
/// we thought we were drawing. Ratatui passes a cell's contents straight
/// through, so anything that reaches a cell reaches the terminal.
///
/// Tabs become four spaces, since a literal tab moves the cursor by an amount
/// the layout has not accounted for. Everything else in the control categories
/// becomes a dot. Borrows when there is nothing to change, which is almost
/// always.
pub fn printable(text: &str) -> Cow<'_, str> {
    if !text.chars().any(|c| c.is_control()) {
        return Cow::Borrowed(text);
    }

    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\t' => out.push_str("    "),
            c if c.is_control() => out.push('·'),
            c => out.push(c),
        }
    }
    Cow::Owned(out)
}

/// Human-readable byte count, at most one decimal place.
///
/// Shared by the debug pane and the directory listing so that a size means the
/// same thing wherever it appears.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 3] = [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)];

    for (unit, scale) in UNITS {
        if bytes >= scale {
            return format!("{:.1} {}", bytes as f64 / scale as f64, unit);
        }
    }
    format!("{} B", bytes)
}

/// Truncates a path string in the middle if it's too long, keeping the first
/// component and the end of the path.
/// e.g., "a/b/c/d/e.txt" -> "a/.../d/e.txt"
pub fn truncate_path(path_str: &str, max_len: usize) -> String {
    if display_width(path_str) <= max_len {
        return path_str.to_string();
    }

    let path = Path::new(path_str);
    let components: Vec<&str> = path
        .components()
        .map(|c| c.as_os_str().to_str().unwrap_or(""))
        .collect();

    // Don't truncate simple paths
    if components.len() <= 2 {
        return path_str.to_string();
    }

    let head = components.first().unwrap_or(&"");
    let mut tail_parts: Vec<&str> = Vec::new();

    // Start with filename
    let filename = components.last().unwrap_or(&"");
    tail_parts.push(filename);

    // head + "/.../" + filename
    let mut len_so_far = head.len() + 5 + filename.len();

    // Add parts to tail from the end until we run out of space
    // Iterate over parent components in reverse (skipping filename)
    for part in components.iter().rev().skip(1) {
        // Stop if we are about to collide with the head component
        if part == head {
            break;
        }

        if len_so_far + part.len() + 1 > max_len {
            break;
        }

        tail_parts.insert(0, part);
        len_so_far += part.len() + 1;
    }

    format!("{}/.../{}", head, tail_parts.join("/"))
}

/// Abbreviate a path component to its first character
/// e.g., "Users" -> "U", "kshitijlauria" -> "k"
fn abbreviate_component(s: &str) -> String {
    s.chars()
        .next()
        .map(|c| c.to_string())
        .unwrap_or_else(|| s.to_string())
}

/// Truncate an absolute path (for historical files) showing beginning and end with abbreviations
/// e.g., "/Users/kshitijlauria/Library/CloudStorage/Dropbox/src/11-sg/todo.md"
///    -> "/U/k/L/CloudStorage/.../11-sg/todo.md"
pub fn truncate_absolute_path(path_str: &str, max_len: usize) -> String {
    if display_width(path_str) <= max_len {
        return path_str.to_string();
    }

    // Split path into components manually to avoid Path component issues
    let parts: Vec<&str> = path_str.split('/').filter(|s| !s.is_empty()).collect();

    if parts.is_empty() {
        return path_str.to_string();
    }

    // Always keep the filename
    let filename = parts.last().unwrap_or(&"");

    // Start with just the filename
    let mut tail_count = 1; // Start with 1 for filename
    let mut estimated_len = 3 + 1 + filename.len(); // "..." + "/" + filename

    // Add components from the end (before filename)
    for i in (0..parts.len().saturating_sub(1)).rev() {
        let part = parts[i];
        if estimated_len + 1 + part.len() > max_len {
            break;
        }
        estimated_len += 1 + part.len(); // "/" + part
        tail_count += 1;
    }

    // Add abbreviated components from the beginning
    let mut head_parts: Vec<String> = Vec::new();
    for i in 0..parts.len() {
        if i >= parts.len() - tail_count {
            // We've reached the tail section
            break;
        }
        let part = parts[i];

        // Try full component first
        if estimated_len + 1 + part.len() <= max_len {
            estimated_len += 1 + part.len();
            head_parts.push(part.to_string());
        } else {
            // Try abbreviated component
            let abbrev = abbreviate_component(part);
            if estimated_len + 1 + abbrev.len() <= max_len {
                estimated_len += 1 + abbrev.len();
                head_parts.push(abbrev);
            } else {
                // Can't fit even abbreviated, stop adding head parts
                break;
            }
        }
    }

    // Build the result
    let is_absolute = path_str.starts_with('/');
    let head_count = head_parts.len();

    if head_count + tail_count >= parts.len() {
        // Everything fits (possibly with abbreviations)
        if head_parts.iter().all(|h| parts.contains(&h.as_str())) {
            // No abbreviations were used
            path_str.to_string()
        } else {
            // Some abbreviations, reconstruct
            let tail: Vec<&str> = parts
                .iter()
                .skip(parts.len() - tail_count)
                .copied()
                .collect();
            if is_absolute {
                format!("/{}/{}", head_parts.join("/"), tail.join("/"))
            } else {
                format!("{}/{}", head_parts.join("/"), tail.join("/"))
            }
        }
    } else if head_count == 0 {
        // Only tail fits
        let tail: Vec<&str> = parts
            .iter()
            .skip(parts.len() - tail_count)
            .copied()
            .collect();
        if is_absolute {
            format!("/.../{}", tail.join("/"))
        } else {
            format!(".../{}", tail.join("/"))
        }
    } else {
        // Both head and tail, with ellipsis
        let tail: Vec<&str> = parts
            .iter()
            .skip(parts.len() - tail_count)
            .copied()
            .collect();
        if is_absolute {
            format!("/{}/.../{}", head_parts.join("/"), tail.join("/"))
        } else {
            format!("{}/.../{}", head_parts.join("/"), tail.join("/"))
        }
    }
}

/// Convert Unix timestamp to human-readable "time ago" string
pub fn get_time_ago(mtime: Option<i64>) -> String {
    if let Some(mtime_secs) = mtime {
        // Convert Unix timestamp to SystemTime
        let mtime_systime = std::time::UNIX_EPOCH + Duration::from_secs(mtime_secs as u64);

        let duration = std::time::SystemTime::now()
            .duration_since(mtime_systime)
            .unwrap_or(Duration::from_secs(0));

        let formatter = timeago::Formatter::new();
        return formatter.convert(duration);
    }
    String::from("unknown")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_path_simple() {
        let result = truncate_path("a/b/c/d/e.txt", 12);
        assert_eq!(
            result, "a/.../e.txt",
            "Should truncate middle components when too long"
        );
    }

    #[test]
    fn test_truncate_path_no_truncation_needed() {
        let result = truncate_path("a/b.txt", 20);
        assert_eq!(result, "a/b.txt", "Should not truncate short paths");
    }

    #[test]
    fn test_truncate_path_keeps_filename() {
        let result = truncate_path("foo/bar/baz/qux/file.rs", 20);
        assert!(result.contains("file.rs"), "Should always keep filename");
        assert!(result.contains("..."), "Should use ellipsis for truncation");
    }

    #[test]
    fn test_truncate_path_builds_from_end() {
        let result = truncate_path("a/b/c/d/e/f/g.txt", 15);
        // Actual: "a/.../e/f/g.txt" = 15 chars
        assert_eq!(
            result, "a/.../e/f/g.txt",
            "Should build from end to fit more context"
        );
    }

    #[test]
    fn test_truncate_absolute_path_no_truncation() {
        let result = truncate_absolute_path("/short/path.txt", 50);
        assert_eq!(result, "/short/path.txt", "Should not truncate short paths");
    }

    #[test]
    fn test_truncate_absolute_path_preserves_leading_slash() {
        let result = truncate_absolute_path("/a/b/c/d/e/f/g.txt", 20);
        assert!(result.starts_with('/'), "Should preserve leading slash");
    }

    #[test]
    fn test_truncate_absolute_path_relative() {
        let result = truncate_absolute_path("a/b/c/d/e.txt", 15);
        assert!(
            !result.starts_with('/'),
            "Should not add slash to relative paths"
        );
    }

    #[test]
    fn test_truncate_absolute_path_fits_more_tail() {
        let result = truncate_absolute_path("/a/b/c/d/e/file.txt", 25);
        // Should abbreviate head and keep tail
        assert!(result.contains("file.txt"), "Should keep filename");
        assert!(result.starts_with('/'), "Should have leading slash");
    }

    #[test]
    fn test_truncate_absolute_path_only_tail() {
        let result = truncate_absolute_path("/very/long/path/components/file.txt", 15);
        // With very limited space, should show only tail
        assert!(result.contains("file.txt"), "Should keep filename");
        assert!(result.contains("..."), "Should use ellipsis");
    }

    #[test]
    fn test_truncate_absolute_path_both_ends() {
        let result = truncate_absolute_path("/Users/kshitijlauria/src/project/file.rs", 30);
        // Should have abbreviated head, ellipsis, and tail
        assert!(result.contains("file.rs"), "Should keep filename");
        assert!(result.contains("..."), "Should use ellipsis for middle");
        assert!(result.len() <= 30, "Should respect max length");
    }

    #[test]
    fn test_truncate_absolute_path_exact_example() {
        // Test the exact example from the docstring
        let path = "/Users/kshitijlauria/Library/CloudStorage/Dropbox/src/11-sg/todo.md";
        let result = truncate_absolute_path(path, 40);
        // Should abbreviate early components and keep end
        assert!(result.contains("todo.md"), "Should keep filename");
        assert!(result.contains("..."), "Should use ellipsis");
        assert!(result.starts_with('/'), "Should start with slash");
        assert!(result.len() <= 40, "Should respect max length");
    }
}

#[cfg(test)]
mod byte_tests {
    use super::human_bytes;

    #[test]
    fn test_human_bytes() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.0 KB");
        assert_eq!(human_bytes(62_914_560), "60.0 MB");
        assert_eq!(human_bytes(3 << 30), "3.0 GB");
    }
}

#[cfg(test)]
mod printable_tests {
    use super::printable;

    #[test]
    fn test_ordinary_text_is_left_alone_and_not_copied() {
        let text = printable("src/main.rs");
        assert_eq!(text, "src/main.rs");
        assert!(
            matches!(text, std::borrow::Cow::Borrowed(_)),
            "no allocation"
        );
    }

    #[test]
    fn test_escape_sequences_cannot_reach_the_terminal() {
        assert_eq!(
            printable("red \x1b[31mnot red"),
            "red ·[31mnot red",
            "The ESC is the whole problem; the rest is only text"
        );
    }

    #[test]
    fn test_cursor_moving_characters_are_defanged() {
        assert_eq!(printable("a\rb"), "a·b", "carriage return");
        assert_eq!(printable("a\x08b"), "a·b", "backspace");
        assert_eq!(printable("a\x07b"), "a·b", "bell");
        assert_eq!(printable("a\x7fb"), "a·b", "delete");
    }

    #[test]
    fn test_tabs_become_spaces_the_layout_can_count() {
        assert_eq!(printable("a\tb"), "a    b");
    }
}

#[cfg(test)]
mod width_tests {
    use super::{display_width, truncate_to_width};

    #[test]
    fn test_width_counts_columns_not_bytes_or_characters() {
        assert_eq!(display_width("abc"), 3);
        assert_eq!(display_width("héllo"), 5, "5 columns, 6 bytes");
        assert_eq!(display_width("日本語"), 6, "3 characters, 6 columns");
    }

    #[test]
    fn test_short_text_is_left_alone() {
        assert_eq!(truncate_to_width("abc", 10), "abc");
        assert_eq!(truncate_to_width("abc", 3), "abc");
    }

    #[test]
    fn test_truncation_never_splits_a_character() {
        // The byte at offset 5 is in the middle of the é. Slicing there is a
        // panic, which is what took the UI down with the debug pane open.
        let text = "caf\u{e9} au lait";
        let cut = truncate_to_width(text, 5);

        assert_eq!(cut, "café…");
        assert!(display_width(&cut) <= 5);
    }

    #[test]
    fn test_truncation_respects_wide_characters() {
        let cut = truncate_to_width("日本語テスト", 5);

        assert!(
            display_width(&cut) <= 5,
            "{:?} is {} columns",
            cut,
            display_width(&cut)
        );
        assert_eq!(cut, "日本…");
    }

    #[test]
    fn test_absurdly_narrow_widths_do_not_panic() {
        for width in 0..3 {
            let cut = truncate_to_width("日本語", width);
            assert!(display_width(&cut) <= width.max(1));
        }
    }
}
