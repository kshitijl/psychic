use std::path::Path;
use std::time::Duration;

/// Truncates a path string in the middle if it's too long, keeping the first
/// component and the end of the path.
/// e.g., "a/b/c/d/e.txt" -> "a/.../d/e.txt"
pub fn truncate_path(path_str: &str, max_len: usize) -> String {
    if path_str.len() <= max_len {
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
    if path_str.len() <= max_len {
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
            return path_str.to_string();
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
        assert_eq!(result, "a/.../e.txt", "Should truncate middle components when too long");
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
        assert!(!result.starts_with('/'), "Should not add slash to relative paths");
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
