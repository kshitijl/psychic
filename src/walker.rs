use crate::search_worker::{WalkerFileMetadata, WalkerMessage};
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use walkdir::WalkDir;

const IGNORED_DIRS: &[&str] = &[".git", "node_modules", ".venv", "target"];
const MAX_FILES: usize = 250_000;
const SHALLOW_MODE_THRESHOLD: usize = 8_000;

pub fn start_file_walker(root: PathBuf, tx: Sender<WalkerMessage>) {
    std::thread::spawn(move || {
        // First pass: try full-depth exploration
        let mut item_count = 0;
        let mut items = Vec::new();

        for entry in WalkDir::new(&root)
            .follow_links(true)
            .into_iter()
            .filter_entry(|e| {
                // Skip ignored directories
                if e.file_type().is_dir()
                    && let Some(name) = e.file_name().to_str()
                {
                    return !IGNORED_DIRS.contains(&name);
                }
                true
            })
            .filter_map(|e| e.ok())
        {
            // Skip root itself
            let is_root = entry.path() == root;
            if is_root {
                continue;
            }

            item_count += 1;

            // If we exceed the shallow mode threshold, switch to depth=1 exploration
            if item_count > SHALLOW_MODE_THRESHOLD {
                // Clear collected items and restart with depth=1
                items.clear();
                item_count = 0;

                for entry in WalkDir::new(&root)
                    .follow_links(true)
                    .max_depth(1)
                    .into_iter()
                    .filter_entry(|e| {
                        if e.file_type().is_dir()
                            && let Some(name) = e.file_name().to_str()
                        {
                            return !IGNORED_DIRS.contains(&name);
                        }
                        true
                    })
                    .filter_map(|e| e.ok())
                {
                    let is_root = entry.path() == root;
                    if !is_root && item_count < MAX_FILES {
                        let is_dir = entry.file_type().is_dir();
                        let metadata = entry.metadata().ok();
                        let mtime = metadata
                            .as_ref()
                            .and_then(|m| m.modified().ok())
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs() as i64);

                        let atime = metadata
                            .as_ref()
                            .and_then(|m| m.accessed().ok())
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs() as i64);

                        let file_size = metadata.as_ref().map(|m| m.len() as i64);

                        let _ = tx.send(WalkerMessage::FileMetadata(WalkerFileMetadata {
                            path: entry.path().to_path_buf(),
                            mtime,
                            atime,
                            file_size,
                            is_dir,
                        }));
                        item_count += 1;
                    }
                }
                // Send AllDone message after shallow mode walk completes
                let _ = tx.send(WalkerMessage::AllDone);
                return; // Exit after shallow mode pass
            }

            // Continue collecting items in full-depth mode
            if item_count < MAX_FILES {
                let is_dir = entry.file_type().is_dir();
                let metadata = entry.metadata().ok();
                let mtime = metadata
                    .as_ref()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64);

                let atime = metadata
                    .as_ref()
                    .and_then(|m| m.accessed().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64);

                let file_size = metadata.as_ref().map(|m| m.len() as i64);

                items.push(WalkerFileMetadata {
                    path: entry.path().to_path_buf(),
                    mtime,
                    atime,
                    file_size,
                    is_dir,
                });
            }
        }

        // If we finished without hitting the threshold, send all collected items
        for item in items {
            let _ = tx.send(WalkerMessage::FileMetadata(item));
        }
        // Send AllDone message after full-depth walk completes
        let _ = tx.send(WalkerMessage::AllDone);
    });
}
