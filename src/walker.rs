use crate::search_worker::WalkerFileMetadata;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use walkdir::WalkDir;

const IGNORED_DIRS: &[&str] = &[".git", "node_modules", ".venv", "target"];
const MAX_FILES: usize = 250_000;

pub fn start_file_walker(root: PathBuf, tx: Sender<WalkerFileMetadata>) {
    std::thread::spawn(move || {
        let mut item_count = 0;
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
            // Send both files AND directories (but not the root itself)
            let is_root = entry.path() == root;
            if !is_root {
                if item_count >= MAX_FILES {
                    break;
                }

                let is_dir = entry.file_type().is_dir();

                // Extract metadata from cached walkdir metadata
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

                let _ = tx.send(WalkerFileMetadata {
                    path: entry.path().to_path_buf(),
                    mtime,
                    atime,
                    file_size,
                    is_dir,
                });
                item_count += 1;
            }
        }
    });
}
