use crate::search_worker::{WalkerCommand, WalkerFileMetadata, WalkerMessage};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use walkdir::WalkDir;

const IGNORED_DIRS: &[&str] = &[".git", "node_modules", ".venv", "target"];
const MAX_FILES: usize = 250_000;
const SHALLOW_MODE_THRESHOLD: usize = 8_000;
const COMMAND_CHECK_INTERVAL: usize = 100; // Check for commands every N files

pub fn start_file_walker(
    initial_root: PathBuf,
    command_rx: Receiver<WalkerCommand>,
    message_tx: Sender<WalkerMessage>,
) {
    std::thread::spawn(move || {
        let mut current_root = initial_root;

        'outer: loop {
            walk_directory(&current_root, &command_rx, &message_tx);

            // After walk completes, send AllDone
            let _ = message_tx.send(WalkerMessage::AllDone);

            // Wait for next command (blocking)
            match command_rx.recv() {
                Ok(WalkerCommand::ChangeCwd(new_root)) => {
                    log::info!("Walker: Changing directory to {:?}", new_root);
                    current_root = new_root;
                    // Continue to next iteration of outer loop
                }
                Err(_) => {
                    // Channel closed, exit thread
                    log::info!("Walker: Command channel closed, exiting");
                    break 'outer;
                }
            }
        }
    });
}

fn walk_directory(
    root: &PathBuf,
    command_rx: &Receiver<WalkerCommand>,
    tx: &Sender<WalkerMessage>,
) {
    // First pass: try full-depth exploration
    let mut item_count = 0;
    let mut items = Vec::new();

    for entry in WalkDir::new(root)
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

        // Check for commands periodically
        if item_count % COMMAND_CHECK_INTERVAL == 0
            && let Ok(WalkerCommand::ChangeCwd(_)) = command_rx.try_recv()
        {
            log::info!("Walker: Received ChangeCwd during walk, aborting current walk");
            return; // Abort current walk, outer loop will handle the command
        }

        // If we exceed the shallow mode threshold, switch to depth=1 exploration
        if item_count > SHALLOW_MODE_THRESHOLD {
            // Clear collected items and restart with depth=1
            items.clear();
            item_count = 0;

            for entry in WalkDir::new(root)
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
                let is_root = entry.path() == root.as_path();
                if !is_root && item_count < MAX_FILES {
                    // Check for commands periodically in shallow mode too
                    if item_count % COMMAND_CHECK_INTERVAL == 0
                        && let Ok(WalkerCommand::ChangeCwd(_)) = command_rx.try_recv()
                    {
                        log::info!("Walker: Received ChangeCwd during shallow walk, aborting");
                        return;
                    }
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
            return; // Exit after shallow mode pass (AllDone sent by outer loop)
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
    // AllDone will be sent by outer loop
}
