use crate::metadata_ext::MetadataExt;
use crate::search_worker::{WalkerCommand, WalkerFileMetadata, WalkerMessage};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use walkdir::WalkDir;

const IGNORED_DIRS: &[&str] = &[".git", "node_modules", ".venv", "target"];

/// Whether the walker should descend into `entry`.
///
/// Two reasons not to: it is one of the always-noisy directories, or it is a
/// directory the user has hidden. Skipping hidden ones here is what makes
/// hiding *cheaper* than showing - the tree is never walked at all, rather than
/// walked and filtered afterwards.
///
/// `hidden` holds only the prefixes that apply to the current root (the worker
/// drops any that contain it), so starting psychic inside a hidden directory
/// walks it normally.
fn should_descend(is_dir: bool, path: &Path, hidden: &[PathBuf]) -> bool {
    if !is_dir {
        return true;
    }

    if let Some(name) = path.file_name().and_then(|n| n.to_str())
        && IGNORED_DIRS.contains(&name)
    {
        return false;
    }

    // Whole-component comparison, so hiding /a/b does not skip /a/bcd.
    !hidden.iter().any(|prefix| path.starts_with(prefix))
}
const MAX_FILES: usize = 250_000;
const SHALLOW_MODE_THRESHOLD: usize = 8_000;
const COMMAND_CHECK_INTERVAL: usize = 100; // Check for commands every N files

pub fn start_file_walker(
    initial_root: PathBuf,
    initial_hidden: Vec<PathBuf>,
    command_rx: Receiver<WalkerCommand>,
    message_tx: Sender<WalkerMessage>,
) {
    std::thread::spawn(move || {
        let mut current_root = initial_root;
        let mut current_hidden = initial_hidden;

        'outer: loop {
            walk_directory(&current_root, &current_hidden, &command_rx, &message_tx);

            // After walk completes, send AllDone
            let _ = message_tx.send(WalkerMessage::AllDone);

            // Wait for next command (blocking)
            match command_rx.recv() {
                Ok(WalkerCommand::ChangeCwd { path, hidden }) => {
                    log::info!("Walker: Changing directory to {:?}", path);
                    current_root = path;
                    // Which hidden directories apply depends on the root, so
                    // the worker recomputes them and sends them along with it.
                    current_hidden = hidden;
                    // Continue to next iteration of outer loop
                }
                Err(_) => {
                    // Channel closed, exit thread (don't log - this is expected during shutdown)
                    break 'outer;
                }
            }
        }
    });
}

fn walk_directory(
    root: &PathBuf,
    hidden: &[PathBuf],
    command_rx: &Receiver<WalkerCommand>,
    tx: &Sender<WalkerMessage>,
) {
    // First pass: try full-depth exploration
    let mut item_count = 0;
    let mut items = Vec::new();

    for entry in WalkDir::new(root)
        .follow_links(true)
        .into_iter()
        .filter_entry(|e| should_descend(e.file_type().is_dir(), e.path(), hidden))
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
            && let Ok(WalkerCommand::ChangeCwd { .. }) = command_rx.try_recv()
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
                .filter_entry(|e| should_descend(e.file_type().is_dir(), e.path(), hidden))
                .filter_map(|e| e.ok())
            {
                let is_root = entry.path() == root.as_path();
                if !is_root && item_count < MAX_FILES {
                    // Check for commands periodically in shallow mode too
                    if item_count % COMMAND_CHECK_INTERVAL == 0
                        && let Ok(WalkerCommand::ChangeCwd { .. }) = command_rx.try_recv()
                    {
                        log::info!("Walker: Received ChangeCwd during shallow walk, aborting");
                        return;
                    }
                    let is_dir = entry.file_type().is_dir();
                    let metadata = entry.metadata().ok();
                    let mtime = metadata.as_ref().and_then(|m| m.mtime_as_secs());
                    let atime = metadata.as_ref().and_then(|m| m.atime_as_secs());
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
            let mtime = metadata.as_ref().and_then(|m| m.mtime_as_secs());
            let atime = metadata.as_ref().and_then(|m| m.atime_as_secs());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_files_are_always_walked() {
        assert!(
            should_descend(false, Path::new("/a/target"), &[]),
            "A file named like an ignored directory is still a file"
        );
        assert!(
            should_descend(
                false,
                Path::new("/a/hidden/notes.txt"),
                &[PathBuf::from("/a/hidden")]
            ),
            "The filter only decides whether to descend, so files pass it"
        );
    }

    #[test]
    fn test_noisy_directories_are_skipped() {
        for name in [".git", "node_modules", ".venv", "target"] {
            assert!(
                !should_descend(true, &Path::new("/a").join(name), &[]),
                "{} should never be walked",
                name
            );
        }
        assert!(
            should_descend(true, Path::new("/a/src"), &[]),
            "An ordinary directory is walked"
        );
    }

    #[test]
    fn test_hidden_directories_are_skipped() {
        let hidden = vec![PathBuf::from("/a/old")];

        assert!(
            !should_descend(true, Path::new("/a/old"), &hidden),
            "The hidden directory itself is not descended into"
        );
        assert!(
            !should_descend(true, Path::new("/a/old/deeper"), &hidden),
            "Nor is anything under it"
        );
        assert!(
            should_descend(true, Path::new("/a/older"), &hidden),
            "A sibling sharing a name prefix is still walked"
        );
        assert!(
            should_descend(true, Path::new("/a/current"), &hidden),
            "An unrelated directory is still walked"
        );
    }

    #[test]
    fn test_starting_inside_a_hidden_directory_walks_it() {
        // The worker drops prefixes containing the root before sending them, so
        // by the time they reach the walker there is nothing to skip.
        let hidden: Vec<PathBuf> = Vec::new();

        assert!(
            should_descend(true, Path::new("/a/old/deeper"), &hidden),
            "Started inside the hidden directory, so it walks normally"
        );
    }

    /// Walk a real tree and report every path the walker emitted.
    fn walked_paths(root: &Path, hidden: &[PathBuf]) -> Vec<PathBuf> {
        let (_command_tx, command_rx) = std::sync::mpsc::channel::<WalkerCommand>();
        let (tx, rx) = std::sync::mpsc::channel::<WalkerMessage>();

        walk_directory(&root.to_path_buf(), hidden, &command_rx, &tx);
        drop(tx);

        let mut paths: Vec<PathBuf> = rx
            .into_iter()
            .filter_map(|m| match m {
                WalkerMessage::FileMetadata(f) => Some(f.path),
                WalkerMessage::AllDone => None,
            })
            .collect();
        paths.sort();
        paths
    }

    /// `a/b/c/d/buried.txt`, plus a file beside each level, in a temp dir.
    struct TempTree {
        root: PathBuf,
    }

    impl TempTree {
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir()
                .join(format!("psychic-walk-{}-{}", name, std::process::id()))
                .join("a");
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(root.join("b/c/d")).expect("create tree");
            std::fs::create_dir_all(root.join("b/other")).expect("create sibling");

            for path in [
                "top.txt",
                "b/mid.txt",
                "b/c/near.txt",
                "b/c/d/buried.txt",
                "b/other/kept.txt",
            ] {
                std::fs::write(root.join(path), b"x").expect("create file");
            }

            // Canonical, because that is the form hidden prefixes are stored
            // in and the walk root arrives in.
            let root = root.canonicalize().expect("canonicalize tree root");
            Self { root }
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            if let Some(parent) = self.root.parent() {
                let _ = std::fs::remove_dir_all(parent);
            }
        }
    }

    #[test]
    fn test_nested_hidden_directory_is_not_walked() {
        // Start in `a` with `a/b/c/d` hidden: the intermediate directories must
        // still be walked, because everything else is reached through them, but
        // the hidden directory and its contents must not be.
        let tree = TempTree::new("nested");
        let hidden = vec![tree.root.join("b/c/d")];

        let walked = walked_paths(&tree.root, &hidden);

        assert!(
            !walked.contains(&tree.root.join("b/c/d")),
            "The hidden directory itself was walked: {:?}",
            walked
        );
        assert!(
            !walked.contains(&tree.root.join("b/c/d/buried.txt")),
            "A file inside the hidden directory was walked: {:?}",
            walked
        );
        assert!(
            walked.contains(&tree.root.join("b/c/near.txt")),
            "A file beside the hidden directory must still be found: {:?}",
            walked
        );
        assert!(
            walked.contains(&tree.root.join("b/other/kept.txt")),
            "An unrelated subtree must still be walked: {:?}",
            walked
        );
        assert!(
            walked.contains(&tree.root.join("b/c")),
            "The parent of the hidden directory must still be walked: {:?}",
            walked
        );
    }

    #[test]
    fn test_nothing_is_skipped_without_hidden_directories() {
        let tree = TempTree::new("nothing-hidden");

        let walked = walked_paths(&tree.root, &[]);

        assert!(
            walked.contains(&tree.root.join("b/c/d/buried.txt")),
            "With nothing hidden the whole tree is walked: {:?}",
            walked
        );
    }
}
