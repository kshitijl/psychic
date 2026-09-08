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
/// Most direct children of the root we will report.
///
/// A guard against a pathological directory, not a normal limit.
const MAX_FILES: usize = 250_000;

/// How much may live *below* the root's children before the tree is declared
/// too big to index and only those children are shown.
///
/// Two things depend on it. It bounds the **walk**: reaching it stops the
/// descent, so starting in `/` or a system directory does not stat a few hundred
/// thousand files in the background for nobody - the first pass has already
/// shown that directory's children. And it bounds every **keystroke**
/// afterwards, because each entry reported becomes a registry entry that is
/// filtered and ranked on every keypress. `~` is the everyday case; `/` is the
/// one that would otherwise never stop.
const SHALLOW_MODE_THRESHOLD: usize = 8_000;

/// How often, in entries, to look for a command telling us to go elsewhere.
const COMMAND_CHECK_INTERVAL: usize = 100;

pub fn start_file_walker(
    initial_root: PathBuf,
    initial_hidden: Vec<PathBuf>,
    command_rx: Receiver<WalkerCommand>,
    message_tx: Sender<WalkerMessage>,
) {
    std::thread::spawn(move || {
        let mut root = initial_root;
        let mut hidden = initial_hidden;

        loop {
            let interrupted_by = walk_directory(
                &root,
                &hidden,
                SHALLOW_MODE_THRESHOLD,
                &command_rx,
                &message_tx,
            );

            // A walk that was interrupted carries the command that interrupted
            // it. Taking it back here is the point: the interrupt used to
            // `try_recv` the command and drop it on the floor, after which the
            // blocking `recv` below waited for a command that had already been
            // delivered - so navigating during a long walk meant the directory
            // you navigated to was never walked at all.
            let next = match interrupted_by {
                Some(command) => Some(command),
                None => {
                    // Only a walk that ran to the end gets to say so.
                    let _ = message_tx.send(WalkerMessage::AllDone);
                    command_rx.recv().ok()
                }
            };

            match next {
                Some(WalkerCommand::ChangeCwd { path, hidden: now }) => {
                    log::info!("Walker: Changing directory to {:?}", path);
                    root = path;
                    // Which hidden directories apply depends on the root, so
                    // the worker recomputes them and sends them along with it.
                    hidden = now;
                }
                // Channel closed: shutting down. Not logged, it is expected.
                None => break,
            }
        }
    });
}

/// Walk `root` and report what is in it.
///
/// Two passes, because the two halves of a tree are worth very different
/// amounts. The root's own children are what the user is looking at and cost
/// one `readdir`, so they go out immediately. Everything below them is the part
/// that can be enormous, so it is held back until it either finishes or grows
/// past `max_below`, at which point the tree is declared too big to index and
/// the children stand alone.
///
/// This used to be one full-depth walk that buffered everything, and on passing
/// the threshold threw all of it away and started again at depth 1 - so the
/// common case of starting in `~` paid for two walks and showed nothing until
/// both were done. Now the expensive pass is the one that can be abandoned, and
/// abandoning it costs nothing already shown.
///
/// Returns the command that interrupted the walk, if one did.
fn walk_directory(
    root: &Path,
    hidden: &[PathBuf],
    max_below: usize,
    command_rx: &Receiver<WalkerCommand>,
    tx: &Sender<WalkerMessage>,
) -> Option<WalkerCommand> {
    let mut sent = 0;
    for entry in entries(root, hidden, 1..=1) {
        if sent >= MAX_FILES {
            log::warn!("Walker: {:?} has more than {} children", root, MAX_FILES);
            break;
        }
        if tx
            .send(WalkerMessage::FileMetadata(describe(&entry)))
            .is_err()
        {
            return None;
        }
        sent += 1;

        if let Some(command) = command_if_due(sent, command_rx) {
            return Some(command);
        }
    }

    // The worker publishes these without waiting for its debounce: they are
    // the whole answer for a directory with nothing much under it, and the
    // first useful answer for one with a lot.
    if tx.send(WalkerMessage::ChildrenDone).is_err() {
        return None;
    }

    let mut below = Vec::new();
    for entry in entries(root, hidden, 2..=usize::MAX) {
        if below.len() >= max_below {
            log::info!(
                "Walker: more than {} entries below the children of {:?}; \
                 showing only its children",
                max_below,
                root
            );
            return None;
        }
        below.push(describe(&entry));

        if let Some(command) = command_if_due(below.len(), command_rx) {
            return Some(command);
        }
    }

    for item in below {
        if tx.send(WalkerMessage::FileMetadata(item)).is_err() {
            return None;
        }
    }

    None
}

/// The entries at depths `depth`, never entering a subtree we do not want.
fn entries<'a>(
    root: &Path,
    hidden: &'a [PathBuf],
    depth: std::ops::RangeInclusive<usize>,
) -> impl Iterator<Item = walkdir::DirEntry> + 'a {
    WalkDir::new(root)
        .follow_links(true)
        .min_depth(*depth.start())
        .max_depth(*depth.end())
        .into_iter()
        .filter_entry(move |entry| {
            // The root is where the user asked to be, so it is never skipped
            // for its name: running psychic inside a directory called `target`
            // used to show an empty screen.
            entry.depth() == 0 || should_descend(entry.file_type().is_dir(), entry.path(), hidden)
        })
        .filter_map(Result::ok)
}

/// Look for a command every [`COMMAND_CHECK_INTERVAL`] entries, so that a walk
/// of somewhere enormous can still be abandoned when the user moves on.
///
/// Any command supersedes the ones before it, so only the last is returned.
fn command_if_due(count: usize, command_rx: &Receiver<WalkerCommand>) -> Option<WalkerCommand> {
    if !count.is_multiple_of(COMMAND_CHECK_INTERVAL) {
        return None;
    }

    let mut latest = command_rx.try_recv().ok()?;
    while let Ok(newer) = command_rx.try_recv() {
        latest = newer;
    }

    log::info!("Walker: told to go elsewhere mid-walk; abandoning this one");
    Some(latest)
}

/// What the worker needs to know about an entry, from the metadata the walk
/// already had to fetch. Asking again later would be a second syscall.
fn describe(entry: &walkdir::DirEntry) -> WalkerFileMetadata {
    let metadata = entry.metadata().ok();

    WalkerFileMetadata {
        path: entry.path().to_path_buf(),
        mtime: metadata.as_ref().and_then(|m| m.mtime_as_secs()),
        atime: metadata.as_ref().and_then(|m| m.atime_as_secs()),
        file_size: metadata.as_ref().map(|m| m.len() as i64),
        is_dir: entry.file_type().is_dir(),
    }
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

    /// Walk a real tree and report what the walker emitted, in order.
    ///
    /// `None` marks `ChildrenDone`, so a test can see which paths were sent
    /// before it and which after.
    fn walk_messages(root: &Path, hidden: &[PathBuf], max_below: usize) -> Vec<Option<PathBuf>> {
        let (_command_tx, command_rx) = std::sync::mpsc::channel::<WalkerCommand>();
        let (tx, rx) = std::sync::mpsc::channel::<WalkerMessage>();

        walk_directory(root, hidden, max_below, &command_rx, &tx);
        drop(tx);

        rx.into_iter()
            .filter_map(|m| match m {
                WalkerMessage::FileMetadata(f) => Some(Some(f.path)),
                WalkerMessage::ChildrenDone => Some(None),
                WalkerMessage::AllDone => None,
            })
            .collect()
    }

    /// Every path the walker emitted, sorted. `max_below` is generous, so this
    /// is the whole tree unless a test says otherwise.
    fn walked_paths(root: &Path, hidden: &[PathBuf]) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = walk_messages(root, hidden, 10_000)
            .into_iter()
            .flatten()
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

    #[test]
    fn test_the_children_are_sent_first_and_announced() {
        let tree = TempTree::new("children-first");

        let messages = walk_messages(&tree.root, &[], 10_000);
        let announced = messages
            .iter()
            .position(|m| m.is_none())
            .expect("ChildrenDone must be sent");

        let before: Vec<&PathBuf> = messages[..announced].iter().flatten().collect();
        let after: Vec<&PathBuf> = messages[announced..].iter().flatten().collect();

        assert_eq!(
            before.len(),
            2,
            "Only the root's own children come first, got {:?}",
            before
        );
        for child in &before {
            assert_eq!(
                child.parent(),
                Some(tree.root.as_path()),
                "{:?} is not a child of the root",
                child
            );
        }
        assert!(
            after.contains(&&tree.root.join("b/c/d/buried.txt")),
            "Everything deeper follows: {:?}",
            after
        );
    }

    #[test]
    fn test_a_tree_too_big_to_index_keeps_its_children() {
        let tree = TempTree::new("too-big");

        // The tree has seven entries below the root's children; allow three.
        let walked = walked_paths_with_limit(&tree.root, 3);

        assert_eq!(
            walked,
            vec![tree.root.join("b"), tree.root.join("top.txt")],
            "Past the limit only the children stand, and they were already sent"
        );
    }

    #[test]
    fn test_a_tree_just_inside_the_limit_is_reported_whole() {
        let tree = TempTree::new("just-inside");

        // Seven entries live below the children: b/mid.txt, b/c, b/other,
        // b/c/near.txt, b/c/d, b/other/kept.txt, b/c/d/buried.txt.
        let walked = walked_paths_with_limit(&tree.root, 7);

        assert!(
            walked.contains(&tree.root.join("b/c/d/buried.txt")),
            "Exactly at the limit is still under it: {:?}",
            walked
        );
    }

    #[test]
    fn test_being_told_to_go_elsewhere_mid_walk_hands_the_command_back() {
        // The command that stops a walk used to be read off the channel and
        // dropped, after which start_file_walker blocked waiting for a command
        // that had already arrived - so navigating during a long walk left the
        // walker parked and the new directory never walked.
        let tree = TempTree::new("interrupted");
        let wide = tree.root.join("wide");
        std::fs::create_dir_all(&wide).expect("create wide dir");
        // More than COMMAND_CHECK_INTERVAL entries, so the check actually runs.
        for i in 0..(COMMAND_CHECK_INTERVAL * 2) {
            std::fs::write(wide.join(format!("{:04}.txt", i)), b"x").expect("create file");
        }

        let (command_tx, command_rx) = std::sync::mpsc::channel::<WalkerCommand>();
        let (tx, _rx) = std::sync::mpsc::channel::<WalkerMessage>();
        let elsewhere = PathBuf::from("/somewhere/else");
        command_tx
            .send(WalkerCommand::ChangeCwd {
                path: elsewhere.clone(),
                hidden: Vec::new(),
            })
            .expect("queue the command");

        let interrupted_by = walk_directory(&tree.root, &[], 10_000, &command_rx, &tx);

        let Some(WalkerCommand::ChangeCwd { path, .. }) = interrupted_by else {
            panic!("The walk must hand back the command that stopped it");
        };
        assert_eq!(path, elsewhere);
    }

    #[test]
    fn test_only_the_last_of_several_commands_survives() {
        let tree = TempTree::new("superseded");
        let wide = tree.root.join("wide");
        std::fs::create_dir_all(&wide).expect("create wide dir");
        for i in 0..(COMMAND_CHECK_INTERVAL * 2) {
            std::fs::write(wide.join(format!("{:04}.txt", i)), b"x").expect("create file");
        }

        let (command_tx, command_rx) = std::sync::mpsc::channel::<WalkerCommand>();
        let (tx, _rx) = std::sync::mpsc::channel::<WalkerMessage>();
        for path in ["/first", "/second", "/third"] {
            command_tx
                .send(WalkerCommand::ChangeCwd {
                    path: PathBuf::from(path),
                    hidden: Vec::new(),
                })
                .expect("queue the command");
        }

        let interrupted_by = walk_directory(&tree.root, &[], 10_000, &command_rx, &tx);

        let Some(WalkerCommand::ChangeCwd { path, .. }) = interrupted_by else {
            panic!("The walk must hand back a command");
        };
        assert_eq!(
            path,
            PathBuf::from("/third"),
            "Walking somewhere the user has already left is wasted work"
        );
    }

    fn walked_paths_with_limit(root: &Path, max_below: usize) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = walk_messages(root, &[], max_below)
            .into_iter()
            .flatten()
            .collect();
        paths.sort();
        paths
    }

    #[test]
    fn test_a_root_named_like_an_ignored_directory_is_still_walked() {
        // `should_descend` says no to a directory called `target`, but the root
        // is where the user asked to be. This used to show an empty screen.
        let tree = TempTree::new("ignored-name");
        let root = tree.root.join("target");
        std::fs::create_dir_all(&root).expect("create target dir");
        std::fs::write(root.join("inside.txt"), b"x").expect("create file");

        let walked = walked_paths(&root, &[]);

        assert_eq!(
            walked,
            vec![root.join("inside.txt")],
            "Running psychic inside a directory called `target` must still work"
        );
    }
}
