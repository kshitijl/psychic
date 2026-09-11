use crate::metadata_ext::MetadataExt;
use crate::search_worker::{WalkerCommand, WalkerFileMetadata, WalkerMessage};
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};

/// Directories never worth walking, whatever any ignore file says.
///
/// A floor under the gitignore rules, not a replacement for them: outside a git
/// repository there is nothing to read, and these four are noise everywhere.
/// `.git` is here rather than left to gitignore because no gitignore ever lists
/// it - git excludes it implicitly, and we walk dotfiles.
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

/// `message_tx` is generic so the walker can send straight into the worker's
/// own request channel, which is what lets the worker block on one `recv()`
/// instead of polling two channels.
pub fn start_file_walker<T>(
    initial_root: PathBuf,
    initial_hidden: Vec<PathBuf>,
    respect_gitignore: bool,
    command_rx: Receiver<WalkerCommand>,
    message_tx: Sender<T>,
) where
    T: From<WalkerMessage> + Send + 'static,
{
    std::thread::spawn(move || {
        let mut root = initial_root;
        let mut hidden = initial_hidden;

        loop {
            let interrupted_by = walk_directory(
                &root,
                &hidden,
                WalkLimits {
                    max_below: SHALLOW_MODE_THRESHOLD,
                    respect_gitignore,
                },
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
                    let _ = message_tx.send(WalkerMessage::AllDone.into());
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
/// What bounds a walk, beyond where it starts.
#[derive(Debug, Clone, Copy)]
struct WalkLimits {
    /// See [`SHALLOW_MODE_THRESHOLD`].
    max_below: usize,
    /// Whether ignore files are consulted. `--no-ignore` clears it.
    respect_gitignore: bool,
}

fn walk_directory<T>(
    root: &Path,
    hidden: &[PathBuf],
    limits: WalkLimits,
    command_rx: &Receiver<WalkerCommand>,
    tx: &Sender<T>,
) -> Option<WalkerCommand>
where
    T: From<WalkerMessage>,
{
    let mut sent = 0;
    for entry in entries(root, hidden, limits.respect_gitignore, 1..=1) {
        if sent >= MAX_FILES {
            log::warn!("Walker: {:?} has more than {} children", root, MAX_FILES);
            break;
        }
        if tx
            .send(WalkerMessage::FileMetadata(describe(&entry)).into())
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
    if tx.send(WalkerMessage::ChildrenDone.into()).is_err() {
        return None;
    }

    let mut below = Vec::new();
    for entry in entries(root, hidden, limits.respect_gitignore, 2..=usize::MAX) {
        if below.len() >= limits.max_below {
            log::info!(
                "Walker: more than {} entries below the children of {:?}; \
                 showing only its children",
                limits.max_below,
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
        if tx.send(WalkerMessage::FileMetadata(item).into()).is_err() {
            return None;
        }
    }

    None
}

/// The entries at depths `depth`, honouring ignore files and our own floor.
///
/// `respect_gitignore` is what `--no-ignore` turns off.
fn entries(
    root: &Path,
    hidden: &[PathBuf],
    respect_gitignore: bool,
    depth: std::ops::RangeInclusive<usize>,
) -> impl Iterator<Item = ignore::DirEntry> {
    // `filter_entry` wants a closure that outlives the walk, so the hidden
    // prefixes are copied in. There are a handful of them.
    let hidden = hidden.to_vec();

    let least = *depth.start();

    WalkBuilder::new(root)
        .follow_links(true)
        // Deliberately *not* `min_depth`. Setting it stops the crate applying
        // ignore rules to anything shallower, so an ignored directory at depth
        // one is never pruned and the entire thing is walked - `target` here is
        // 57,000 entries, enough on its own to push a repository past the
        // threshold and into showing only its top level. The depth range is
        // applied at the end instead, which costs one extra `readdir` of the
        // root on the second pass.
        .max_depth(Some(*depth.end()))
        // Dotfiles stay searchable. psychic is for finding `.zshrc` as much as
        // `main.rs`, and the model has an `is_hidden` feature that would go
        // blind if they never appeared. This is the one place we deliberately
        // differ from ripgrep's defaults.
        .hidden(false)
        // Everything git would ignore, and `.ignore` files besides, so a
        // directory can be kept out of psychic without touching `.gitignore`.
        .git_ignore(respect_gitignore)
        .git_exclude(respect_gitignore)
        .git_global(respect_gitignore)
        .ignore(respect_gitignore)
        // A repository's root `.gitignore` applies when psychic is launched in
        // one of its subdirectories.
        .parents(respect_gitignore)
        // Without this, ignore files are only consulted inside a git
        // repository. If someone wrote one, honour it wherever it is.
        .require_git(false)
        .filter_entry(move |entry| {
            // The root is where the user asked to be, so it is never skipped
            // for its name: running psychic inside a directory called `target`
            // used to show an empty screen.
            entry.depth() == 0
                || should_descend(
                    entry.file_type().is_some_and(|t| t.is_dir()),
                    entry.path(),
                    &hidden,
                )
        })
        .build()
        .filter_map(Result::ok)
        .filter(move |entry| entry.depth() >= least)
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
fn describe(entry: &ignore::DirEntry) -> WalkerFileMetadata {
    let metadata = entry.metadata().ok();

    WalkerFileMetadata {
        path: entry.path().to_path_buf(),
        mtime: metadata.as_ref().and_then(|m| m.mtime_as_secs()),
        file_size: metadata.as_ref().map(|m| m.len() as i64),
        is_dir: entry.file_type().is_some_and(|t| t.is_dir()),
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
    fn walk_messages(root: &Path, hidden: &[PathBuf], limits: WalkLimits) -> Vec<Option<PathBuf>> {
        let (_command_tx, command_rx) = std::sync::mpsc::channel::<WalkerCommand>();
        let (tx, rx) = std::sync::mpsc::channel::<WalkerMessage>();

        walk_directory(root, hidden, limits, &command_rx, &tx);
        drop(tx);

        rx.into_iter()
            .filter_map(|m| match m {
                WalkerMessage::FileMetadata(f) => Some(Some(f.path)),
                WalkerMessage::ChildrenDone => Some(None),
                WalkerMessage::AllDone => None,
            })
            .collect()
    }

    /// Limits that get in the way of nothing: a generous budget, ignore files
    /// respected. Tests that care about either say so.
    fn open_limits() -> WalkLimits {
        WalkLimits {
            max_below: 10_000,
            respect_gitignore: true,
        }
    }

    /// Every path the walker emitted, sorted.
    fn walked_paths(root: &Path, hidden: &[PathBuf]) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = walk_messages(root, hidden, open_limits())
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

    /// A tree with a `.gitignore` in it, and the noise it describes.
    struct IgnoredTree {
        root: PathBuf,
    }

    impl IgnoredTree {
        fn new(name: &str, gitignore: &str) -> Self {
            let root = std::env::temp_dir()
                .join(format!("psychic-ignore-{}-{}", name, std::process::id()))
                .join("project");
            let _ = std::fs::remove_dir_all(&root);
            for dir in ["src", "build", "src/nested"] {
                std::fs::create_dir_all(root.join(dir)).expect("create dir");
            }
            for file in [
                ".gitignore",
                "README.md",
                "notes.log",
                ".hidden-config",
                "src/main.rs",
                "src/main.o",
                "src/nested/deep.rs",
                "build/artifact.bin",
            ] {
                std::fs::write(root.join(file), b"x").expect("create file");
            }
            std::fs::write(root.join(".gitignore"), gitignore).expect("write gitignore");

            let root = root.canonicalize().expect("canonicalize");
            Self { root }
        }

        /// Paths under the root, relative and slash-separated, sorted.
        fn walked(&self, respect_gitignore: bool) -> Vec<String> {
            let limits = WalkLimits {
                max_below: 10_000,
                respect_gitignore,
            };
            let mut names: Vec<String> = walk_messages(&self.root, &[], limits)
                .into_iter()
                .flatten()
                .map(|p| {
                    p.strip_prefix(&self.root)
                        .unwrap_or(&p)
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for IgnoredTree {
        fn drop(&mut self) {
            if let Some(parent) = self.root.parent() {
                let _ = std::fs::remove_dir_all(parent);
            }
        }
    }

    #[test]
    fn test_gitignored_files_and_directories_are_skipped() {
        let tree = IgnoredTree::new("basic", "build/\n*.o\n*.log\n");

        let walked = tree.walked(true);

        assert!(!walked.contains(&"build".to_string()), "{:?}", walked);
        assert!(
            !walked.iter().any(|p| p.starts_with("build")),
            "nothing inside an ignored directory either: {:?}",
            walked
        );
        assert!(
            !walked.contains(&"src/main.o".to_string()),
            "ignoring files, not just directories, is most of the point: {:?}",
            walked
        );
        assert!(!walked.contains(&"notes.log".to_string()), "{:?}", walked);
        assert!(walked.contains(&"src/main.rs".to_string()), "{:?}", walked);
        assert!(
            walked.contains(&"src/nested/deep.rs".to_string()),
            "and the walk still goes deep where it should: {:?}",
            walked
        );
    }

    #[test]
    fn test_an_ignored_directory_at_the_top_is_pruned_by_the_second_pass_too() {
        // The hazard is the split between passes. The first covers depth one
        // and prunes `build` there; the second starts at depth two, so if it is
        // told to *begin* at depth two rather than to skip what it finds above,
        // nothing ever prunes `build` and everything under it is walked. That
        // is 57,000 entries for a `target` directory, which is by itself enough
        // to push a repository past the threshold and into showing only its top
        // level.
        let tree = IgnoredTree::new("depth-one", "build/\n");

        let walked = tree.walked(true);

        assert!(
            !walked.iter().any(|p| p.starts_with("build")),
            "the second pass has to prune it as well: {:?}",
            walked
        );
        assert!(
            walked.contains(&"src/nested/deep.rs".to_string()),
            "while still reaching everything below the depth it starts at: {:?}",
            walked
        );
    }

    #[test]
    fn test_dotfiles_are_still_searchable() {
        let tree = IgnoredTree::new("dotfiles", "build/\n");

        let walked = tree.walked(true);

        assert!(
            walked.contains(&".hidden-config".to_string()),
            "psychic is for finding .zshrc as much as main.rs, so it differs \
             from ripgrep here on purpose: {:?}",
            walked
        );
        assert!(
            walked.contains(&".gitignore".to_string()),
            "including the ignore file itself: {:?}",
            walked
        );
    }

    #[test]
    fn test_no_ignore_shows_everything() {
        let tree = IgnoredTree::new("no-ignore", "build/\n*.o\n*.log\n");

        let walked = tree.walked(false);

        for expected in ["build/artifact.bin", "src/main.o", "notes.log"] {
            assert!(
                walked.contains(&expected.to_string()),
                "--no-ignore means what it says, missing {}: {:?}",
                expected,
                walked
            );
        }
    }

    #[test]
    fn test_the_built_in_floor_applies_with_no_ignore_file() {
        // Outside a repository there is nothing to read, and these four are
        // noise everywhere.
        let tree = IgnoredTree::new("floor", "");
        for dir in ["node_modules", "target", ".venv"] {
            std::fs::create_dir_all(tree.root.join(dir)).expect("create dir");
            std::fs::write(tree.root.join(dir).join("junk.txt"), b"x").expect("create file");
        }

        let walked = tree.walked(true);

        for noise in ["node_modules", "target", ".venv"] {
            assert!(
                !walked.iter().any(|p| p.starts_with(noise)),
                "{} should be skipped with no gitignore to say so: {:?}",
                noise,
                walked
            );
        }
    }

    #[test]
    fn test_a_parent_gitignore_applies_from_a_subdirectory() {
        let tree = IgnoredTree::new("parents", "*.o\n");

        let limits = WalkLimits {
            max_below: 10_000,
            respect_gitignore: true,
        };
        let walked: Vec<PathBuf> = walk_messages(&tree.root.join("src"), &[], limits)
            .into_iter()
            .flatten()
            .collect();

        assert!(
            !walked.iter().any(|p| p.ends_with("main.o")),
            "launching inside a repository still obeys its root .gitignore: {:?}",
            walked
        );
        assert!(
            walked.iter().any(|p| p.ends_with("main.rs")),
            "{:?}",
            walked
        );
    }

    #[test]
    fn test_the_children_are_sent_first_and_announced() {
        let tree = TempTree::new("children-first");

        let messages = walk_messages(&tree.root, &[], open_limits());
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

        let interrupted_by = walk_directory(&tree.root, &[], open_limits(), &command_rx, &tx);

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

        let interrupted_by = walk_directory(&tree.root, &[], open_limits(), &command_rx, &tx);

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
        let mut paths: Vec<PathBuf> = walk_messages(
            root,
            &[],
            WalkLimits {
                max_below,
                ..open_limits()
            },
        )
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
