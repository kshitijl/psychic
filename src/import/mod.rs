//! Seed psychic with the files and directories the user already works with,
//! so that a fresh install has more to offer than whatever is under the
//! current directory.
//!
//! Sources ([`Source`]) are read into raw [`Sighting`]s by `sources.rs` and
//! `shell_history.rs`. This module then decides, for each path, whether it is
//! safe to touch at all ([`is_off_limits`]), resolves it without stepping into
//! anywhere that is not ([`resolve_safely`]), drops junk and gitignored paths,
//! and merges what is left into one capped [`ImportedPath`] per path.
//!
//! The result goes into the `imported_paths` table, never `events`: these are
//! not clicks, the model never trains on them, and `psychic stats` never sees
//! them. They make a path a search candidate and give the simple model one
//! extra term (`ranker.rs`). See "Importing what you already use" in
//! `docs/how-it-works.md`.

mod shell_history;
mod sources;

use crate::db::{Database, ImportedRow};
use anyhow::Result;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// Where a path was seen. A closed list: adding a source means adding it here,
/// and `name` will not compile until it has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Source {
    Zoxide,
    ZshHistory,
    BashHistory,
    FishHistory,
    VsCode,
    Cursor,
    Windsurf,
    Vim,
    Spotlight,
    ClaudeCode,
    Codex,
    Pi,
    JetBrains,
    GitRepos,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Zoxide => "zoxide",
            Source::ZshHistory => "zsh history",
            Source::BashHistory => "bash history",
            Source::FishHistory => "fish history",
            Source::VsCode => "VS Code",
            Source::Cursor => "Cursor",
            Source::Windsurf => "Windsurf",
            Source::Vim => "vim",
            Source::Spotlight => "Spotlight",
            Source::ClaudeCode => "Claude Code",
            Source::Codex => "Codex",
            Source::Pi => "pi",
            Source::JetBrains => "JetBrains",
            Source::GitRepos => "git repos",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PathKind {
    File,
    Dir,
}

/// One source saying "this path was used", before anything is checked.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Sighting {
    path: PathBuf,
    source: Source,
    uses: u32,
    /// Unix seconds, when the source records it.
    last_used: Option<i64>,
}

/// A path as it is stored: checked, canonical, merged across sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedPath {
    pub full_path: PathBuf,
    /// Summed across sources and capped at [`MAX_USES`].
    pub uses: u32,
    /// The latest use any source records; the import time when none does.
    pub last_used: i64,
    pub sources: BTreeSet<Source>,
}

/// What an import found, for printing and for the TUI's one-line summary.
pub struct ImportReport {
    pub paths: Vec<ImportedPath>,
}

impl ImportReport {
    /// Short enough for the TUI's title line: "Imported 212 paths you use
    /// (zoxide, zsh history, +4 more); undo: psychic import --undo".
    pub fn short_summary(&self) -> String {
        let sources = self.sources_by_count();
        if sources.is_empty() {
            return "Found no files or directories to import".to_string();
        }
        let mut named: Vec<&str> = sources.iter().take(2).map(|(s, _)| s.name()).collect();
        let more = sources.len().saturating_sub(2);
        let more = format!("+{more} more");
        if sources.len() > 2 {
            named.push(&more);
        }
        format!(
            "Imported {} paths you use ({}); undo: psychic import --undo",
            self.paths.len(),
            named.join(", ")
        )
    }

    /// Each source and how many kept paths it reported, most first.
    fn sources_by_count(&self) -> Vec<(Source, usize)> {
        let mut per_source: HashMap<Source, usize> = HashMap::new();
        for path in &self.paths {
            for source in &path.sources {
                *per_source.entry(*source).or_default() += 1;
            }
        }
        let mut counts: Vec<(Source, usize)> = per_source.into_iter().collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        counts
    }

    /// "Imported 212 paths from zoxide (150), zsh history (80), ...". A path
    /// seen by two sources counts towards both.
    pub fn summary(&self) -> String {
        if self.paths.is_empty() {
            return "Imported nothing: found no files or directories you use".to_string();
        }
        let listed: Vec<String> = self
            .sources_by_count()
            .iter()
            .map(|(source, count)| format!("{} ({})", source.name(), count))
            .collect();
        format!(
            "Imported {} paths from {}",
            self.paths.len(),
            listed.join(", ")
        )
    }
}

/// Most uses one path can carry. A directory `cd`'d into a thousand times and
/// one visited five times are both "used a lot"; past that, the simple model
/// should be deciding on fuzzy match and recency, not on which source counts
/// more generously.
const MAX_USES: u32 = 5;

/// Most paths one import keeps, most recently used first. Every one is a
/// registry entry each keystroke filters over, as history paths are.
const MAX_IMPORTED_PATHS: usize = 1_000;

/// Paths last used longer ago than this are left out, as history is.
const MAX_AGE_DAYS: i64 = 365;

// ---------------------------------------------------------------------------
// What may be touched
// ---------------------------------------------------------------------------

/// Folders macOS guards with a permission prompt ("Terminal would like to
/// access files in your Documents folder"), relative to `$HOME`.
///
/// Apple lists Desktop, Documents, Downloads, iCloud Drive, network volumes and
/// removable volumes as needing consent ("Controlling app access to files",
/// Apple Platform Security guide,
/// <https://support.apple.com/guide/security/secddd1d86a6/web>). iCloud Drive
/// lives in `~/Library/Mobile Documents`, and the other things that prompt -
/// cloud storage through File Provider (`~/Library/CloudStorage`), other apps'
/// containers, Mail, Messages, Safari - are under `~/Library` too, so all of
/// `~/Library` is left alone: nobody `cd`s there to work. The Trash prompts
/// as well.
const MACOS_PROTECTED_IN_HOME: &[&str] =
    &["Desktop", "Documents", "Downloads", "Library", ".Trash"];

/// Volumes: removable and network ones prompt, and an unmounted one hangs.
const MACOS_PROTECTED_ROOTS: &[&str] = &["/Volumes", "/Network"];

/// Must not be touched at all: stat'ing it could put up a macOS permission
/// prompt. Decided from the path's text alone, before any syscall.
fn is_off_limits(path: &Path, home: &Path) -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    if MACOS_PROTECTED_IN_HOME
        .iter()
        .any(|name| path.starts_with(home.join(name)))
        || MACOS_PROTECTED_ROOTS
            .iter()
            .any(|root| path.starts_with(root))
    {
        return true;
    }
    // Another user's home.
    if path.starts_with("/Users") && !path.starts_with(home) && path != Path::new("/Users") {
        return true;
    }
    // The Photos library prompts wherever it is.
    path.components().any(|component| {
        component
            .as_os_str()
            .to_string_lossy()
            .ends_with(".photoslibrary")
    })
}

/// Paths never worth suggesting, whatever a source says: scratch and system
/// directories, the inside of `.git` and `node_modules`, and the directories
/// above `$HOME` (`/`, `/Users`), which only ever appear on the way somewhere.
fn is_junk(path: &Path, home: &Path) -> bool {
    if home.starts_with(path) && home != path {
        return true;
    }
    const JUNK_ROOTS: &[&str] = &[
        "/tmp",
        "/private/tmp",
        "/var",
        "/private/var",
        "/dev",
        "/proc",
        "/sys",
        "/run",
    ];
    JUNK_ROOTS.iter().any(|root| path.starts_with(root))
        || path.components().any(|component| {
            let name = component.as_os_str();
            name == ".git" || name == "node_modules"
        })
}

/// A path resolved to its canonical form.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Resolved {
    path: PathBuf,
    kind: PathKind,
}

/// Most symlinks one resolution follows, as the kernel's own limit does.
const MAX_SYMLINK_HOPS: usize = 40;

/// `fs::canonicalize`, except that it checks each step against
/// [`is_off_limits`] *before* touching it.
///
/// `canonicalize` would follow `~/Dropbox` into `~/Library/CloudStorage/...`
/// and stat what is there, which is exactly the access that prompts. This
/// walks one component at a time, reads each symlink, and gives up the moment
/// the path it is building enters somewhere off limits.
fn resolve_safely(path: &Path, home: &Path) -> Option<Resolved> {
    assert!(path.is_absolute(), "resolve_safely wants an absolute path");
    let mut pending: Vec<std::ffi::OsString> = path
        .components()
        .rev()
        .filter_map(|component| match component {
            std::path::Component::Normal(name) => Some(name.to_os_string()),
            std::path::Component::ParentDir => Some("..".into()),
            std::path::Component::RootDir
            | std::path::Component::CurDir
            | std::path::Component::Prefix(_) => None,
        })
        .collect();
    let mut resolved = PathBuf::from("/");
    let mut hops = 0;
    let mut kind = PathKind::Dir;
    while let Some(name) = pending.pop() {
        if name == ".." {
            resolved.pop();
            continue;
        }
        let candidate = resolved.join(&name);
        if is_off_limits(&candidate, home) {
            return None;
        }
        let metadata = std::fs::symlink_metadata(&candidate).ok()?;
        if metadata.file_type().is_symlink() {
            hops += 1;
            if hops > MAX_SYMLINK_HOPS {
                return None;
            }
            let target = std::fs::read_link(&candidate).ok()?;
            if target.is_absolute() {
                resolved = PathBuf::from("/");
            }
            for component in target.components().rev() {
                match component {
                    std::path::Component::Normal(part) => pending.push(part.to_os_string()),
                    std::path::Component::ParentDir => pending.push("..".into()),
                    std::path::Component::RootDir
                    | std::path::Component::CurDir
                    | std::path::Component::Prefix(_) => {}
                }
            }
            continue;
        }
        kind = if metadata.is_dir() {
            PathKind::Dir
        } else {
            PathKind::File
        };
        resolved = candidate;
    }
    Some(Resolved {
        path: resolved,
        kind,
    })
}

// ---------------------------------------------------------------------------
// gitignore
// ---------------------------------------------------------------------------

/// Answers "would git ignore this?" for paths inside a repository, caching the
/// ignore files it reads. Paths outside any repository are never ignored.
struct IgnoreChecker {
    /// The repository root a directory is in, if any.
    repo_of: HashMap<PathBuf, Option<PathBuf>>,
    /// A directory's `.gitignore`, parsed, if it has one.
    gitignore_in: HashMap<PathBuf, Option<ignore::gitignore::Gitignore>>,
    global: ignore::gitignore::Gitignore,
}

impl IgnoreChecker {
    fn new() -> Self {
        let (global, _error) = ignore::gitignore::Gitignore::global();
        IgnoreChecker {
            repo_of: HashMap::new(),
            gitignore_in: HashMap::new(),
            global,
        }
    }

    fn repo_of(&mut self, dir: &Path) -> Option<PathBuf> {
        if let Some(known) = self.repo_of.get(dir) {
            return known.clone();
        }
        let repo = if dir.join(".git").exists() {
            Some(dir.to_path_buf())
        } else {
            dir.parent().and_then(|parent| self.repo_of(parent))
        };
        self.repo_of.insert(dir.to_path_buf(), repo.clone());
        repo
    }

    fn gitignore_in(&mut self, dir: &Path) -> Option<&ignore::gitignore::Gitignore> {
        self.gitignore_in
            .entry(dir.to_path_buf())
            .or_insert_with(|| {
                let file = dir.join(".gitignore");
                file.is_file()
                    .then(|| ignore::gitignore::Gitignore::new(&file).0)
            })
            .as_ref()
    }

    fn is_ignored(&mut self, path: &Path, kind: PathKind) -> bool {
        let Some(parent) = path.parent() else {
            return false;
        };
        let Some(repo) = self.repo_of(parent) else {
            return false;
        };
        // The path and each directory above it, up to the repository root:
        // `target/` ignores everything under it, not just itself.
        let mut candidates = vec![(path.to_path_buf(), kind == PathKind::Dir)];
        for ancestor in parent.ancestors() {
            if !ancestor.starts_with(&repo) || ancestor == repo {
                break;
            }
            candidates.push((ancestor.to_path_buf(), true));
        }
        // Deepest .gitignore first: it overrides the ones above it.
        let dirs: Vec<PathBuf> = parent
            .ancestors()
            .take_while(|dir| dir.starts_with(&repo))
            .map(Path::to_path_buf)
            .collect();
        for dir in dirs {
            let Some(gitignore) = self.gitignore_in(&dir) else {
                continue;
            };
            for (candidate, is_dir) in &candidates {
                if !candidate.starts_with(&dir) {
                    continue;
                }
                match gitignore.matched(candidate, *is_dir) {
                    ignore::Match::Ignore(_) => return true,
                    ignore::Match::Whitelist(_) => return false,
                    ignore::Match::None => {}
                }
            }
        }
        candidates
            .iter()
            .any(|(candidate, is_dir)| self.global.matched(candidate, *is_dir).is_ignore())
    }
}

// ---------------------------------------------------------------------------
// Putting it together
// ---------------------------------------------------------------------------

/// Group checked sightings by path, cap them, and keep the freshest.
fn merge(found: Vec<(PathBuf, Sighting)>, now: i64) -> Vec<ImportedPath> {
    let oldest = now - MAX_AGE_DAYS * 24 * 60 * 60;
    struct Merged {
        uses: u32,
        /// The latest time any source recorded. Sources that record none
        /// (zoxide, VS Code) do not count here, or one of them would make
        /// every path it shares with a dated source look used today.
        last_used: Option<i64>,
        sources: BTreeSet<Source>,
    }
    let mut by_path: HashMap<PathBuf, Merged> = HashMap::new();
    for (path, sighting) in found {
        let entry = by_path.entry(path).or_insert_with(|| Merged {
            uses: 0,
            last_used: None,
            sources: BTreeSet::new(),
        });
        entry.uses = (entry.uses + sighting.uses).min(MAX_USES);
        entry.last_used = entry.last_used.max(sighting.last_used.map(|t| t.min(now)));
        entry.sources.insert(sighting.source);
    }
    let mut paths: Vec<ImportedPath> = by_path
        .into_iter()
        .map(|(full_path, merged)| ImportedPath {
            full_path,
            uses: merged.uses,
            // Only undated sources saw it: they say "in use", so, now.
            last_used: merged.last_used.unwrap_or(now),
            sources: merged.sources,
        })
        .filter(|path| path.last_used >= oldest)
        .collect();
    paths.sort_by(|a, b| {
        b.last_used
            .cmp(&a.last_used)
            .then_with(|| a.full_path.cmp(&b.full_path))
    });
    paths.truncate(MAX_IMPORTED_PATHS);
    paths
}

/// Read every source, check every path, and merge. Touches nothing it should
/// not, writes nothing.
pub fn gather(home: &Path, now: i64) -> ImportReport {
    let mut cache: HashMap<PathBuf, Option<Resolved>> = HashMap::new();
    let mut resolve = |path: &Path| -> Option<Resolved> {
        if !path.is_absolute() || is_off_limits(path, home) || is_junk(path, home) {
            return None;
        }
        cache
            .entry(path.to_path_buf())
            .or_insert_with(|| resolve_safely(path, home))
            .clone()
    };

    let mut sightings = Vec::new();
    let timed = |name: &str, start: std::time::Instant, count: usize| {
        log::info!(
            "TIMING {{\"op\":\"import_{}\",\"ms\":{},\"count\":{}}}",
            name,
            start.elapsed().as_secs_f64() * 1000.0,
            count
        );
    };

    let start = std::time::Instant::now();
    let shell = sources::shell_histories(home, &mut |path| resolve(path).map(|r| r.kind));
    timed("shell_history", start, shell.len());
    sightings.extend(shell);

    let start = std::time::Instant::now();
    let git = sources::git_repos(home, &|path| is_off_limits(path, home));
    timed("git_repos", start, git.len());
    sightings.extend(git);

    for (name, read) in [
        ("zoxide", sources::zoxide as fn(&Path) -> Vec<Sighting>),
        ("vscode", sources::vscode_family),
        ("vim", sources::vim),
        ("spotlight", sources::spotlight),
        ("agents", sources::coding_agents),
        ("jetbrains", sources::jetbrains),
    ] {
        let start = std::time::Instant::now();
        let found = read(home);
        timed(name, start, found.len());
        sightings.extend(found);
    }

    let start = std::time::Instant::now();
    let mut ignore = IgnoreChecker::new();
    let found: Vec<(PathBuf, Sighting)> = sightings
        .into_iter()
        .filter_map(|sighting| {
            let resolved = resolve(&sighting.path)?;
            if is_off_limits(&resolved.path, home)
                || is_junk(&resolved.path, home)
                || ignore.is_ignored(&resolved.path, resolved.kind)
            {
                return None;
            }
            Some((resolved.path, sighting))
        })
        .collect();
    timed("check_paths", start, found.len());

    ImportReport {
        paths: merge(found, now),
    }
}

/// Gather, and replace whatever an earlier import stored.
pub fn run(db: &Database, home: &Path, now: i64) -> Result<ImportReport> {
    let report = gather(home, now);
    let rows: Vec<ImportedRow> = report
        .paths
        .iter()
        .map(|path| ImportedRow {
            full_path: path.full_path.to_string_lossy().into_owned(),
            uses: path.uses,
            last_used: path.last_used,
            sources: path
                .sources
                .iter()
                .map(|source| source.name())
                .collect::<Vec<_>>()
                .join(", "),
        })
        .collect();
    db.replace_imported_paths(&rows, now)?;
    Ok(report)
}

/// Import on its own only on what is, as far as psychic can tell, a new
/// install: no import has run, and nothing has been clicked or visited.
/// Someone with history has psychic's own record of what they use, and can
/// still run `psychic import` by hand.
pub fn should_auto_import(db: &Database) -> Result<bool> {
    Ok(!db.has_import_run()? && !db.has_engagements()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seen(path: &str, source: Source, uses: u32, last_used: Option<i64>) -> (PathBuf, Sighting) {
        (
            PathBuf::from(path),
            Sighting {
                path: PathBuf::from(path),
                source,
                uses,
                last_used,
            },
        )
    }

    const NOW: i64 = 1_800_000_000;
    const DAY: i64 = 24 * 60 * 60;

    #[test]
    fn merge_sums_caps_and_orders_by_recency() {
        let merged = merge(
            vec![
                seen("/h/a", Source::Zoxide, 4, None),
                seen("/h/a", Source::ZshHistory, 3, Some(NOW - DAY)),
                seen("/h/b", Source::Vim, 1, Some(NOW - 2 * DAY)),
                seen("/h/old", Source::Vim, 1, Some(NOW - 400 * DAY)),
                seen("/h/undated", Source::Zoxide, 1, None),
            ],
            NOW,
        );
        assert_eq!(
            merged,
            vec![
                ImportedPath {
                    full_path: PathBuf::from("/h/undated"),
                    uses: 1,
                    last_used: NOW,
                    sources: BTreeSet::from([Source::Zoxide]),
                },
                ImportedPath {
                    full_path: PathBuf::from("/h/a"),
                    uses: 5,
                    last_used: NOW - DAY,
                    sources: BTreeSet::from([Source::Zoxide, Source::ZshHistory]),
                },
                ImportedPath {
                    full_path: PathBuf::from("/h/b"),
                    uses: 1,
                    last_used: NOW - 2 * DAY,
                    sources: BTreeSet::from([Source::Vim]),
                },
            ]
        );
    }

    #[test]
    fn summary_counts_each_source() {
        let report = ImportReport {
            paths: merge(
                vec![
                    seen("/h/a", Source::Zoxide, 1, None),
                    seen("/h/a", Source::ZshHistory, 1, None),
                    seen("/h/b", Source::ZshHistory, 1, None),
                ],
                NOW,
            ),
        };
        assert_eq!(
            report.summary(),
            "Imported 2 paths from zsh history (2), zoxide (1)"
        );
        assert_eq!(
            report.short_summary(),
            "Imported 2 paths you use (zsh history, zoxide); undo: psychic import --undo"
        );
    }

    #[test]
    fn junk() {
        assert_eq!(
            [
                "/private/tmp/x",
                "/h/src/.git/config",
                "/h/web/node_modules/react",
                "/",
                "/home",
                "/home/me",
                "/home/me/src/app",
            ]
            .map(|path| is_junk(Path::new(path), Path::new("/home/me"))),
            [true, true, true, true, true, false, false]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn off_limits_on_macos() {
        let home = Path::new("/Users/me");
        assert_eq!(
            [
                "/Users/me/Documents/taxes.pdf",
                "/Users/me/Desktop",
                "/Users/me/Downloads/x.zip",
                "/Users/me/Library/CloudStorage/Dropbox/src",
                "/Users/me/Library/Mobile Documents/com~apple~CloudDocs/a",
                "/Users/me/.Trash/old",
                "/Volumes/USB/file",
                "/Users/someone/src",
                "/Users/me/Pictures/Photos Library.photoslibrary/database",
                "/Users/me/DocumentsArchive",
                "/Users/me/src/app",
                "/Users/me",
                "/opt/homebrew",
            ]
            .map(|path| is_off_limits(Path::new(path), home)),
            [
                true, true, true, true, true, true, true, true, true, false, false, false, false
            ]
        );
    }

    /// A scratch directory for the tests that need a real filesystem.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("psychic-import-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.canonicalize().unwrap()
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_safely_refuses_a_symlink_into_a_protected_folder() {
        // A fake home whose `Dropbox` links into its own `Library`, the way the
        // real Dropbox app sets it up. Nothing under `Library` may be touched,
        // so the link resolves to nothing - even though the target exists.
        let home = scratch("protected");
        std::fs::create_dir_all(home.join("Library/CloudStorage/Dropbox/src")).unwrap();
        std::os::unix::fs::symlink(
            home.join("Library/CloudStorage/Dropbox"),
            home.join("Dropbox"),
        )
        .unwrap();
        std::fs::create_dir_all(home.join("work")).unwrap();
        std::os::unix::fs::symlink("work", home.join("w")).unwrap();

        assert_eq!(resolve_safely(&home.join("Dropbox/src"), &home), None);
        assert_eq!(
            resolve_safely(&home.join("w/../work"), &home),
            Some(Resolved {
                path: home.join("work"),
                kind: PathKind::Dir,
            })
        );
        std::fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn gitignored_paths_are_ignored_and_others_are_not() {
        let root = scratch("ignore");
        let repo = root.join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(repo.join("target/release")).unwrap();
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::write(repo.join(".gitignore"), "target/\n*.log\n").unwrap();
        std::fs::write(repo.join("src/main.rs"), "").unwrap();
        std::fs::write(repo.join("src/debug.log"), "").unwrap();
        std::fs::create_dir_all(root.join("elsewhere")).unwrap();

        let mut checker = IgnoreChecker::new();
        assert_eq!(
            [
                (repo.join("target/release"), PathKind::Dir),
                (repo.join("src/debug.log"), PathKind::File),
                (repo.join("src/main.rs"), PathKind::File),
                (repo.clone(), PathKind::Dir),
                (root.join("elsewhere"), PathKind::Dir),
            ]
            .map(|(path, kind)| checker.is_ignored(&path, kind)),
            [true, true, false, false, false]
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
