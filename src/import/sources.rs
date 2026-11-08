//! Every place outside psychic that knows which files and directories the
//! user works with, read into [`Sighting`]s.
//!
//! Each reader returns whatever it found and never fails the import: a source
//! that is not installed is simply empty, and one that is unreadable is
//! logged and treated as empty. The text formats are parsed by pure functions
//! with tests; the functions that find and read the files are thin.
//!
//! Paths come out as the source wrote them, not yet checked. `mod.rs` decides
//! what is safe to touch, what exists, and what is junk.

use super::{PathKind, Sighting, Source, shell_history};
use std::path::{Path, PathBuf};
use std::process::Command;

fn sighting(path: PathBuf, source: Source, uses: u32, last_used: Option<i64>) -> Sighting {
    Sighting {
        path,
        source,
        uses,
        last_used,
    }
}

fn read_or_empty(path: &Path) -> Vec<u8> {
    match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => {
            log::warn!("import: could not read {}: {}", path.display(), error);
            Vec::new()
        }
    }
}

fn mtime(path: &Path) -> Option<i64> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    let seconds = modified
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    i64::try_from(seconds).ok()
}

/// Where per-user application data lives: `~/Library/Application Support`
/// on macOS, `$XDG_CONFIG_HOME` (or `~/.config`) elsewhere. VS Code and
/// JetBrains both follow this.
fn app_config_dir(home: &Path) -> PathBuf {
    if cfg!(target_os = "macos") {
        home.join("Library/Application Support")
    } else {
        match std::env::var_os("XDG_CONFIG_HOME") {
            Some(dir) => PathBuf::from(dir),
            None => home.join(".config"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shell history
// ---------------------------------------------------------------------------

pub(super) fn shell_histories(
    home: &Path,
    probe: &mut dyn FnMut(&Path) -> Option<PathKind>,
) -> Vec<Sighting> {
    let mut zsh_files = vec![home.join(".zsh_history")];
    if let Some(zdotdir) = std::env::var_os("ZDOTDIR") {
        let file = PathBuf::from(zdotdir).join(".zsh_history");
        if !zsh_files.contains(&file) {
            zsh_files.push(file);
        }
    }
    let fish_file = match std::env::var_os("XDG_DATA_HOME") {
        Some(dir) => PathBuf::from(dir).join("fish/fish_history"),
        None => home.join(".local/share/fish/fish_history"),
    };

    let mut sightings = Vec::new();
    for file in zsh_files {
        let commands = shell_history::parse_zsh(&read_or_empty(&file));
        sightings.extend(shell_history::replay(
            &commands,
            Source::ZshHistory,
            home,
            probe,
        ));
    }
    let commands = shell_history::parse_bash(&read_or_empty(&home.join(".bash_history")));
    sightings.extend(shell_history::replay(
        &commands,
        Source::BashHistory,
        home,
        probe,
    ));
    let commands = shell_history::parse_fish(&read_or_empty(&fish_file));
    sightings.extend(shell_history::replay(
        &commands,
        Source::FishHistory,
        home,
        probe,
    ));
    sightings
}

// ---------------------------------------------------------------------------
// zoxide
// ---------------------------------------------------------------------------

/// zoxide's score, roughly visits weighted by recency, as a use count: its
/// log, so a directory visited hundreds of times does not swamp the rest.
/// The import caps uses anyway; this keeps the order among the capped ones.
fn zoxide_uses(score: f64) -> u32 {
    (score.max(1.0).ln().ceil() as u32).max(1)
}

/// `zoxide query --list --score`: `<score> <path>` per line, score padded.
fn parse_zoxide(output: &str) -> Vec<Sighting> {
    output
        .lines()
        .filter_map(|line| {
            let (score, path) = line.trim_start().split_once(' ')?;
            let score: f64 = score.parse().ok()?;
            Some(sighting(
                PathBuf::from(path.trim_start()),
                Source::Zoxide,
                zoxide_uses(score),
                None,
            ))
        })
        .collect()
}

/// Asks the `zoxide` binary rather than reading its database, whose format is
/// zoxide's own business and whose location it decides (`_ZO_DATA_DIR`).
pub(super) fn zoxide(_home: &Path) -> Vec<Sighting> {
    match Command::new("zoxide")
        .args(["query", "--list", "--score"])
        .output()
    {
        Ok(output) if output.status.success() => {
            parse_zoxide(&String::from_utf8_lossy(&output.stdout))
        }
        // Not installed, or no database yet.
        Ok(_) | Err(_) => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// VS Code and its forks
// ---------------------------------------------------------------------------

/// Decode `%XX` escapes in a `file://` URI's path.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && let Some(hex) = text.get(index + 1..index + 3)
            && let Ok(byte) = u8::from_str_radix(hex, 16)
        {
            out.push(byte);
            index += 3;
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The value of `history.recentlyOpenedPathsList`: `{"entries": [...]}`, each
/// a `folderUri` or a `fileUri`, most recent first. Remote URIs
/// (`vscode-remote://`) and workspaces are skipped.
fn parse_vscode_recents(json: &str, source: Source) -> Vec<Sighting> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(json) else {
        return Vec::new();
    };
    let Some(entries) = value.get("entries").and_then(|entries| entries.as_array()) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let uri = entry
                .get("folderUri")
                .or_else(|| entry.get("fileUri"))?
                .as_str()?;
            let path = uri.strip_prefix("file://")?;
            Some(sighting(
                PathBuf::from(percent_decode(path)),
                source,
                1,
                None,
            ))
        })
        .collect()
}

pub(super) fn vscode_family(home: &Path) -> Vec<Sighting> {
    let apps = [
        ("Code", Source::VsCode),
        ("Code - Insiders", Source::VsCode),
        ("VSCodium", Source::VsCode),
        ("Cursor", Source::Cursor),
        ("Windsurf", Source::Windsurf),
    ];
    let config = app_config_dir(home);
    let mut sightings = Vec::new();
    for (app, source) in apps {
        let db = config.join(app).join("User/globalStorage/state.vscdb");
        if !db.exists() {
            continue;
        }
        match read_vscode_db(&db) {
            Ok(Some(json)) => sightings.extend(parse_vscode_recents(&json, source)),
            Ok(None) => {}
            Err(error) => log::warn!("import: could not read {}: {}", db.display(), error),
        }
    }
    sightings
}

fn read_vscode_db(db: &Path) -> rusqlite::Result<Option<String>> {
    // Read-only, so an editor that is running and writing it is not disturbed.
    let conn = rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let mut stmt =
        conn.prepare("SELECT value FROM ItemTable WHERE key = 'history.recentlyOpenedPathsList'")?;
    let mut rows = stmt.query([])?;
    match rows.next()? {
        Some(row) => {
            // Stored as TEXT by some versions and BLOB by others.
            let value = row.get_ref(0)?;
            Ok(match value {
                rusqlite::types::ValueRef::Text(bytes) | rusqlite::types::ValueRef::Blob(bytes) => {
                    Some(String::from_utf8_lossy(bytes).into_owned())
                }
                rusqlite::types::ValueRef::Null
                | rusqlite::types::ValueRef::Integer(_)
                | rusqlite::types::ValueRef::Real(_) => None,
            })
        }
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Vim
// ---------------------------------------------------------------------------

/// `~/.viminfo`'s "History of marks within files": a `> <path>` line, then
/// tab-indented marks, the `*` one carrying when the file was last left.
fn parse_viminfo(text: &str, home: &Path) -> Vec<Sighting> {
    let mut sightings: Vec<Sighting> = Vec::new();
    let mut in_entry = false;
    for line in text.lines() {
        if let Some(path) = line.strip_prefix("> ") {
            let path = match path.strip_prefix("~/") {
                Some(rest) => home.join(rest),
                None => PathBuf::from(path),
            };
            sightings.push(sighting(path, Source::Vim, 1, None));
            in_entry = true;
        } else if in_entry && let Some(rest) = line.strip_prefix("\t*\t") {
            if let Some(last) = sightings.last_mut() {
                last.last_used = rest.split('\t').next().and_then(|t| t.parse().ok());
            }
        } else if !line.starts_with('\t') {
            in_entry = false;
        }
    }
    sightings
}

pub(super) fn vim(home: &Path) -> Vec<Sighting> {
    let bytes = read_or_empty(&home.join(".viminfo"));
    parse_viminfo(&String::from_utf8_lossy(&bytes), home)
}

// ---------------------------------------------------------------------------
// Spotlight
// ---------------------------------------------------------------------------

/// One line of `mdfind -attr kMDItemLastUsedDate -attr kMDItemUseCount`:
/// `<path>   kMDItemLastUsedDate = 2026-11-06 06:02:04 +0000   kMDItemUseCount = 4`.
/// Either value can be `(null)`.
fn parse_mdfind_line(line: &str) -> Option<Sighting> {
    let (path, rest) = line.split_once("   kMDItemLastUsedDate = ")?;
    let (date, rest) = rest.split_once("   kMDItemUseCount = ")?;
    let last_used = jiff::Timestamp::strptime("%Y-%m-%d %H:%M:%S %z", date.trim())
        .ok()
        .map(|timestamp| timestamp.as_second());
    let uses = rest.trim().parse().unwrap_or(1).max(1);
    // Applications show up because launching one sets its last-used date.
    if Path::new(path)
        .components()
        .any(|component| component.as_os_str().to_string_lossy().ends_with(".app"))
    {
        return None;
    }
    Some(sighting(
        PathBuf::from(path),
        Source::Spotlight,
        uses,
        last_used,
    ))
}

/// What the Finder and other apps opened in the last year, from Spotlight's
/// index. macOS only. Asking `mdfind` touches no files itself: the index
/// answers, so a result in a protected folder costs nothing until something
/// opens it, and `mod.rs` filters those out before anything does.
pub(super) fn spotlight(home: &Path) -> Vec<Sighting> {
    if !cfg!(target_os = "macos") {
        return Vec::new();
    }
    let output = Command::new("mdfind")
        .arg("-onlyin")
        .arg(home)
        .args([
            "-attr",
            "kMDItemLastUsedDate",
            "-attr",
            "kMDItemUseCount",
            "kMDItemLastUsedDate >= $time.today(-365)",
        ])
        .output();
    match output {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(parse_mdfind_line)
            .collect(),
        Ok(_) | Err(_) => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Coding agents: Claude Code, Codex, pi
// ---------------------------------------------------------------------------

/// How many lines of a transcript to look through for its working directory.
/// It is on the first line or near it in every format seen; the cap keeps a
/// transcript without one from being read to the end.
const TRANSCRIPT_LINES_TO_SCAN: usize = 50;

/// The working directory in one JSONL transcript line, if it has one:
/// top-level `cwd` (Claude Code; pi's `session` header) or `payload.cwd`
/// (Codex's `session_meta`).
fn transcript_cwd(line: &str) -> Option<PathBuf> {
    if !line.contains("\"cwd\"") {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    let cwd = value
        .get("cwd")
        .or_else(|| value.get("payload").and_then(|payload| payload.get("cwd")))?
        .as_str()?;
    Some(PathBuf::from(cwd))
}

fn transcript_file_cwd(file: &Path) -> Option<PathBuf> {
    use std::io::BufRead;
    let reader = std::io::BufReader::new(std::fs::File::open(file).ok()?);
    reader
        .lines()
        .take(TRANSCRIPT_LINES_TO_SCAN)
        .map_while(Result::ok)
        .find_map(|line| transcript_cwd(&line))
}

/// Every `.jsonl` under `dir`, at most `depth` directories down.
fn jsonl_files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if file_type.is_dir() && depth > 0 {
            jsonl_files(&path, depth - 1, out);
        } else if file_type.is_file() && path.extension().is_some_and(|ext| ext == "jsonl") {
            out.push(path);
        }
    }
}

/// One sighting per session: its directory, used once, at the transcript's
/// mtime - when the session last wrote anything.
fn agent_sessions(sessions_dir: &Path, depth: usize, source: Source) -> Vec<Sighting> {
    let mut files = Vec::new();
    jsonl_files(sessions_dir, depth, &mut files);
    files
        .iter()
        .filter_map(|file| {
            let cwd = transcript_file_cwd(file)?;
            Some(sighting(cwd, source, 1, mtime(file)))
        })
        .collect()
}

pub(super) fn coding_agents(home: &Path) -> Vec<Sighting> {
    let env_dir = |var: &str, default: &str| match std::env::var_os(var) {
        Some(dir) => PathBuf::from(dir),
        None => home.join(default),
    };
    let mut sightings = Vec::new();
    // ~/.claude/projects/<encoded cwd>/<session>.jsonl
    sightings.extend(agent_sessions(
        &env_dir("CLAUDE_CONFIG_DIR", ".claude").join("projects"),
        1,
        Source::ClaudeCode,
    ));
    // ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl
    sightings.extend(agent_sessions(
        &env_dir("CODEX_HOME", ".codex").join("sessions"),
        3,
        Source::Codex,
    ));
    // ~/.pi/agent/sessions/<encoded cwd>/<timestamp>_<id>.jsonl
    sightings.extend(agent_sessions(
        &home.join(".pi/agent/sessions"),
        1,
        Source::Pi,
    ));
    sightings
}

// ---------------------------------------------------------------------------
// JetBrains IDEs
// ---------------------------------------------------------------------------

fn xml_unescape(text: &str) -> String {
    text.replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

/// `options/recentProjects.xml`: one `<entry key="<path>">` per project, the
/// path written with `$USER_HOME$` for the home directory, and inside it the
/// project's meta info, whose `projectOpenTimestamp` is in milliseconds.
fn parse_jetbrains_recents(xml: &str, home: &Path) -> Vec<Sighting> {
    let home = home.to_string_lossy();
    let mut sightings = Vec::new();
    for chunk in xml.split("<entry key=\"").skip(1) {
        let Some((key, body)) = chunk.split_once('"') else {
            continue;
        };
        let body = body.split("</entry>").next().unwrap_or(body);
        let last_used = body
            .split_once("name=\"projectOpenTimestamp\" value=\"")
            .and_then(|(_, rest)| rest.split('"').next())
            .and_then(|millis| millis.parse::<i64>().ok())
            .map(|millis| millis / 1000);
        let path = xml_unescape(key).replace("$USER_HOME$", &home);
        sightings.push(sighting(
            PathBuf::from(path),
            Source::JetBrains,
            1,
            last_used,
        ));
    }
    sightings
}

pub(super) fn jetbrains(home: &Path) -> Vec<Sighting> {
    let config = app_config_dir(home);
    // One directory per product and version: IntelliJIdea2026.2, PyCharm2025.3...
    // Android Studio is a JetBrains IDE that lives under Google.
    let mut files = Vec::new();
    for vendor in ["JetBrains", "Google"] {
        let Ok(entries) = std::fs::read_dir(config.join(vendor)) else {
            continue;
        };
        for entry in entries.flatten() {
            let file = entry.path().join("options/recentProjects.xml");
            if file.is_file() {
                files.push(file);
            }
        }
    }
    files
        .iter()
        .flat_map(|file| {
            parse_jetbrains_recents(&String::from_utf8_lossy(&read_or_empty(file)), home)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Git repositories
// ---------------------------------------------------------------------------

/// How deep under `$HOME` to look for repositories. `~/src/project` is depth
/// 2, `~/work/team/project/service` is 4.
const GIT_SEARCH_DEPTH: usize = 4;

/// Most directories to list while looking, so that a home directory full of
/// data cannot make the import slow.
const GIT_SEARCH_MAX_DIRS: usize = 20_000;

/// Directories never worth descending into while looking for repositories.
const GIT_SEARCH_SKIP: &[&str] = &["node_modules", "target", "venv", "Applications"];

/// Repositories under `$HOME`, each at the time its index last changed - a
/// commit, a checkout, `git add`, and usually `git status` all write it.
/// Hidden directories, symlinks, anything `is_off_limits` rejects, and the
/// inside of a repository are not searched.
pub(super) fn git_repos(home: &Path, is_off_limits: &dyn Fn(&Path) -> bool) -> Vec<Sighting> {
    let mut sightings = Vec::new();
    let mut queue = std::collections::VecDeque::from([(home.to_path_buf(), 0usize)]);
    let mut listed = 0;
    while let Some((dir, depth)) = queue.pop_front() {
        if listed >= GIT_SEARCH_MAX_DIRS {
            log::warn!("import: stopped looking for git repositories after {listed} directories");
            break;
        }
        listed += 1;
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut children = Vec::new();
        let mut git_dir = None;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == ".git" {
                git_dir = Some(entry.path());
                break;
            }
            // `file_type` does not follow symlinks, so a link is never a dir here.
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir()
                && !name.starts_with('.')
                && !GIT_SEARCH_SKIP.contains(&name.as_ref())
                && depth < GIT_SEARCH_DEPTH
            {
                let path = entry.path();
                if !is_off_limits(&path) {
                    children.push((path, depth + 1));
                }
            }
        }
        match git_dir {
            Some(git_dir) => {
                // A worktree's `.git` is a file; its own mtime is the best we have.
                let last_used = mtime(&git_dir.join("index")).or_else(|| mtime(&git_dir));
                sightings.push(sighting(dir, Source::GitRepos, 1, last_used));
            }
            None => queue.extend(children),
        }
    }
    sightings
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(sightings: &[Sighting]) -> Vec<String> {
        sightings
            .iter()
            .map(|s| {
                format!(
                    "{} uses={} last={:?}",
                    s.path.display(),
                    s.uses,
                    s.last_used
                )
            })
            .collect()
    }

    #[test]
    fn zoxide_scores_become_log_uses() {
        assert_eq!(
            summary(&parse_zoxide("  33.5 /h/a b\n   1.0 /h/c\n  0.2 /h/d\n")),
            vec![
                "/h/a b uses=4 last=None",
                "/h/c uses=1 last=None",
                "/h/d uses=1 last=None",
            ]
        );
    }

    #[test]
    fn vscode_recents_skip_remote_and_workspaces() {
        let json = r#"{"entries":[
            {"folderUri":"file:///h/my%20proj"},
            {"fileUri":"file:///h/notes.md"},
            {"folderUri":"vscode-remote://ssh-remote%2Bbox/h/x"},
            {"workspace":{"id":"1","configPath":"file:///h/w.code-workspace"}}
        ]}"#;
        assert_eq!(
            summary(&parse_vscode_recents(json, Source::VsCode)),
            vec![
                "/h/my proj uses=1 last=None",
                "/h/notes.md uses=1 last=None"
            ]
        );
    }

    #[test]
    fn viminfo_marks_with_timestamps() {
        let text = "# History of marks within files (newest to oldest):\n\n\
                    > ~/src/a.rs\n\t*\t1763855733\t0\n\t\"\t238\t45\n\n\
                    > /etc/hosts\n\t\"\t1\t0\n";
        assert_eq!(
            summary(&parse_viminfo(text, Path::new("/h"))),
            vec![
                "/h/src/a.rs uses=1 last=Some(1763855733)",
                "/etc/hosts uses=1 last=None"
            ]
        );
    }

    #[test]
    fn mdfind_lines() {
        assert_eq!(
            summary(
                &[
                    "/h/a b.pdf   kMDItemLastUsedDate = 2026-11-06 06:02:04 +0000   kMDItemUseCount = 4",
                    "/h/x   kMDItemLastUsedDate = (null)   kMDItemUseCount = (null)",
                    "/h/Applications/Foo.app   kMDItemLastUsedDate = 2026-11-06 06:02:04 +0000   kMDItemUseCount = 9",
                ]
                .iter()
                .filter_map(|line| parse_mdfind_line(line))
                .collect::<Vec<_>>()
            ),
            vec!["/h/a b.pdf uses=4 last=Some(1793944924)", "/h/x uses=1 last=None"]
        );
    }

    #[test]
    fn transcript_cwd_in_each_agents_format() {
        let claude = r#"{"type":"user","cwd":"/h/claude","sessionId":"x"}"#;
        let codex =
            r#"{"timestamp":"t","type":"session_meta","payload":{"id":"x","cwd":"/h/codex"}}"#;
        let pi = r#"{"type":"session","version":3,"id":"x","timestamp":"t","cwd":"/h/pi"}"#;
        let none = r#"{"type":"mode","mode":"normal"}"#;
        assert_eq!(
            [claude, codex, pi, none].map(transcript_cwd),
            [
                Some(PathBuf::from("/h/claude")),
                Some(PathBuf::from("/h/codex")),
                Some(PathBuf::from("/h/pi")),
                None,
            ]
        );
    }

    #[test]
    fn jetbrains_recent_projects() {
        // Written from the shape of IntelliJ's RecentProjectsManager; not
        // checked against a real file, as there is no JetBrains IDE here.
        let xml = r#"<application>
  <component name="RecentProjectsManager">
    <option name="additionalInfo">
      <map>
        <entry key="$USER_HOME$/src/app">
          <value>
            <RecentProjectMetaInfo frameTitle="app">
              <option name="projectOpenTimestamp" value="1700000000123" />
            </RecentProjectMetaInfo>
          </value>
        </entry>
        <entry key="/opt/a &amp; b">
          <value><RecentProjectMetaInfo /></value>
        </entry>
      </map>
    </option>
  </component>
</application>"#;
        assert_eq!(
            summary(&parse_jetbrains_recents(xml, Path::new("/h"))),
            vec![
                "/h/src/app uses=1 last=Some(1700000000)",
                "/opt/a & b uses=1 last=None"
            ]
        );
    }
}
