//! Directories and files from zsh, bash and fish history.
//!
//! None of the three records the directory a command ran in, so a relative
//! `cd foo` or `vim notes.md` means nothing on its own. We *replay* the
//! history instead: start in `$HOME`, follow every `cd`, and resolve each path
//! against where the replay says the shell was. As soon as the replay cannot
//! know - `cd` to somewhere that does not exist any more, `z foo`, `popd` - the
//! current directory becomes unknown and relative paths are skipped until an
//! absolute `cd` (or a bare `cd`, which goes home) pins it down again.
//!
//! It is a guess, and it is wrong whenever several terminals share one history
//! file (zsh's `SHARE_HISTORY`) and their commands interleave. Every path it
//! produces must exist on disk, which keeps a wrong guess from inventing
//! anything, though it can still credit a real directory the user never
//! visited.
//!
//! Parsing and replay are pure: the filesystem is reached only through the
//! `probe` the caller passes in, which is also what keeps macOS's protected
//! folders out of reach.

use super::{PathKind, Sighting, Source};
use std::path::{Component, Path, PathBuf};

/// One command from a history file, with when it ran if the file says.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct HistoryCommand {
    pub text: String,
    pub timestamp: Option<i64>,
}

/// Commands that leave the shell somewhere the replay cannot follow.
const CWD_LOSING_COMMANDS: &[&str] = &[
    "popd",
    "z",
    "zi",
    "j",
    "jo",
    "autojump",
    "__zoxide_z",
    "__zoxide_zi",
    "p",
    "pd",
    "pc",
];

/// Commands whose path arguments are files (or directories) the user opened.
const OPENING_COMMANDS: &[&str] = &[
    "vim",
    "vi",
    "nvim",
    "nano",
    "emacs",
    "emacsclient",
    "hx",
    "helix",
    "micro",
    "kak",
    "code",
    "cursor",
    "subl",
    "zed",
    "mate",
    "less",
    "bat",
    "open",
    "xdg-open",
];

// ---------------------------------------------------------------------------
// Reading the three formats
// ---------------------------------------------------------------------------

/// zsh stores non-ASCII bytes "metafied": 0x83 followed by the byte XOR 0x20.
fn unmetafy(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut iter = bytes.iter();
    while let Some(&byte) = iter.next() {
        if byte == 0x83 {
            if let Some(&next) = iter.next() {
                out.push(next ^ 0x20);
            }
        } else {
            out.push(byte);
        }
    }
    out
}

/// `~/.zsh_history`: either plain lines, or `: <start>:<elapsed>;<command>`
/// with `EXTENDED_HISTORY`. A command spanning lines ends each but the last
/// with a backslash.
pub(super) fn parse_zsh(bytes: &[u8]) -> Vec<HistoryCommand> {
    let text = String::from_utf8_lossy(&unmetafy(bytes)).into_owned();
    let mut commands = Vec::new();
    let mut pending: Option<HistoryCommand> = None;
    for line in text.lines() {
        let continued = match pending.take() {
            Some(mut command) => {
                command.text.push('\n');
                command.text.push_str(line);
                command
            }
            None => parse_zsh_line(line),
        };
        if continued.text.ends_with('\\') {
            let mut command = continued;
            command.text.pop();
            pending = Some(command);
        } else {
            commands.push(continued);
        }
    }
    commands.extend(pending);
    commands
}

fn parse_zsh_line(line: &str) -> HistoryCommand {
    if let Some(rest) = line.strip_prefix(": ")
        && let Some((header, command)) = rest.split_once(';')
        && let Some((start, _elapsed)) = header.split_once(':')
        && let Ok(timestamp) = start.trim().parse::<i64>()
    {
        return HistoryCommand {
            text: command.to_string(),
            timestamp: Some(timestamp),
        };
    }
    HistoryCommand {
        text: line.to_string(),
        timestamp: None,
    }
}

/// `~/.bash_history`: one command per line, each preceded by `#<epoch>` when
/// `HISTTIMEFORMAT` was set.
pub(super) fn parse_bash(bytes: &[u8]) -> Vec<HistoryCommand> {
    let text = String::from_utf8_lossy(bytes);
    let mut commands = Vec::new();
    let mut timestamp = None;
    for line in text.lines() {
        if let Some(digits) = line.strip_prefix('#')
            && let Ok(parsed) = digits.parse::<i64>()
        {
            timestamp = Some(parsed);
            continue;
        }
        commands.push(HistoryCommand {
            text: line.to_string(),
            timestamp: timestamp.take(),
        });
    }
    commands
}

/// `fish_history`: a YAML-like list of `- cmd: <command>` entries with a
/// `  when: <epoch>` line. Newlines and backslashes in the command are escaped
/// as `\n` and `\\`.
pub(super) fn parse_fish(bytes: &[u8]) -> Vec<HistoryCommand> {
    let text = String::from_utf8_lossy(bytes);
    let mut commands: Vec<HistoryCommand> = Vec::new();
    for line in text.lines() {
        if let Some(command) = line.strip_prefix("- cmd: ") {
            commands.push(HistoryCommand {
                text: unescape_fish(command),
                timestamp: None,
            });
        } else if let Some(when) = line.strip_prefix("  when: ")
            && let Some(last) = commands.last_mut()
        {
            last.timestamp = when.trim().parse().ok();
        }
    }
    commands
}

fn unescape_fish(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match (c, chars.clone().next()) {
            ('\\', Some('n')) => {
                out.push('\n');
                chars.next();
            }
            ('\\', Some('\\')) => {
                out.push('\\');
                chars.next();
            }
            _ => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Splitting a command line
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum Token {
    /// A word, unquoted. `None` when it holds something the replay cannot
    /// evaluate - a `$variable`, a `$(substitution)`, a glob - so the word's
    /// position is kept but its value is unknown.
    Word(Option<String>),
    /// `;`, `&&`, `||`, `|`, `&`, or a newline: the end of a simple command.
    Separator,
}

/// Split a command line into words and separators, the way a shell would
/// for the cases that matter here: quotes, backslashes, and the separators.
fn tokenize(line: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut evaluable = true;
    let mut chars = line.chars().peekable();

    fn finish(
        tokens: &mut Vec<Token>,
        word: &mut String,
        in_word: &mut bool,
        evaluable: &mut bool,
    ) {
        if *in_word {
            tokens.push(Token::Word(evaluable.then(|| std::mem::take(word))));
        }
        word.clear();
        *in_word = false;
        *evaluable = true;
    }

    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' => finish(&mut tokens, &mut word, &mut in_word, &mut evaluable),
            '\n' | ';' | '|' | '&' => {
                finish(&mut tokens, &mut word, &mut in_word, &mut evaluable);
                // `&&` and `||` are one separator, not two.
                if (c == '|' || c == '&') && chars.peek() == Some(&c) {
                    chars.next();
                }
                tokens.push(Token::Separator);
            }
            '\'' => {
                in_word = true;
                for quoted in chars.by_ref() {
                    if quoted == '\'' {
                        break;
                    }
                    word.push(quoted);
                }
            }
            '"' => {
                in_word = true;
                while let Some(quoted) = chars.next() {
                    match quoted {
                        '"' => break,
                        '\\' => {
                            if let Some(escaped) = chars.next() {
                                word.push(escaped);
                            }
                        }
                        '$' | '`' => {
                            evaluable = false;
                            word.push(quoted);
                        }
                        _ => word.push(quoted),
                    }
                }
            }
            '\\' => {
                in_word = true;
                if let Some(escaped) = chars.next() {
                    word.push(escaped);
                }
            }
            '$' | '`' | '*' | '?' | '[' | '{' | '(' | ')' | '<' | '>' => {
                in_word = true;
                evaluable = false;
                word.push(c);
            }
            _ => {
                in_word = true;
                word.push(c);
            }
        }
    }
    finish(&mut tokens, &mut word, &mut in_word, &mut evaluable);
    tokens
}

/// The simple commands in a line, each a list of words.
fn simple_commands(line: &str) -> Vec<Vec<Option<String>>> {
    let mut commands = vec![Vec::new()];
    for token in tokenize(line) {
        match token {
            Token::Word(word) => commands.last_mut().expect("never empty").push(word),
            Token::Separator => commands.push(Vec::new()),
        }
    }
    commands.retain(|words| !words.is_empty());
    commands
}

// ---------------------------------------------------------------------------
// Replaying
// ---------------------------------------------------------------------------

/// Join `path` onto `base`, resolving `.` and `..` by name, the way `cd`
/// does by default (`cd -L`).
fn join_logically(base: &Path, path: &Path) -> PathBuf {
    let mut joined = if path.is_absolute() {
        PathBuf::from("/")
    } else {
        base.to_path_buf()
    };
    for component in path.components() {
        match component {
            Component::RootDir | Component::Prefix(_) => joined = PathBuf::from("/"),
            Component::CurDir => {}
            Component::ParentDir => {
                joined.pop();
            }
            Component::Normal(name) => joined.push(name),
        }
    }
    joined
}

/// Where a word points, given where the shell is. `None` when it depends on
/// a directory the replay does not know.
fn resolve(word: &str, cwd: Option<&Path>, home: &Path) -> Option<PathBuf> {
    if word == "~" {
        return Some(home.to_path_buf());
    }
    if let Some(rest) = word.strip_prefix("~/") {
        return Some(join_logically(home, Path::new(rest)));
    }
    if word.starts_with('~') {
        // `~otheruser`: not ours to chase.
        return None;
    }
    let path = Path::new(word);
    if path.is_absolute() {
        return Some(join_logically(Path::new("/"), path));
    }
    cwd.map(|cwd| join_logically(cwd, path))
}

/// Replay `commands` from `$HOME`, reporting every directory `cd`'d into and
/// every path handed to an editor or pager. `probe` says whether a path exists
/// and what it is, and returns `None` for anything off limits.
pub(super) fn replay(
    commands: &[HistoryCommand],
    source: Source,
    home: &Path,
    probe: &mut dyn FnMut(&Path) -> Option<PathKind>,
) -> Vec<Sighting> {
    let mut sightings = Vec::new();
    let mut cwd: Option<PathBuf> = Some(home.to_path_buf());
    let mut previous: Option<PathBuf> = None;

    let sight = |path: PathBuf, timestamp: Option<i64>| Sighting {
        path,
        source,
        uses: 1,
        last_used: timestamp,
    };

    for command in commands {
        for words in simple_commands(&command.text) {
            // Skip `FOO=bar` assignments and `builtin`/`command` in front.
            let mut words = words.into_iter().skip_while(|word| match word {
                Some(word) => {
                    word.contains('=') && !word.starts_with('=')
                        || word == "builtin"
                        || word == "command"
                }
                None => false,
            });
            let Some(Some(name)) = words.next() else {
                continue;
            };
            let arguments: Vec<Option<String>> = words.collect();

            if name == "cd" || name == "pushd" {
                let operands: Vec<&Option<String>> = arguments
                    .iter()
                    .filter(
                        |word| !matches!(word, Some(word) if word.starts_with('-') && word != "-"),
                    )
                    .collect();
                let target = match operands.as_slice() {
                    [] => Some(home.to_path_buf()),
                    [Some(word)] if word == "-" => previous.clone(),
                    [Some(word)] => resolve(word, cwd.as_deref(), home),
                    _ => None,
                };
                let target = target.filter(|path| probe(path) == Some(PathKind::Dir));
                if let Some(target) = &target {
                    sightings.push(sight(target.clone(), command.timestamp));
                }
                previous = std::mem::replace(&mut cwd, target);
            } else if CWD_LOSING_COMMANDS.contains(&name.as_str()) {
                previous = cwd.take();
            } else if OPENING_COMMANDS.contains(&name.as_str()) {
                for word in arguments.iter().flatten() {
                    if word.starts_with('-') || word.starts_with('+') {
                        continue;
                    }
                    if let Some(path) = resolve(word, cwd.as_deref(), home)
                        && probe(&path).is_some()
                    {
                        sightings.push(sight(path, command.timestamp));
                    }
                }
            }
        }
    }
    sightings
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn command(text: &str, timestamp: Option<i64>) -> HistoryCommand {
        HistoryCommand {
            text: text.to_string(),
            timestamp,
        }
    }

    #[test]
    fn zsh_extended_plain_and_multiline() {
        let history = b": 1700000000:0;cd ~/src\nls\n: 1700000100:3;echo a \\\nb\n";
        assert_eq!(
            parse_zsh(history),
            vec![
                command("cd ~/src", Some(1700000000)),
                command("ls", None),
                command("echo a \nb", Some(1700000100)),
            ]
        );
    }

    #[test]
    fn zsh_unmetafies_non_ascii() {
        // "é" is 0xC3 0xA9; zsh writes 0xA9 as 0x83 0x89.
        let history = b": 1:0;cd caf\xC3\x83\x89\n";
        assert_eq!(parse_zsh(history), vec![command("cd café", Some(1))]);
    }

    #[test]
    fn bash_with_and_without_timestamps() {
        assert_eq!(
            parse_bash(b"#1700000000\ncd /tmp\nls\n"),
            vec![command("cd /tmp", Some(1700000000)), command("ls", None)]
        );
    }

    #[test]
    fn fish_entries() {
        let history = b"- cmd: cd ~/src\n  when: 1700000000\n- cmd: echo a\\nb\n  when: 1700000001\n  paths:\n    - src\n";
        assert_eq!(
            parse_fish(history),
            vec![
                command("cd ~/src", Some(1700000000)),
                command("echo a\nb", Some(1700000001)),
            ]
        );
    }

    #[test]
    fn tokenize_quotes_separators_and_unknowns() {
        assert_eq!(
            simple_commands(r#"cd "my dir" && vim 'a b'\ c; ls $HOME | less"#),
            vec![
                vec![Some("cd".to_string()), Some("my dir".to_string())],
                vec![Some("vim".to_string()), Some("a b c".to_string())],
                vec![Some("ls".to_string()), None],
                vec![Some("less".to_string())],
            ]
        );
    }

    /// A fake filesystem: directories and files by absolute path.
    fn fs(entries: &[(&str, PathKind)]) -> impl FnMut(&Path) -> Option<PathKind> {
        let map: HashMap<PathBuf, PathKind> = entries
            .iter()
            .map(|(path, kind)| (PathBuf::from(path), *kind))
            .collect();
        move |path| map.get(path).copied()
    }

    fn replayed(lines: &[&str], entries: &[(&str, PathKind)]) -> Vec<String> {
        let commands: Vec<HistoryCommand> =
            lines.iter().map(|line| command(line, Some(5))).collect();
        let mut probe = fs(entries);
        replay(&commands, Source::ZshHistory, Path::new("/h"), &mut probe)
            .into_iter()
            .map(|sighting| sighting.path.display().to_string())
            .collect()
    }

    #[test]
    fn replay_follows_relative_cds_from_home() {
        assert_eq!(
            replayed(
                &["cd src", "cd psychic", "vim README.md", "cd ..", "cd -"],
                &[
                    ("/h/src", PathKind::Dir),
                    ("/h/src/psychic", PathKind::Dir),
                    ("/h/src/psychic/README.md", PathKind::File),
                ],
            ),
            vec![
                "/h/src",
                "/h/src/psychic",
                "/h/src/psychic/README.md",
                "/h/src",
                "/h/src/psychic"
            ]
        );
    }

    #[test]
    fn replay_loses_the_cwd_on_a_missing_dir_and_regains_it_on_an_absolute_cd() {
        assert_eq!(
            replayed(
                &[
                    "cd gone",
                    "cd src",
                    "cd /h/src",
                    "cd psychic",
                    "cd",
                    "cd src"
                ],
                &[
                    ("/h", PathKind::Dir),
                    ("/h/src", PathKind::Dir),
                    ("/h/src/psychic", PathKind::Dir),
                ],
            ),
            vec!["/h/src", "/h/src/psychic", "/h", "/h/src"]
        );
    }

    #[test]
    fn replay_loses_the_cwd_after_a_fuzzy_jump() {
        assert_eq!(
            replayed(
                &["z proj", "cd src", "cd ~/src"],
                &[("/h/src", PathKind::Dir)],
            ),
            vec!["/h/src"]
        );
    }

    #[test]
    fn replay_skips_flags_and_unknown_words() {
        assert_eq!(
            replayed(
                &["FOO=1 vim -p +10 notes.md $X", "cd -P src"],
                &[("/h/notes.md", PathKind::File), ("/h/src", PathKind::Dir)],
            ),
            vec!["/h/notes.md", "/h/src"]
        );
    }

    #[test]
    fn replay_does_not_credit_a_cd_into_a_file() {
        assert_eq!(
            replayed(
                &["cd notes.md", "vim notes.md"],
                &[("/h/notes.md", PathKind::File)]
            ),
            Vec::<String>::new()
        );
    }
}
