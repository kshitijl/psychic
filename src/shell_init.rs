//! Shell integration: the scripts (`psychic zsh`, `psychic bash`,
//! `psychic fish`), and `psychic init`, which adds the line that loads one to
//! the shell's startup file, or takes it out again with `--reverse`.
//!
//! The decisions about the startup file are pure functions over its text
//! ([`plan_add`], [`plan_remove`]); [`run`] reads the file, asks, and writes.
//! The TUI's "not set up" hint ([`missing_integration_hint`]) lives here too,
//! because the variable it checks is set by the scripts this module ships.

use anyhow::{Context, Result, bail};
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

/// The shells psychic integrates with. Adding one means a script in `shell/`
/// and an arm in every match below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Shell {
    Zsh,
    Bash,
    Fish,
}

impl Shell {
    const ALL: [Shell; 3] = [Shell::Zsh, Shell::Bash, Shell::Fish];

    fn name(self) -> &'static str {
        match self {
            Shell::Zsh => "zsh",
            Shell::Bash => "bash",
            Shell::Fish => "fish",
        }
    }

    /// The integration script, as `psychic <shell>` prints it.
    pub fn script(self) -> &'static str {
        match self {
            Shell::Zsh => include_str!("../shell/psychic.zsh"),
            Shell::Bash => include_str!("../shell/psychic.bash"),
            Shell::Fish => include_str!("../shell/psychic.fish"),
        }
    }

    /// The line `init` adds, and the only form `--reverse` removes on its own.
    fn load_line(self) -> &'static str {
        match self {
            Shell::Zsh => r#"eval "$(psychic zsh)""#,
            Shell::Bash => r#"eval "$(psychic bash)""#,
            Shell::Fish => "psychic fish | source",
        }
    }

    /// The startup file every interactive shell of this kind reads.
    fn rc_path(self, home: &Path) -> PathBuf {
        match self {
            Shell::Zsh => match std::env::var_os("ZDOTDIR") {
                Some(dir) => PathBuf::from(dir).join(".zshrc"),
                None => home.join(".zshrc"),
            },
            Shell::Bash => home.join(".bashrc"),
            Shell::Fish => match std::env::var_os("XDG_CONFIG_HOME") {
                Some(dir) => PathBuf::from(dir).join("fish/config.fish"),
                None => home.join(".config/fish/config.fish"),
            },
        }
    }

    /// The shell named by `$SHELL`, the user's login shell.
    fn from_env() -> Result<Shell> {
        let shell = std::env::var("SHELL").unwrap_or_default();
        let name = Path::new(&shell)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        Shell::ALL
            .into_iter()
            .find(|candidate| candidate.name() == name)
            .with_context(|| {
                format!(
                    "$SHELL is {shell:?}, which psychic has no integration for; \
                     name one: psychic init zsh|bash|fish"
                )
            })
    }
}

/// Set by every script in `shell/`. Its absence is how the TUI knows the
/// shell integration is not loaded.
const SHELL_ENV_VAR: &str = "PSYCHIC_SHELL";

/// Written above the load line, so someone reading their startup file knows
/// where it came from and how to undo it. Starts with `#`, a comment in all
/// three shells.
const MARKER_LINE: &str = "# psychic shell integration (undo with `psychic init --reverse`)";

/// What `p`, `pd` and `pc` need, said in one title line.
const HINT: &str = "No shell integration (p, pd, pc, Ctrl-T): run `psychic init`";

/// The hint for the TUI's title line, or `None` when the integration is loaded.
///
/// It is a status message, so the first keypress clears it, and it comes back
/// on the next launch until the integration is set up. Setting `PSYCHIC_SHELL`
/// to anything silences it for someone who does not want the integration.
pub fn missing_integration_hint() -> Option<&'static str> {
    match std::env::var_os(SHELL_ENV_VAR) {
        Some(_) => None,
        None => Some(HINT),
    }
}

/// A line of the startup file, numbered from 1, as it is shown to the user.
#[derive(Debug, PartialEq, Eq)]
struct RcLine {
    number: usize,
    text: String,
}

#[derive(Debug, PartialEq, Eq)]
enum AddPlan {
    /// Some line already loads the integration; nothing to do.
    AlreadyPresent(RcLine),
    /// Append this text to the end of the file.
    Append(String),
}

#[derive(Debug, PartialEq, Eq)]
struct RemovePlan {
    /// The file's new contents, without the marker and the load line.
    new_contents: String,
    /// How many lines that took out.
    removed: usize,
    /// Lines that load the integration in some other form. Not ours to guess
    /// at, so they are left in and shown to the user.
    left_alone: Vec<RcLine>,
}

/// Does this line load the integration, in any form?
///
/// Deliberately loose - `eval "$(psychic zsh)"`, `source <(psychic zsh)`, a
/// full path to the binary - so that `init` never adds a second copy next to
/// one the user wrote themselves. Comments do not count.
fn loads_integration(line: &str, shell: Shell) -> bool {
    let trimmed = line.trim();
    !trimmed.starts_with('#') && trimmed.contains(&format!("psychic {}", shell.name()))
}

fn plan_add(rc: &str, shell: Shell) -> AddPlan {
    for (index, line) in rc.lines().enumerate() {
        if loads_integration(line, shell) {
            return AddPlan::AlreadyPresent(RcLine {
                number: index + 1,
                text: line.to_string(),
            });
        }
    }
    // A blank line before the block, unless the file is empty; and a newline
    // first if the last line lacks one, or the marker would join it.
    let separator = match (rc.is_empty(), rc.ends_with('\n')) {
        (true, _) => "",
        (false, true) => "\n",
        (false, false) => "\n\n",
    };
    AddPlan::Append(format!("{separator}{MARKER_LINE}\n{}\n", shell.load_line()))
}

fn plan_remove(rc: &str, shell: Shell) -> RemovePlan {
    let mut kept = Vec::new();
    let mut removed = 0;
    let mut left_alone = Vec::new();
    for (index, line) in rc.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed == MARKER_LINE {
            // `init` put a blank line before the marker; take that too, so
            // adding and removing leaves the file as it was.
            if kept
                .last()
                .is_some_and(|previous: &&str| previous.trim().is_empty())
            {
                kept.pop();
            }
            removed += 1;
            continue;
        }
        if trimmed == shell.load_line() {
            removed += 1;
            continue;
        }
        if loads_integration(line, shell) {
            left_alone.push(RcLine {
                number: index + 1,
                text: line.to_string(),
            });
        }
        kept.push(line);
    }
    let mut new_contents = kept.join("\n");
    if rc.ends_with('\n') && !new_contents.is_empty() {
        new_contents.push('\n');
    }
    RemovePlan {
        new_contents,
        removed,
        left_alone,
    }
}

/// macOS terminals start bash as a login shell, which reads `~/.bash_profile`
/// and not `~/.bashrc` - unless the profile sources it. Is that a problem for
/// this profile's text? `None` (no profile) is not: bash then falls back to
/// `~/.profile`, and a login shell with neither reads nothing either way.
fn bash_profile_skips_bashrc(profile: Option<&str>) -> bool {
    profile.is_some_and(|text| !text.contains("bashrc"))
}

/// How to remove the binary, judged by where it is installed.
fn binary_removal_command(exe: &Path) -> String {
    let path = exe.to_string_lossy();
    if path.contains("/Cellar/") {
        "brew uninstall psychic".to_string()
    } else if path.contains("/.cargo/bin/") {
        "cargo uninstall psychic-cli".to_string()
    } else {
        format!("rm {}", exe.display())
    }
}

fn home() -> Result<PathBuf> {
    Ok(PathBuf::from(
        std::env::var_os("HOME").context("HOME is not set")?,
    ))
}

/// `~` in place of the home directory, for messages.
fn tilde(path: &Path) -> String {
    match home() {
        Ok(home) => match path.strip_prefix(&home) {
            Ok(rest) => format!("~/{}", rest.display()),
            Err(_) => path.display().to_string(),
        },
        Err(_) => path.display().to_string(),
    }
}

/// Ask a yes/no question on the terminal. Anything but y or yes is no.
fn confirm(question: &str) -> Result<bool> {
    print!("{question} [y/N] ");
    std::io::stdout().flush()?;
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer)?;
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn read_if_present(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

/// `psychic init [SHELL] [--reverse] [--yes]`.
///
/// Adding uses the shell named, or `$SHELL`'s. Removing with no shell named
/// cleans all three startup files, since someone uninstalling may have set
/// up a shell they no longer use.
pub fn run(shell: Option<Shell>, reverse: bool, yes: bool, data_dir: &Path) -> Result<()> {
    if !yes && !std::io::stdin().is_terminal() {
        bail!("stdin is not a terminal, so psychic can't ask before editing; pass --yes");
    }
    let home = home()?;
    if reverse {
        let shells = match shell {
            Some(shell) => vec![shell],
            None => Shell::ALL.to_vec(),
        };
        for shell in shells {
            remove(shell, &shell.rc_path(&home), yes)?;
        }
        print_uninstall_hints(data_dir);
        Ok(())
    } else {
        let shell = match shell {
            Some(shell) => shell,
            None => Shell::from_env()?,
        };
        add(shell, &shell.rc_path(&home), &home, yes)
    }
}

fn add(shell: Shell, rc_path: &Path, home: &Path, yes: bool) -> Result<()> {
    let rc = read_if_present(rc_path)?.unwrap_or_default();
    let shown = tilde(rc_path);
    match plan_add(&rc, shell) {
        AddPlan::AlreadyPresent(line) => {
            println!(
                "Already set up: {shown} line {}: {}",
                line.number,
                line.text.trim()
            );
            Ok(())
        }
        AddPlan::Append(text) => {
            println!("psychic will add these lines to the end of {shown}:\n");
            for line in text.trim_start_matches('\n').lines() {
                println!("    {line}");
            }
            println!();
            if !yes && !confirm(&format!("Add them to {shown}?"))? {
                println!("Nothing changed.");
                return Ok(());
            }
            if let Some(parent) = rc_path.parent() {
                // fish's config directory may not exist yet.
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            // Appending through the path follows a symlink, so a dotfile
            // manager's link stays a link.
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(rc_path)
                .with_context(|| format!("opening {}", rc_path.display()))?;
            file.write_all(text.as_bytes())
                .with_context(|| format!("writing {}", rc_path.display()))?;
            println!(
                "Done. Open a new shell, or run `exec {}`, then try `p`, or Ctrl-T to insert a path.",
                shell.name()
            );
            if shell == Shell::Bash && cfg!(target_os = "macos") {
                let profile = home.join(".bash_profile");
                if bash_profile_skips_bashrc(read_if_present(&profile)?.as_deref()) {
                    println!(
                        "\nNote: macOS terminals start bash as a login shell, which reads \
                         ~/.bash_profile, and yours does not load ~/.bashrc. Add this to \
                         ~/.bash_profile:\n\n    [ -f ~/.bashrc ] && . ~/.bashrc"
                    );
                }
            }
            Ok(())
        }
    }
}

fn remove(shell: Shell, rc_path: &Path, yes: bool) -> Result<()> {
    let shown = tilde(rc_path);
    let Some(rc) = read_if_present(rc_path)? else {
        return Ok(());
    };
    let plan = plan_remove(&rc, shell);
    if plan.removed > 0 {
        println!(
            "psychic will remove {} line(s) from {shown}:\n",
            plan.removed
        );
        println!("    {MARKER_LINE}\n    {}\n", shell.load_line());
        if !yes && !confirm(&format!("Remove them from {shown}?"))? {
            println!("Nothing changed in {shown}.");
            return Ok(());
        }
        // Written through to the real file, so a symlinked startup file stays
        // a symlink rather than being replaced by a copy.
        let target = std::fs::canonicalize(rc_path)
            .with_context(|| format!("resolving {}", rc_path.display()))?;
        std::fs::write(&target, &plan.new_contents)
            .with_context(|| format!("writing {}", target.display()))?;
        println!("Removed. Shells already open keep p, pd and pc until you open a new one.");
    }
    for line in &plan.left_alone {
        println!(
            "Not removed, because psychic did not write it: {shown} line {}: {}",
            line.number,
            line.text.trim()
        );
    }
    Ok(())
}

fn print_uninstall_hints(data_dir: &Path) {
    println!("\nTo uninstall psychic completely:");
    let exe = std::env::current_exe().and_then(std::fs::canonicalize);
    match exe {
        Ok(exe) => println!("    {}", binary_removal_command(&exe)),
        Err(_) => println!("    remove the psychic binary from your PATH"),
    }
    println!(
        "    rm -r {}    # your click history and model",
        tilde(data_dir)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADDED_ZSH: &str = "\
# psychic shell integration (undo with `psychic init --reverse`)
eval \"$(psychic zsh)\"
";

    #[test]
    fn add_to_an_empty_file() {
        assert_eq!(
            plan_add("", Shell::Zsh),
            AddPlan::Append(ADDED_ZSH.to_string())
        );
    }

    #[test]
    fn each_shell_gets_its_own_load_line() {
        assert_eq!(
            Shell::ALL.map(|shell| match plan_add("", shell) {
                AddPlan::Append(text) => text.lines().last().unwrap_or("").to_string(),
                AddPlan::AlreadyPresent(_) => "present".to_string(),
            }),
            [
                r#"eval "$(psychic zsh)""#.to_string(),
                r#"eval "$(psychic bash)""#.to_string(),
                "psychic fish | source".to_string(),
            ]
        );
    }

    #[test]
    fn add_leaves_a_blank_line_after_existing_contents() {
        assert_eq!(
            plan_add("export EDITOR=vim\n", Shell::Zsh),
            AddPlan::Append(format!("\n{ADDED_ZSH}"))
        );
    }

    #[test]
    fn add_finishes_an_unterminated_last_line_first() {
        assert_eq!(
            plan_add("export EDITOR=vim", Shell::Zsh),
            AddPlan::Append(format!("\n\n{ADDED_ZSH}"))
        );
    }

    #[test]
    fn add_is_a_no_op_when_any_form_is_present() {
        assert_eq!(
            plan_add("export A=1\n  source <(/opt/bin/psychic zsh)\n", Shell::Zsh),
            AddPlan::AlreadyPresent(RcLine {
                number: 2,
                text: "  source <(/opt/bin/psychic zsh)".to_string(),
            })
        );
    }

    #[test]
    fn another_shells_line_does_not_count() {
        assert_eq!(
            plan_add("eval \"$(psychic zsh)\"\n", Shell::Bash),
            AddPlan::Append(
                "\n# psychic shell integration (undo with `psychic init --reverse`)\n\
                 eval \"$(psychic bash)\"\n"
                    .to_string()
            )
        );
    }

    #[test]
    fn add_ignores_a_commented_out_line() {
        assert_eq!(
            plan_add("# eval \"$(psychic zsh)\"\n", Shell::Zsh),
            AddPlan::Append(format!("\n{ADDED_ZSH}"))
        );
    }

    #[test]
    fn add_then_remove_gives_back_the_original() {
        let original = "set -x EDITOR vim\nalias ll 'ls -l'\n";
        let AddPlan::Append(added) = plan_add(original, Shell::Fish) else {
            panic!("expected an append");
        };
        assert_eq!(
            plan_remove(&format!("{original}{added}"), Shell::Fish),
            RemovePlan {
                new_contents: original.to_string(),
                removed: 2,
                left_alone: vec![],
            }
        );
    }

    #[test]
    fn add_then_remove_on_an_unterminated_file_only_adds_its_newline() {
        let AddPlan::Append(added) = plan_add("export EDITOR=vim", Shell::Zsh) else {
            panic!("expected an append");
        };
        assert_eq!(
            plan_remove(&format!("export EDITOR=vim{added}"), Shell::Zsh),
            RemovePlan {
                new_contents: "export EDITOR=vim\n".to_string(),
                removed: 2,
                left_alone: vec![],
            }
        );
    }

    #[test]
    fn remove_keeps_blank_lines_that_are_not_before_the_marker() {
        assert_eq!(
            plan_remove("a\n\neval \"$(psychic zsh)\"\n\nb\n", Shell::Zsh),
            RemovePlan {
                new_contents: "a\n\n\nb\n".to_string(),
                removed: 1,
                left_alone: vec![],
            }
        );
    }

    #[test]
    fn remove_leaves_other_forms_and_reports_them() {
        assert_eq!(
            plan_remove("a\nsource <(psychic bash)\n", Shell::Bash),
            RemovePlan {
                new_contents: "a\nsource <(psychic bash)\n".to_string(),
                removed: 0,
                left_alone: vec![RcLine {
                    number: 2,
                    text: "source <(psychic bash)".to_string(),
                }],
            }
        );
    }

    #[test]
    fn remove_from_a_file_that_is_only_the_block() {
        assert_eq!(
            plan_remove(ADDED_ZSH, Shell::Zsh),
            RemovePlan {
                new_contents: String::new(),
                removed: 2,
                left_alone: vec![],
            }
        );
    }

    #[test]
    fn bash_profile_check() {
        assert_eq!(
            [
                None,
                Some("export PATH=x\n"),
                Some("[ -f ~/.bashrc ] && . ~/.bashrc\n"),
            ]
            .map(bash_profile_skips_bashrc),
            [false, true, false]
        );
    }

    #[test]
    fn binary_removal_by_install_location() {
        assert_eq!(
            binary_removal_command(Path::new("/opt/homebrew/Cellar/psychic/0.3.0/bin/psychic")),
            "brew uninstall psychic"
        );
        assert_eq!(
            binary_removal_command(Path::new("/Users/a/.cargo/bin/psychic")),
            "cargo uninstall psychic-cli"
        );
        assert_eq!(
            binary_removal_command(Path::new("/Users/a/.local/bin/psychic")),
            "rm /Users/a/.local/bin/psychic"
        );
    }
}
