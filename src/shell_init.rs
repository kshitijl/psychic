//! `psychic init`: add the shell integration line to the user's `~/.zshrc`, or
//! take it out again with `--reverse`.
//!
//! The decisions about the rc file are pure functions over its text
//! ([`plan_add`], [`plan_remove`]); [`run`] reads the file, asks, and writes.
//! The TUI's "not set up" hint ([`missing_integration_hint`]) lives here too,
//! because the variable it checks is set by the script this module installs.

use anyhow::{Context, Result, bail};
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

/// Set by `shell/psychic.zsh`. Its absence is how the TUI knows the shell
/// integration is not loaded.
const SHELL_ENV_VAR: &str = "PSYCHIC_SHELL";

/// The line `init` adds, and the only form `--reverse` removes on its own.
const EVAL_LINE: &str = r#"eval "$(psychic zsh)""#;

/// Written above [`EVAL_LINE`], so someone reading their rc file knows where it
/// came from and how to undo it.
const MARKER_LINE: &str = "# psychic shell integration (undo with `psychic init --reverse`)";

/// What `p`, `pd` and `pc` need, said in one title line.
const HINT: &str = "No shell integration (p, pd, pc): run `psychic init`";

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

/// A line of the rc file, numbered from 1, as it is shown to the user.
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
    /// The file's new contents, without [`MARKER_LINE`] and [`EVAL_LINE`].
    new_contents: String,
    /// How many lines that took out.
    removed: usize,
    /// Lines that mention `psychic zsh` in some other form. Not ours to guess
    /// at, so they are left in and shown to the user.
    left_alone: Vec<RcLine>,
}

/// Does this line load the integration, in any form?
///
/// Deliberately loose - `eval "$(psychic zsh)"`, `source <(psychic zsh)`, a
/// full path to the binary - so that `init` never adds a second copy next to
/// one the user wrote themselves. Comments do not count.
fn loads_integration(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.starts_with('#') && trimmed.contains("psychic zsh")
}

fn plan_add(rc: &str) -> AddPlan {
    for (index, line) in rc.lines().enumerate() {
        if loads_integration(line) {
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
    AddPlan::Append(format!("{separator}{MARKER_LINE}\n{EVAL_LINE}\n"))
}

fn plan_remove(rc: &str) -> RemovePlan {
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
        if trimmed == EVAL_LINE {
            removed += 1;
            continue;
        }
        if loads_integration(line) {
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

/// `~/.zshrc`, or `$ZDOTDIR/.zshrc` when zsh has been told to look elsewhere.
fn zshrc_path() -> Result<PathBuf> {
    let dir = match std::env::var_os("ZDOTDIR") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?),
    };
    Ok(dir.join(".zshrc"))
}

/// `~` in place of the home directory, for messages.
fn tilde(path: &Path) -> String {
    match std::env::var_os("HOME").map(PathBuf::from) {
        Some(home) => match path.strip_prefix(&home) {
            Ok(rest) => format!("~/{}", rest.display()),
            Err(_) => path.display().to_string(),
        },
        None => path.display().to_string(),
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

/// `psychic init [--reverse] [--yes]`.
pub fn run(reverse: bool, yes: bool, data_dir: &Path) -> Result<()> {
    let rc_path = zshrc_path()?;
    if !yes && !std::io::stdin().is_terminal() {
        bail!("stdin is not a terminal, so psychic can't ask before editing; pass --yes");
    }
    if reverse {
        remove(&rc_path, yes, data_dir)
    } else {
        add(&rc_path, yes)
    }
}

fn add(rc_path: &Path, yes: bool) -> Result<()> {
    let shell = std::env::var("SHELL").unwrap_or_default();
    if Path::new(&shell).file_name() != Some(std::ffi::OsStr::new("zsh")) {
        bail!(
            "psychic's shell integration is only for zsh so far, and $SHELL is {:?}",
            shell
        );
    }
    let rc = match std::fs::read_to_string(rc_path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(error).with_context(|| format!("reading {}", rc_path.display()));
        }
    };
    let shown = tilde(rc_path);
    match plan_add(&rc) {
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
            // Appending through the path follows a symlink, so a dotfile
            // manager's link stays a link.
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(rc_path)
                .with_context(|| format!("opening {}", rc_path.display()))?;
            file.write_all(text.as_bytes())
                .with_context(|| format!("writing {}", rc_path.display()))?;
            println!("Done. Open a new shell, or run `exec zsh`, then try `p`.");
            Ok(())
        }
    }
}

fn remove(rc_path: &Path, yes: bool, data_dir: &Path) -> Result<()> {
    let shown = tilde(rc_path);
    let rc = match std::fs::read_to_string(rc_path) {
        Ok(text) => Some(text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error).with_context(|| format!("reading {}", rc_path.display()));
        }
    };
    let plan = rc.as_deref().map(plan_remove);
    match &plan {
        None => println!("There is no {shown}, so there is nothing to remove."),
        Some(plan) if plan.removed == 0 => {
            println!("{shown} has no lines that `psychic init` added.");
        }
        Some(plan) => {
            println!(
                "psychic will remove {} line(s) from {shown}:\n",
                plan.removed
            );
            println!("    {MARKER_LINE}\n    {EVAL_LINE}\n");
            if !yes && !confirm(&format!("Remove them from {shown}?"))? {
                println!("Nothing changed.");
                return Ok(());
            }
            // Written through to the real file, so a symlinked rc file stays
            // a symlink rather than being replaced by a copy.
            let target = std::fs::canonicalize(rc_path)
                .with_context(|| format!("resolving {}", rc_path.display()))?;
            std::fs::write(&target, &plan.new_contents)
                .with_context(|| format!("writing {}", target.display()))?;
            println!("Removed. Shells already open keep p, pd and pc until you open a new one.");
        }
    }
    if let Some(plan) = &plan {
        for line in &plan.left_alone {
            println!(
                "Not removed, because psychic did not write it: {shown} line {}: {}",
                line.number,
                line.text.trim()
            );
        }
    }

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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADDED: &str = "\
# psychic shell integration (undo with `psychic init --reverse`)
eval \"$(psychic zsh)\"
";

    #[test]
    fn add_to_an_empty_file() {
        assert_eq!(plan_add(""), AddPlan::Append(ADDED.to_string()));
    }

    #[test]
    fn add_leaves_a_blank_line_after_existing_contents() {
        assert_eq!(
            plan_add("export EDITOR=vim\n"),
            AddPlan::Append(format!("\n{ADDED}"))
        );
    }

    #[test]
    fn add_finishes_an_unterminated_last_line_first() {
        assert_eq!(
            plan_add("export EDITOR=vim"),
            AddPlan::Append(format!("\n\n{ADDED}"))
        );
    }

    #[test]
    fn add_is_a_no_op_when_any_form_is_present() {
        assert_eq!(
            plan_add("export A=1\n  source <(/opt/bin/psychic zsh)\n"),
            AddPlan::AlreadyPresent(RcLine {
                number: 2,
                text: "  source <(/opt/bin/psychic zsh)".to_string(),
            })
        );
    }

    #[test]
    fn add_ignores_a_commented_out_line() {
        assert_eq!(
            plan_add("# eval \"$(psychic zsh)\"\n"),
            AddPlan::Append(format!("\n{ADDED}"))
        );
    }

    #[test]
    fn add_then_remove_gives_back_the_original() {
        let original = "export EDITOR=vim\nalias ll='ls -l'\n";
        let AddPlan::Append(added) = plan_add(original) else {
            panic!("expected an append");
        };
        let after_add = format!("{original}{added}");
        let plan = plan_remove(&after_add);
        assert_eq!(
            plan,
            RemovePlan {
                new_contents: original.to_string(),
                removed: 2,
                left_alone: vec![],
            }
        );
    }

    #[test]
    fn add_then_remove_on_an_unterminated_file_only_adds_its_newline() {
        let AddPlan::Append(added) = plan_add("export EDITOR=vim") else {
            panic!("expected an append");
        };
        assert_eq!(
            plan_remove(&format!("export EDITOR=vim{added}")),
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
            plan_remove("a\n\neval \"$(psychic zsh)\"\n\nb\n"),
            RemovePlan {
                new_contents: "a\n\n\nb\n".to_string(),
                removed: 1,
                left_alone: vec![],
            }
        );
    }

    #[test]
    fn remove_takes_a_hand_written_exact_line_too() {
        assert_eq!(
            plan_remove("a\n  eval \"$(psychic zsh)\"\nb\n"),
            RemovePlan {
                new_contents: "a\nb\n".to_string(),
                removed: 1,
                left_alone: vec![],
            }
        );
    }

    #[test]
    fn remove_leaves_other_forms_and_reports_them() {
        assert_eq!(
            plan_remove("a\nsource <(psychic zsh)\n"),
            RemovePlan {
                new_contents: "a\nsource <(psychic zsh)\n".to_string(),
                removed: 0,
                left_alone: vec![RcLine {
                    number: 2,
                    text: "source <(psychic zsh)".to_string(),
                }],
            }
        );
    }

    #[test]
    fn remove_from_a_file_that_is_only_the_block() {
        assert_eq!(
            plan_remove(ADDED),
            RemovePlan {
                new_contents: String::new(),
                removed: 2,
                left_alone: vec![],
            }
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
