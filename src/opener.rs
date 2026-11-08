//! What Enter on a file does: `$EDITOR` for anything that is text, the
//! system's own opener (`open` on macOS, `xdg-open` elsewhere) for PDFs,
//! images, media and office documents, which an editor can only show as bytes.
//!
//! [`choose`] decides from the extension and the environment, and is pure;
//! [`open_in_app`] runs the opener without giving it the terminal, so the TUI
//! stays up while the PDF opens beside it.

use anyhow::{Context, Result};
use std::path::Path;

/// Extensions an editor is the wrong tool for. Lowercase. Formats people do
/// edit as text even though an app could show them - SVG, HTML, CSV,
/// Markdown - are left out on purpose.
const APP_EXTENSIONS: &[&str] = &[
    // Documents
    "pdf", "epub", "djvu", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods", "odp",
    "pages", "numbers", "key", "rtf", // Images
    "png", "jpg", "jpeg", "gif", "webp", "heic", "heif", "tif", "tiff", "bmp", "ico", "psd",
    // Audio and video
    "mp3", "m4a", "wav", "flac", "ogg", "opus", "aac", "mp4", "m4v", "mov", "mkv", "avi", "webm",
];

/// How a file gets opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opener {
    /// Hand the terminal to `$EDITOR`.
    Editor,
    /// Run this command on the file, in the background.
    App(&'static str),
}

/// What [`choose`] needs to know about where psychic is running.
#[derive(Debug, Clone, Copy)]
pub struct Environment {
    pub macos: bool,
    /// Over SSH, `open` would show the file on the remote machine's screen,
    /// not the user's.
    pub ssh: bool,
    /// `DISPLAY` or `WAYLAND_DISPLAY` is set: there is a desktop to open into.
    pub display: bool,
}

impl Environment {
    pub fn current() -> Self {
        let set = |name: &str| std::env::var_os(name).is_some_and(|value| !value.is_empty());
        Environment {
            macos: cfg!(target_os = "macos"),
            ssh: set("SSH_CONNECTION") || set("SSH_TTY"),
            display: set("DISPLAY") || set("WAYLAND_DISPLAY"),
        }
    }
}

pub fn choose(path: &Path, environment: Environment) -> Opener {
    let wants_app = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| APP_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()));
    if !wants_app {
        return Opener::Editor;
    }
    match (environment.macos, environment.ssh, environment.display) {
        (true, false, _) => Opener::App("open"),
        // X forwarding over SSH sets DISPLAY, and then xdg-open does reach the
        // user's screen.
        (false, _, true) => Opener::App("xdg-open"),
        // Nowhere to show it: the editor at least shows something.
        (true, true, _) | (false, _, false) => Opener::Editor,
    }
}

/// Run `command path` detached from the terminal, and reap it on a thread so
/// it does not linger as a zombie. Returns once it has started.
pub fn open_in_app(command: &'static str, path: &Path) -> Result<()> {
    let mut child = std::process::Command::new(command)
        .arg(path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("running {command}"))?;
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAC: Environment = Environment {
        macos: true,
        ssh: false,
        display: false,
    };
    const MAC_OVER_SSH: Environment = Environment {
        macos: true,
        ssh: true,
        display: false,
    };
    const LINUX_DESKTOP: Environment = Environment {
        macos: false,
        ssh: false,
        display: true,
    };
    const LINUX_HEADLESS: Environment = Environment {
        macos: false,
        ssh: true,
        display: false,
    };

    #[test]
    fn by_extension_and_environment() {
        let cases = ["paper.PDF", "photo.jpeg", "main.rs", "logo.svg", "Makefile"];
        let chosen =
            |environment: Environment| cases.map(|name| choose(Path::new(name), environment));
        assert_eq!(
            chosen(MAC),
            [
                Opener::App("open"),
                Opener::App("open"),
                Opener::Editor,
                Opener::Editor,
                Opener::Editor
            ]
        );
        assert_eq!(
            chosen(LINUX_DESKTOP),
            [
                Opener::App("xdg-open"),
                Opener::App("xdg-open"),
                Opener::Editor,
                Opener::Editor,
                Opener::Editor
            ]
        );
        assert_eq!(chosen(MAC_OVER_SSH), [Opener::Editor; 5]);
        assert_eq!(chosen(LINUX_HEADLESS), [Opener::Editor; 5]);
    }
}
