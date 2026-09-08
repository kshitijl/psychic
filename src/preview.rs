//! The preview pane: what is in the thing you have selected.
//!
//! Two jobs, kept apart. [`spawn`] starts a thread that turns a path into
//! styled text, and [`PreviewState`] is the UI's side of that conversation: it
//! remembers what is on screen, what has been asked for, and where the pane is
//! scrolled to.
//!
//! ## Why a thread
//!
//! This used to run inside `terminal.draw`, so every frame that changed the
//! selection paid for it before anything could be painted. Holding Down meant
//! one preview per row, in the way of the redraw each time. Generation is much
//! cheaper now, but "cheap" is not "bounded": a file on a slow mount, or a
//! directory with a hundred thousand entries, still takes as long as it takes,
//! and the UI must not be waiting on it.
//!
//! The thread keeps only the newest request. Scrolling a list quickly enqueues
//! a request per row, and all but the last describe a selection the user has
//! already left.
//!
//! ## Why not `bat` and `eza`
//!
//! It used to shell out to both, which cost a process spawn (12-16ms measured)
//! per preview, and then parsed the ANSI they printed back into ratatui spans.
//! It also meant psychic silently degraded to `ls` and unhighlighted text on a
//! machine that did not have them installed. Now the highlighting is `syntect`
//! in process, and the directory listing is a `read_dir`, so styles are built
//! directly and there is nothing to install.

use anyhow::Result;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

use crate::path_display::{human_bytes, printable};

/// Most lines of a file to highlight.
///
/// The pane shows a few dozen at a time, but the whole thing is generated at
/// once so that scrolling costs nothing. This is what stops that from being
/// unbounded: it is a preview, not a pager.
const MAX_LINES: usize = 5_000;

/// Most bytes to read from a file, whatever its line count.
const MAX_BYTES: usize = 4 << 20;

/// How much of a file to look at when deciding whether it is text.
const SNIFF_BYTES: usize = 8 << 10;

/// Below this width the listing drops its permissions and date columns.
const NARROW: u16 = 80;

/// A request for the preview of one path.
pub struct PreviewRequest {
    pub id: u64,
    pub path: PathBuf,
    pub is_dir: bool,
    pub width: u16,
}

/// A generated preview, on its way back to the UI.
pub struct Preview {
    pub id: u64,
    pub text: Text<'static>,
}

/// Start the preview thread.
///
/// The returned sender is how the UI asks for previews; replies arrive on
/// `event_tx` like any other event.
pub fn spawn<T>(event_tx: Sender<T>) -> Sender<PreviewRequest>
where
    T: From<Preview> + Send + 'static,
{
    let (request_tx, request_rx) = std::sync::mpsc::channel::<PreviewRequest>();

    std::thread::spawn(move || {
        // Deserializing the syntax definitions takes long enough to be worth
        // measuring, and is why the highlighter is built once here rather than
        // per file. It happens while the walker is still running, so it is off
        // everyone's critical path.
        let start = std::time::Instant::now();
        let highlighter = Highlighter::new();
        log::info!(
            "TIMING {{\"op\":\"syntax_set_load\",\"ms\":{}}}",
            start.elapsed().as_secs_f64() * 1000.0
        );

        serve(&highlighter, request_rx, event_tx);
        log::debug!("Preview thread exiting");
    });

    request_tx
}

fn serve<T>(highlighter: &Highlighter, request_rx: Receiver<PreviewRequest>, event_tx: Sender<T>)
where
    T: From<Preview>,
{
    while let Ok(request) = request_rx.recv() {
        // Everything but the newest request describes a row the user has
        // already scrolled past.
        let mut request = request;
        while let Ok(newer) = request_rx.try_recv() {
            request = newer;
        }

        let start = std::time::Instant::now();
        let text = if request.is_dir {
            directory(&request.path, request.width)
        } else {
            highlighter.file(&request.path)
        };
        log::info!(
            "TIMING {{\"op\":\"preview_generate\",\"ms\":{},\"dir\":{}}}",
            start.elapsed().as_secs_f64() * 1000.0,
            request.is_dir
        );

        let preview = Preview {
            id: request.id,
            text,
        };
        if event_tx.send(preview.into()).is_err() {
            return;
        }
    }
}

/// The UI's half: what is shown, what has been asked for, and the scroll offset.
pub struct PreviewState {
    /// The path whose preview is in `text`, and the text itself.
    showing: Option<(PathBuf, Text<'static>)>,
    /// The request we are waiting on, if any.
    pending: Option<(u64, PathBuf)>,
    next_id: u64,
    scroll: usize,
    request_tx: Sender<PreviewRequest>,
}

impl PreviewState {
    pub fn new(request_tx: Sender<PreviewRequest>) -> Self {
        Self {
            showing: None,
            pending: None,
            next_id: 0,
            scroll: 0,
            request_tx,
        }
    }

    /// The preview to draw for `path`, if we have it.
    ///
    /// `None` means "nothing to show yet", not "nothing to show": the pane is
    /// left empty for the frame or two before the answer arrives.
    pub fn text_for(&self, path: &Path) -> Option<&Text<'static>> {
        match &self.showing {
            Some((shown, text)) if shown == path => Some(text),
            _ => None,
        }
    }

    /// The lines of `path`'s preview that fit the pane, from the scroll offset.
    ///
    /// Slicing here rather than handing the whole thing to `Paragraph::scroll`
    /// is what keeps a five thousand line preview from being cloned on every
    /// frame: only what is on screen is copied.
    pub fn visible(&self, path: &Path, height: u16) -> Text<'static> {
        let Some(text) = self.text_for(path) else {
            return Text::default();
        };

        let first = self.scroll.min(text.lines.len().saturating_sub(1));
        let last = first.saturating_add(height as usize).min(text.lines.len());

        Text::from(text.lines[first..last].to_vec())
    }

    /// Ask for `path`'s preview unless we already have it or asked for it.
    ///
    /// Called after each frame, because that is when the pane's size is known.
    pub fn request(&mut self, path: &Path, is_dir: bool, width: u16) {
        let already_showing = matches!(&self.showing, Some((shown, _)) if shown == path);
        let already_asked = matches!(&self.pending, Some((_, asked)) if asked == path);
        if already_showing || already_asked {
            return;
        }

        self.next_id += 1;
        let id = self.next_id;
        self.pending = Some((id, path.to_path_buf()));

        let _ = self.request_tx.send(PreviewRequest {
            id,
            path: path.to_path_buf(),
            is_dir,
            width,
        });
    }

    /// Take a generated preview, if it is still the one we are waiting for.
    pub fn ready(&mut self, preview: Preview) {
        let Some((pending_id, path)) = self.pending.take() else {
            return;
        };

        if pending_id != preview.id {
            // A newer request has already replaced this one.
            self.pending = Some((pending_id, path));
            return;
        }

        self.showing = Some((path, preview.text));
    }

    /// Forget what is on screen, because the selection moved.
    pub fn clear(&mut self) {
        self.showing = None;
        self.pending = None;
        self.scroll = 0;
    }

    /// Move the pane, clamped to the preview we actually have.
    ///
    /// Clamping matters at the bottom, not just the top: without it, scrolling
    /// past the end walks the offset off into the distance and the user has to
    /// scroll back the same distance before the view moves at all.
    pub fn scroll(&mut self, delta: isize) {
        let last = self
            .showing
            .as_ref()
            .map(|(_, text)| text.lines.len().saturating_sub(1))
            .unwrap_or(0);

        self.scroll = (self.scroll as isize + delta).clamp(0, last as isize) as usize;
    }

    /// One word for the debug pane.
    pub fn status(&self, path: &Path) -> &'static str {
        if self.text_for(path).is_some() {
            "ready"
        } else if self.pending.is_some() {
            "generating"
        } else {
            "none"
        }
    }
}

/// Syntax definitions and a theme, loaded once.
struct Highlighter {
    syntaxes: SyntaxSet,
    theme: Theme,
}

impl Highlighter {
    fn new() -> Self {
        let mut themes = ThemeSet::load_defaults();
        let theme = themes
            .themes
            .remove("base16-ocean.dark")
            .or_else(|| themes.themes.values().next().cloned())
            .expect("syntect ships with themes");

        Self {
            syntaxes: SyntaxSet::load_defaults_newlines(),
            theme,
        }
    }

    /// A file, highlighted, with line numbers.
    fn file(&self, path: &Path) -> Text<'static> {
        let content = match read_text(path) {
            Ok(Text_::Text(content)) => content,
            Ok(Text_::Binary) => return note("[binary file]"),
            Ok(Text_::Empty) => return note("[empty file]"),
            Err(e) => return note(&format!("[unable to read: {}]", e)),
        };

        let syntax = self
            .syntaxes
            .find_syntax_for_file(path)
            .ok()
            .flatten()
            .or_else(|| {
                self.syntaxes
                    .find_syntax_by_first_line(content.lines().next().unwrap_or(""))
            })
            .unwrap_or_else(|| self.syntaxes.find_syntax_plain_text());

        let mut highlighting = HighlightLines::new(syntax, &self.theme);
        let width = line_number_width(&content);

        let lines = LinesWithEndings::from(&content)
            .take(MAX_LINES)
            .enumerate()
            .map(|(i, line)| {
                let mut spans = vec![Span::styled(
                    format!("{:>width$} ", i + 1, width = width),
                    Style::default().fg(Color::DarkGray),
                )];

                match highlighting.highlight_line(line, &self.syntaxes) {
                    Ok(ranges) => spans.extend(ranges.into_iter().map(|(style, text)| {
                        // Foreground only: the theme's background would paint
                        // over the terminal's own, which is the one the user
                        // chose and the rest of psychic honours.
                        Span::styled(
                            text.trim_end_matches(['\n', '\r']).to_string(),
                            Style::default().fg(Color::Rgb(
                                style.foreground.r,
                                style.foreground.g,
                                style.foreground.b,
                            )),
                        )
                    })),
                    Err(_) => {
                        spans.push(Span::raw(line.trim_end_matches(['\n', '\r']).to_string()))
                    }
                }

                Line::from(spans)
            })
            .collect::<Vec<_>>();

        Text::from(lines)
    }
}

/// What reading a file for preview found.
enum Text_ {
    Text(String),
    Binary,
    Empty,
}

/// Read at most [`MAX_BYTES`], and decide whether it is worth showing.
fn read_text(path: &Path) -> Result<Text_, std::io::Error> {
    use std::io::Read;

    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_BYTES as u64).read_to_end(&mut bytes)?;

    if bytes.is_empty() {
        return Ok(Text_::Empty);
    }

    // A NUL byte early on is the same test `grep` uses, and it is what stopped
    // a scrolled-past binary from painting garbage over the pane.
    if bytes[..bytes.len().min(SNIFF_BYTES)].contains(&0) {
        return Ok(Text_::Binary);
    }

    // Sanitised here, once, so that nothing downstream has to remember to.
    // The NUL check above only sniffs the start of the file; this is what makes
    // a file that turns to rubbish halfway through render as dots instead of
    // driving the terminal.
    //
    // Line by line, because a newline is a control character too: it is the one
    // that has a meaning here, and `printable` is otherwise right to replace it
    // (a filename containing one must not become two rows). CRLF is normalised
    // first so that Windows line endings do not each leave a dot behind.
    let raw = String::from_utf8_lossy(&bytes).replace("\r\n", "\n");
    let text = raw
        .split('\n')
        .map(|line| printable(line))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(Text_::Text(text))
}

/// Room needed for the largest line number we will print.
fn line_number_width(content: &str) -> usize {
    let lines = content.lines().count().clamp(1, MAX_LINES);
    lines.to_string().len()
}

/// A one-line message in place of a preview.
fn note(message: &str) -> Text<'static> {
    Text::from(Line::from(Span::styled(
        message.to_string(),
        Style::default().fg(Color::DarkGray),
    )))
}

/// A directory listing: permissions, size, date, name.
///
/// The columns after the name are dropped on a narrow pane, which is what the
/// `--no-permissions --no-user --no-time` flags used to do for `eza`.
fn directory(path: &Path, width: u16) -> Text<'static> {
    let mut entries: Vec<Entry> = match std::fs::read_dir(path) {
        Ok(dir) => dir.flatten().map(Entry::of).collect(),
        Err(e) => return note(&format!("[unable to list: {}]", e)),
    };

    if entries.is_empty() {
        return note("[empty directory]");
    }

    entries.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.name.cmp(&b.name))
    });

    let detailed = width >= NARROW;
    let size_width = entries.iter().map(|e| e.size.len()).max().unwrap_or(0);

    let lines = entries
        .iter()
        .map(|entry| {
            let mut spans = Vec::new();

            if detailed {
                spans.push(Span::styled(
                    format!("{} ", entry.permissions),
                    Style::default().fg(Color::DarkGray),
                ));
            }

            spans.push(Span::styled(
                format!("{:>width$} ", entry.size, width = size_width),
                Style::default().fg(Color::DarkGray),
            ));

            if detailed {
                spans.push(Span::styled(
                    format!("{} ", entry.modified),
                    Style::default().fg(Color::DarkGray),
                ));
            }

            spans.push(Span::styled(entry.name.clone(), entry.style()));

            Line::from(spans)
        })
        .collect::<Vec<_>>();

    Text::from(lines)
}

/// One row of a directory listing.
struct Entry {
    name: String,
    permissions: String,
    size: String,
    modified: String,
    is_dir: bool,
    is_symlink: bool,
    is_executable: bool,
}

impl Entry {
    fn of(entry: std::fs::DirEntry) -> Self {
        // The link itself, not its target: a listing should say what is here.
        let metadata = entry.path().symlink_metadata().ok();
        let is_symlink = metadata
            .as_ref()
            .is_some_and(|m| m.file_type().is_symlink());
        let is_dir = match &metadata {
            // A symlink to a directory still reads as a directory here, which
            // is what the user cares about when navigating.
            Some(m) if m.file_type().is_symlink() => entry.path().is_dir(),
            Some(m) => m.is_dir(),
            None => false,
        };

        let mode = mode_of(metadata.as_ref());
        let mut name = printable(&entry.file_name().to_string_lossy()).into_owned();
        if is_dir {
            name.push('/');
        }

        Self {
            name,
            permissions: permissions(mode, is_dir, is_symlink),
            size: match &metadata {
                Some(m) if !is_dir => human_bytes(m.len()),
                _ => "-".to_string(),
            },
            modified: metadata
                .as_ref()
                .and_then(|m| m.modified().ok())
                .map(format_time)
                .unwrap_or_else(|| " ".repeat(12)),
            is_dir,
            is_symlink,
            is_executable: mode & 0o111 != 0,
        }
    }

    /// Same colours the file list uses, so the two panes agree.
    fn style(&self) -> Style {
        if self.is_symlink {
            Style::default().fg(Color::Magenta)
        } else if self.is_dir {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else if self.is_executable {
            Style::default().fg(Color::Green)
        } else {
            Style::default()
        }
    }
}

fn mode_of(metadata: Option<&std::fs::Metadata>) -> u32 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.map(|m| m.permissions().mode()).unwrap_or(0)
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        0
    }
}

/// `drwxr-xr-x`, as `ls` writes it.
fn permissions(mode: u32, is_dir: bool, is_symlink: bool) -> String {
    let kind = if is_symlink {
        'l'
    } else if is_dir {
        'd'
    } else {
        '.'
    };

    let mut out = String::with_capacity(10);
    out.push(kind);
    for shift in [6, 3, 0] {
        let bits = (mode >> shift) & 0o7;
        out.push(if bits & 0o4 != 0 { 'r' } else { '-' });
        out.push(if bits & 0o2 != 0 { 'w' } else { '-' });
        out.push(if bits & 0o1 != 0 { 'x' } else { '-' });
    }
    out
}

/// `2026-09-09 14:22`, in the local zone.
fn format_time(time: std::time::SystemTime) -> String {
    let Ok(since_epoch) = time.duration_since(std::time::UNIX_EPOCH) else {
        return " ".repeat(16);
    };

    match jiff::Timestamp::from_second(since_epoch.as_secs() as i64) {
        Ok(timestamp) => timestamp
            .to_zoned(jiff::tz::TimeZone::system())
            .strftime("%Y-%m-%d %H:%M")
            .to_string(),
        Err(_) => " ".repeat(16),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The visible text of a preview, one string per line.
    fn rendered(text: &Text<'static>) -> Vec<String> {
        text.lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "psychic-preview-{}-{}",
                name,
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        fn write(&self, name: &str, contents: &[u8]) -> PathBuf {
            let path = self.path.join(name);
            std::fs::write(&path, contents).expect("write file");
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn test_permissions_read_like_ls() {
        assert_eq!(permissions(0o755, true, false), "drwxr-xr-x");
        assert_eq!(permissions(0o644, false, false), ".rw-r--r--");
        assert_eq!(permissions(0o600, false, false), ".rw-------");
        assert_eq!(permissions(0o777, false, true), "lrwxrwxrwx");
    }

    #[test]
    fn test_a_binary_file_is_named_not_painted() {
        let dir = TempDir::new("binary");
        let path = dir.write("thing.bin", &[0x7f, 0x45, 0x4c, 0x46, 0x00, 0x01, 0x02]);

        assert_eq!(
            rendered(&Highlighter::new().file(&path)),
            vec!["[binary file]"],
            "Painting the bytes is what used to corrupt the pane"
        );
    }

    #[test]
    fn test_an_empty_file_says_so() {
        let dir = TempDir::new("empty");
        let path = dir.write("nothing.txt", b"");

        assert_eq!(
            rendered(&Highlighter::new().file(&path)),
            vec!["[empty file]"]
        );
    }

    #[test]
    fn test_a_missing_file_does_not_panic() {
        let text = Highlighter::new().file(Path::new("/definitely/not/here.txt"));
        assert!(
            rendered(&text)[0].starts_with("[unable to read"),
            "got {:?}",
            rendered(&text)
        );
    }

    #[test]
    fn test_a_file_is_numbered_and_kept_intact() {
        let dir = TempDir::new("numbered");
        let path = dir.write("hello.rs", b"fn main() {\n    let x = 1;\n}\n");

        assert_eq!(
            rendered(&Highlighter::new().file(&path)),
            vec!["1 fn main() {", "2     let x = 1;", "3 }"],
            "Line numbers, and the source unchanged beside them"
        );
    }

    #[test]
    fn test_highlighting_actually_colours_something() {
        let dir = TempDir::new("colours");
        let path = dir.write("hello.rs", b"fn main() {}\n");

        let text = Highlighter::new().file(&path);
        let colours: Vec<_> = text.lines[0]
            .spans
            .iter()
            .filter_map(|s| s.style.fg)
            .collect();

        assert!(
            colours.len() > 2,
            "`fn main() {{}}` should be more than one colour, got {:?}",
            colours
        );
    }

    /// Nothing a file contains may reach the terminal as an instruction.
    fn assert_nothing_escapes(text: &Text<'static>) {
        for line in rendered(text) {
            assert!(
                !line.chars().any(|c| c.is_control()),
                "a control character reached a cell: {:?}",
                line
            );
        }
    }

    #[test]
    fn test_escape_sequences_in_a_text_file_cannot_drive_the_terminal() {
        let dir = TempDir::new("escapes");
        // A captured terminal session, a log with colour, a hostile filename in
        // a text file: all ordinary, all full of ESC.
        let path = dir.write(
            "session.log",
            b"plain\n\x1b[31mred\x1b[0m\n\x1b]0;retitled\x07\ndone\n",
        );

        let text = Highlighter::new().file(&path);
        assert_nothing_escapes(&text);

        let lines = rendered(&text);
        assert!(
            lines[1].contains("[31mred"),
            "the text survives: {:?}",
            lines
        );
        assert!(lines[1].starts_with("2 ·"), "the ESC does not: {:?}", lines);
    }

    #[test]
    fn test_a_file_that_turns_binary_after_the_sniff_still_cannot_corrupt() {
        let dir = TempDir::new("late-binary");
        // Text for longer than we sniff, then rubbish. The "[binary file]"
        // shortcut does not catch this one, so sanitising has to.
        let mut content = b"harmless text\n".repeat(SNIFF_BYTES / 14 + 10);
        content.extend_from_slice(&[0x1b, 0x5b, 0x32, 0x4a, 0x07, 0x08, 0x00, 0xff]);
        let path = dir.write("mixed.txt", &content);

        assert_nothing_escapes(&Highlighter::new().file(&path));
    }

    #[test]
    fn test_windows_line_endings_do_not_leave_a_mark() {
        let dir = TempDir::new("crlf");
        let path = dir.write("dos.txt", b"first\r\nsecond\r\n");

        assert_eq!(
            rendered(&Highlighter::new().file(&path)),
            vec!["1 first", "2 second"],
            "A carriage return is a control character, but CRLF is just a newline"
        );
    }

    #[test]
    fn test_tabs_are_expanded_so_the_layout_can_count_them() {
        let dir = TempDir::new("tabs");
        let path = dir.write("indented.txt", b"a\tb\n");

        assert_eq!(rendered(&Highlighter::new().file(&path)), vec!["1 a    b"]);
    }

    #[test]
    fn test_a_filename_full_of_escapes_cannot_drive_the_terminal() {
        let dir = TempDir::new("bad-name");
        // Filenames are whatever someone managed to create, and they are drawn
        // into cells like anything else.
        if std::fs::write(dir.path.join("we\x1b[2Jird.txt"), b"x").is_err() {
            return; // some filesystems refuse; nothing to check then
        }

        for line in rendered(&directory(&dir.path, 120)) {
            assert!(
                !line.chars().any(|c| c.is_control()),
                "a control character reached a cell: {:?}",
                line
            );
        }
    }

    #[test]
    fn test_a_very_long_file_is_cut_off() {
        let dir = TempDir::new("long");
        let content = "x = 1\n".repeat(MAX_LINES + 500);
        let path = dir.write("long.py", content.as_bytes());

        assert_eq!(
            Highlighter::new().file(&path).lines.len(),
            MAX_LINES,
            "A preview is not a pager"
        );
    }

    #[test]
    fn test_a_directory_lists_its_contents_sorted() {
        let dir = TempDir::new("listing");
        dir.write("beta.txt", b"hello");
        dir.write("Alpha.txt", b"hi");
        std::fs::create_dir(dir.path.join("sub")).expect("create subdir");

        let names: Vec<String> = rendered(&directory(&dir.path, 120))
            .iter()
            .map(|line| line.split_whitespace().last().unwrap_or("").to_string())
            .collect();

        assert_eq!(
            names,
            vec!["Alpha.txt", "beta.txt", "sub/"],
            "Case-insensitive order, directories marked with a slash"
        );
    }

    #[test]
    fn test_a_narrow_pane_drops_the_wide_columns() {
        let dir = TempDir::new("narrow");
        dir.write("f.txt", b"hello");

        let wide = rendered(&directory(&dir.path, 120)).remove(0);
        let narrow = rendered(&directory(&dir.path, NARROW - 1)).remove(0);

        assert!(wide.contains(".rw-"), "wide keeps permissions: {:?}", wide);
        assert!(!narrow.contains(".rw-"), "narrow drops them: {:?}", narrow);
        assert!(narrow.contains("f.txt"), "but keeps the name: {:?}", narrow);
    }

    #[test]
    fn test_an_empty_directory_says_so() {
        let dir = TempDir::new("empty-dir");

        assert_eq!(
            rendered(&directory(&dir.path, 120)),
            vec!["[empty directory]"]
        );
    }

    #[test]
    fn test_an_unreadable_directory_does_not_panic() {
        let text = directory(Path::new("/definitely/not/here"), 120);
        assert!(rendered(&text)[0].starts_with("[unable to list"));
    }
}
