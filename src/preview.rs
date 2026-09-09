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
use syntect::highlighting::Theme;
use syntect::parsing::SyntaxSet;

use crate::path_display::{human_bytes, printable};

/// Most lines of a file to highlight, however far the user scrolls.
///
/// A preview is not a pager.
const MAX_LINES: usize = 5_000;

/// Most bytes to read from a file, whatever its line count.
const MAX_BYTES: usize = 4 << 20;

/// How much of a file to look at when deciding whether it is text.
const SNIFF_BYTES: usize = 8 << 10;

/// Below this width the listing drops its permissions and date columns.
const NARROW: u16 = 80;

/// How much room the preview pane has.
///
/// The width decides a listing's columns; the height decides how much of a file
/// is worth highlighting before the user has scrolled.
#[derive(Debug, Clone, Copy)]
pub struct PreviewPane {
    pub width: u16,
    pub height: u16,
}

/// A request for the preview of one path.
pub struct PreviewRequest {
    pub id: u64,
    pub path: PathBuf,
    pub is_dir: bool,
    pub width: u16,
    /// How many lines from the top are wanted.
    ///
    /// Only a screenful is on show, and highlighting is the expensive part -
    /// markdown at 150ms for five thousand lines, against about two for a
    /// screenful. Syntect carries state from line to line, so the count is
    /// always from the top of the file; scrolling asks for more.
    pub lines: usize,
}

/// A generated preview, on its way back to the UI.
pub struct Preview {
    pub id: u64,
    pub text: Text<'static>,
    /// Whether this is all there will ever be: the file ended, or hit the cap.
    pub complete: bool,
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
        let generator = Generator::new();
        log::info!(
            "TIMING {{\"op\":\"syntax_set_load\",\"ms\":{}}}",
            start.elapsed().as_secs_f64() * 1000.0
        );

        serve(&generator, request_rx, event_tx);
        log::debug!("Preview thread exiting");
    });

    request_tx
}

fn serve<T>(generator: &Generator, request_rx: Receiver<PreviewRequest>, event_tx: Sender<T>)
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
        let generated =
            generator.preview(&request.path, request.is_dir, request.width, request.lines);
        log::info!(
            "TIMING {{\"op\":\"preview_generate\",\"ms\":{},\"lines\":{},\"dir\":{}}}",
            start.elapsed().as_secs_f64() * 1000.0,
            generated.text.lines.len(),
            request.is_dir
        );

        let preview = Preview {
            id: request.id,
            text: generated.text,
            complete: generated.complete,
        };
        if event_tx.send(preview.into()).is_err() {
            return;
        }
    }
}

/// The UI's half: what is shown, what has been asked for, and the scroll offset.
pub struct PreviewState {
    showing: Option<Shown>,
    /// The request we are waiting on: its id, path, and line budget.
    pending: Option<(u64, PathBuf, usize)>,
    next_id: u64,
    scroll: usize,
    request_tx: Sender<PreviewRequest>,
}

/// What is on screen, and how much of the file it covers.
struct Shown {
    path: PathBuf,
    text: Text<'static>,
    /// True when there is no more of this file to ask for.
    complete: bool,
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
            Some(shown) if shown.path == path => Some(&shown.text),
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

    /// Ask for `path`'s preview, if we do not already have enough of it.
    ///
    /// Called after each frame, because that is when the pane's size is known.
    ///
    /// Two states, not a sliding window. Unscrolled, a screenful is generated
    /// and nothing more, because that is all anyone can see and highlighting is
    /// the expensive part. The moment the user scrolls, the rest is generated
    /// in one go and scrolling is free from then on.
    ///
    /// The obvious-looking alternative - grow the budget as the user scrolls -
    /// is worse. Syntect carries state from line to line, so every pass has to
    /// start at line one; a budget that grows by steps re-highlights the whole
    /// preamble each time, costing about twice the total work and paying it in
    /// a series of visible hiccups rather than one.
    pub fn request(&mut self, path: &Path, is_dir: bool, pane: PreviewPane) {
        let wanted = if self.scroll == 0 {
            // The screen on show, and one in hand so that a first nudge of the
            // wheel does not have to wait for anything.
            (2 * pane.height as usize).clamp(1, MAX_LINES)
        } else {
            MAX_LINES
        };

        if let Some(shown) = &self.showing
            && shown.path == path
            && (shown.complete || shown.text.lines.len() >= wanted)
        {
            return;
        }
        if let Some((_, asked, asked_for)) = &self.pending
            && asked == path
            && *asked_for >= wanted
        {
            return;
        }

        self.next_id += 1;
        let id = self.next_id;
        self.pending = Some((id, path.to_path_buf(), wanted));

        let _ = self.request_tx.send(PreviewRequest {
            id,
            path: path.to_path_buf(),
            is_dir,
            width: pane.width,
            lines: wanted,
        });
    }

    /// Take a generated preview, if it is still the one we are waiting for.
    pub fn ready(&mut self, preview: Preview) {
        let Some((pending_id, path, lines)) = self.pending.take() else {
            return;
        };

        if pending_id != preview.id {
            // A newer request has already replaced this one.
            self.pending = Some((pending_id, path, lines));
            return;
        }

        self.showing = Some(Shown {
            path,
            text: preview.text,
            complete: preview.complete,
        });
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
        let limit = match &self.showing {
            // All of it: stop at the last line.
            Some(shown) if shown.complete => shown.text.lines.len().saturating_sub(1),
            // Part of it: a screen of runway past the end of what is generated,
            // so scrolling down asks for more instead of stopping at the edge.
            Some(shown) => shown.text.lines.len(),
            None => 0,
        };

        self.scroll = (self.scroll as isize + delta).clamp(0, limit as isize) as usize;
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

/// Everything needed to turn a path into styled text: syntax definitions and a
/// theme, loaded once because loading them is not free.
///
/// The preview thread owns one. `psychic internal preview` makes one directly,
/// so the generator can be timed with no UI, thread or channel in the way.
pub struct Generator {
    syntaxes: SyntaxSet,
    theme: Theme,
}

impl Default for Generator {
    fn default() -> Self {
        Self::new()
    }
}

impl Generator {
    pub fn new() -> Self {
        // bat's default. syntect's own themes are all present too, but this one
        // gives markdown headings and the like some weight, where
        // `base16-ocean.dark` renders them a grey barely distinct from body
        // text. The theme is also the reason previews only ever set a
        // foreground colour: its background would paint over the terminal's.
        let theme = two_face::theme::extra()
            .get(two_face::theme::EmbeddedThemeName::MonokaiExtended)
            .clone();

        Self {
            syntaxes: two_face::syntax::extra_newlines(),
            theme,
        }
    }

    /// A preview of `path`: a listing if it is a directory, otherwise the
    /// first `max_lines` of the file, highlighted.
    pub fn generate(
        &self,
        path: &Path,
        is_dir: bool,
        width: u16,
        max_lines: usize,
    ) -> Text<'static> {
        self.preview(path, is_dir, width, max_lines).text
    }

    fn preview(&self, path: &Path, is_dir: bool, width: u16, max_lines: usize) -> Generated {
        if is_dir {
            directory(path, width)
        } else {
            self.file(path, max_lines)
        }
    }

    /// The first `max_lines` of a file, highlighted, with line numbers.
    fn file(&self, path: &Path, max_lines: usize) -> Generated {
        let (lines, complete) = match read_text(path, max_lines) {
            Ok(Content::Text { lines, complete }) => (lines, complete),
            Ok(Content::Binary) => return Generated::note("[binary file]"),
            Ok(Content::Empty) => return Generated::note("[empty file]"),
            Err(e) => return Generated::note(&format!("[unable to read: {}]", e)),
        };

        let syntax = self
            .syntaxes
            .find_syntax_for_file(path)
            .ok()
            .flatten()
            .or_else(|| {
                self.syntaxes
                    .find_syntax_by_first_line(lines.first().map_or("", |l| l.as_str()))
            })
            .unwrap_or_else(|| self.syntaxes.find_syntax_plain_text());

        let mut highlighting = HighlightLines::new(syntax, &self.theme);
        let width = lines.len().max(1).to_string().len();

        let rendered = lines
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let mut spans = vec![Span::styled(
                    format!("{:>width$} ", i + 1, width = width),
                    Style::default().fg(Color::DarkGray),
                )];

                // Syntect wants the newline it was parsed with; the span must
                // not have one, or ratatui draws a cell for it.
                let with_ending = format!("{}\n", line);
                match highlighting.highlight_line(&with_ending, &self.syntaxes) {
                    Ok(ranges) => spans.extend(ranges.into_iter().filter_map(|(style, text)| {
                        let text = text.trim_end_matches('\n');
                        if text.is_empty() {
                            return None;
                        }
                        // Foreground only: the theme's background would paint
                        // over the terminal's own, which is the one the user
                        // chose and the rest of psychic honours.
                        Some(Span::styled(
                            text.to_string(),
                            Style::default().fg(Color::Rgb(
                                style.foreground.r,
                                style.foreground.g,
                                style.foreground.b,
                            )),
                        ))
                    })),
                    Err(_) => spans.push(Span::raw(line.clone())),
                }

                Line::from(spans)
            })
            .collect::<Vec<_>>();

        Generated {
            text: Text::from(rendered),
            complete,
        }
    }
}

/// A preview, and whether there is any more of it to ask for.
struct Generated {
    text: Text<'static>,
    complete: bool,
}

impl Generated {
    /// A one-line message in place of a preview. Never has more to come.
    fn note(message: &str) -> Self {
        Self {
            text: Text::from(Line::from(Span::styled(
                message.to_string(),
                Style::default().fg(Color::DarkGray),
            ))),
            complete: true,
        }
    }
}

/// What reading a file for preview found.
enum Content {
    Text { lines: Vec<String>, complete: bool },
    Binary,
    Empty,
}

/// Read at most `max_lines` lines, sanitised, and say whether that was all.
///
/// Line by line rather than reading the file and slicing, because for a long
/// file the read is the cheap part and everything after it scales with what we
/// keep.
fn read_text(path: &Path, max_lines: usize) -> Result<Content, std::io::Error> {
    use std::io::{BufRead, BufReader};

    let mut reader = BufReader::new(std::fs::File::open(path)?);

    // `fill_buf` hands back the first block without consuming it, which is
    // exactly what the binary check wants to look at.
    let head = reader.fill_buf()?;
    if head.is_empty() {
        return Ok(Content::Empty);
    }
    if head[..head.len().min(SNIFF_BYTES)].contains(&0) {
        return Ok(Content::Binary);
    }

    let mut lines = Vec::new();
    let mut raw = Vec::new();
    let mut bytes = 0;
    let mut complete = false;

    while lines.len() < max_lines {
        raw.clear();
        let read = reader.read_until(b'\n', &mut raw)?;
        if read == 0 {
            complete = true;
            break;
        }
        bytes += read;

        // Sanitised here, once, so nothing downstream has to remember to. The
        // NUL check above only sniffs the start of the file; this is what makes
        // a file that turns to rubbish halfway through render as dots instead
        // of driving the terminal.
        let line = String::from_utf8_lossy(&raw);
        let line = line.trim_end_matches('\n').trim_end_matches('\r');
        lines.push(printable(line).into_owned());

        if bytes >= MAX_BYTES {
            complete = true;
            break;
        }
    }

    Ok(Content::Text { lines, complete })
}

/// A directory listing: permissions, size, date, name.
///
/// The columns after the name are dropped on a narrow pane, which is what the
/// `--no-permissions --no-user --no-time` flags used to do for `eza`.
fn directory(path: &Path, width: u16) -> Generated {
    let mut entries: Vec<Entry> = match std::fs::read_dir(path) {
        Ok(dir) => dir.flatten().take(MAX_LINES).map(Entry::of).collect(),
        Err(e) => return Generated::note(&format!("[unable to list: {}]", e)),
    };

    if entries.is_empty() {
        return Generated::note("[empty directory]");
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

    Generated {
        text: Text::from(lines),
        complete: true,
    }
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

    /// A file preview, with room for anything a test writes.
    fn preview_of(path: &Path) -> Generated {
        Generator::new().file(path, MAX_LINES)
    }

    /// The visible text of a preview, one string per line.
    fn rendered(generated: &Generated) -> Vec<String> {
        generated
            .text
            .lines
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
            rendered(&preview_of(&path)),
            vec!["[binary file]"],
            "Painting the bytes is what used to corrupt the pane"
        );
    }

    #[test]
    fn test_an_empty_file_says_so() {
        let dir = TempDir::new("empty");
        let path = dir.write("nothing.txt", b"");

        assert_eq!(rendered(&preview_of(&path)), vec!["[empty file]"]);
    }

    #[test]
    fn test_a_missing_file_does_not_panic() {
        let text = preview_of(Path::new("/definitely/not/here.txt"));
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
            rendered(&preview_of(&path)),
            vec!["1 fn main() {", "2     let x = 1;", "3 }"],
            "Line numbers, and the source unchanged beside them"
        );
    }

    #[test]
    fn test_highlighting_actually_colours_something() {
        let dir = TempDir::new("colours");
        let path = dir.write("hello.rs", b"fn main() {}\n");

        let text = preview_of(&path);
        let colours: Vec<_> = text.text.lines[0]
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
    fn assert_nothing_escapes(generated: &Generated) {
        for line in rendered(generated) {
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

        let text = preview_of(&path);
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

        assert_nothing_escapes(&preview_of(&path));
    }

    #[test]
    fn test_windows_line_endings_do_not_leave_a_mark() {
        let dir = TempDir::new("crlf");
        let path = dir.write("dos.txt", b"first\r\nsecond\r\n");

        assert_eq!(
            rendered(&preview_of(&path)),
            vec!["1 first", "2 second"],
            "A carriage return is a control character, but CRLF is just a newline"
        );
    }

    #[test]
    fn test_tabs_are_expanded_so_the_layout_can_count_them() {
        let dir = TempDir::new("tabs");
        let path = dir.write("indented.txt", b"a\tb\n");

        assert_eq!(rendered(&preview_of(&path)), vec!["1 a    b"]);
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
            preview_of(&path).text.lines.len(),
            MAX_LINES,
            "A preview is not a pager"
        );
    }

    #[test]
    fn test_only_the_lines_asked_for_are_generated() {
        let dir = TempDir::new("budget");
        let path = dir.write("long.md", "# heading\n".repeat(2_000).as_bytes());

        let generated = Generator::new().file(&path, 40);

        assert_eq!(
            generated.text.lines.len(),
            40,
            "Highlighting all two thousand lines to show forty is what made a \
             large markdown file take 150ms"
        );
        assert!(
            !generated.complete,
            "and the caller has to know there is more, or it can never scroll"
        );
    }

    #[test]
    fn test_a_file_shorter_than_the_budget_is_complete() {
        let dir = TempDir::new("short");
        let path = dir.write("short.txt", b"one\ntwo\n");

        let generated = Generator::new().file(&path, 100);

        assert_eq!(rendered(&generated), vec!["1 one", "2 two"]);
        assert!(
            generated.complete,
            "Nothing more to ask for, so scrolling should stop at the end"
        );
    }

    /// A preview of `lines` blank lines, as though the thread had answered.
    fn answered(id: u64, lines: usize, complete: bool) -> Preview {
        Preview {
            id,
            text: Text::from(vec![Line::from("x"); lines]),
            complete,
        }
    }

    #[test]
    fn test_a_screenful_until_the_user_scrolls_then_the_rest_once() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut state = PreviewState::new(tx);
        let pane = PreviewPane {
            width: 100,
            height: 40,
        };
        let path = Path::new("/some/file.md");

        state.request(path, false, pane);
        let first = rx.try_recv().expect("something should have been asked for");
        assert_eq!(first.lines, 80, "the screen on show, and one in hand");

        state.request(path, false, pane);
        assert!(rx.try_recv().is_err(), "asking twice for the same thing");

        state.ready(answered(first.id, 80, false));
        state.request(path, false, pane);
        assert!(
            rx.try_recv().is_err(),
            "a screenful is still all anyone can see"
        );

        // The first scroll buys the rest of the file, in one pass.
        state.scroll(3);
        state.request(path, false, pane);
        let second = rx.try_recv().expect("scrolling should ask for the rest");
        assert_eq!(second.lines, MAX_LINES);

        state.ready(answered(second.id, 500, true));
        for _ in 0..20 {
            state.scroll(3);
            state.request(path, false, pane);
            assert!(
                rx.try_recv().is_err(),
                "and then scrolling is free: no regeneration per wheel click"
            );
        }
    }

    #[test]
    fn test_a_stale_answer_is_not_shown() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut state = PreviewState::new(tx);
        let pane = PreviewPane {
            width: 100,
            height: 40,
        };

        state.request(Path::new("/one.rs"), false, pane);
        let first = rx.try_recv().expect("a request");

        // The selection moves before the answer comes back.
        state.clear();
        state.request(Path::new("/two.rs"), false, pane);
        let second = rx.try_recv().expect("a second request");

        state.ready(answered(first.id, 80, true));
        assert!(
            state.text_for(Path::new("/one.rs")).is_none(),
            "the row it was generated for is not the row selected now"
        );

        state.ready(answered(second.id, 80, true));
        assert!(state.text_for(Path::new("/two.rs")).is_some());
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
