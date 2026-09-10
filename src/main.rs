mod analytics;
mod analyze_perf;
mod app;
mod cli;
mod context;
mod db;
mod feature_defs;
mod features;
mod help;
mod history;
mod input;
mod keymap;
mod metadata_ext;
mod path_display;
mod preview;
mod ranker;
mod render;
mod search_worker;
mod tty_input;
mod ui_state;
mod walker;

use anyhow::{Context, Result};
use clap::Parser;
use crossterm::{
    event::{Event, KeyboardEnhancementFlags, PushKeyboardEnhancementFlags},
    execute,
    terminal::{EnterAlternateScreen, enable_raw_mode},
};
use db::EventData;
use metadata_ext::MetadataExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use search_worker::{WorkerRequest, WorkerResponse};
use std::{
    env,
    sync::mpsc::{self, Receiver},
    time::{Duration, Instant},
};

/// Unified event type for the main event loop.
/// All event sources (worker, keyboard/mouse, tick timer) send to a single channel.
enum AppEvent {
    /// Worker thread sent a response (query results, page data, etc.)
    Worker(WorkerResponse),
    /// User input event (keyboard or mouse)
    Input(Event),
    /// Periodic tick for background tasks (marquee scroll, impression logging)
    Tick,
    /// Model retraining status update
    Retrain(bool),
    /// Database statistics for the debug pane, gathered off the UI thread.
    /// Boxed to keep the event enum small; this arrives at most once per run.
    DbStats(Box<db::DbStats>),
    /// A preview finished generating. Boxed because `Text` is much larger than
    /// every other variant, and this enum is passed by value on every event.
    Preview(Box<preview::Preview>),
}

impl From<WorkerResponse> for AppEvent {
    fn from(response: WorkerResponse) -> Self {
        AppEvent::Worker(response)
    }
}

impl From<Event> for AppEvent {
    fn from(event: Event) -> Self {
        AppEvent::Input(event)
    }
}

impl From<preview::Preview> for AppEvent {
    fn from(generated: preview::Preview) -> Self {
        AppEvent::Preview(Box::new(generated))
    }
}

// Import app types from dedicated module
use app::{App, AppBootstrap, AppOptions, Page};

// Import CLI types from dedicated module
use cli::{Cli, Commands, FilterArg, InternalCommands, OutputFormat};

/// When this process started.
///
/// One shared zero point for every latency psychic reports, so numbers from the
/// debug pane, the TIMING log lines and `internal analyze-perf` can be compared
/// with each other. Threads other than main need it too - the worker times the
/// filesystem walk - and a `Lazy` is the simplest way to give them all the same
/// instant without threading it through every constructor.
///
/// Initialized by the first statement of `main`, so it really is process start.
pub static PROCESS_START: once_cell::sync::Lazy<Instant> = once_cell::sync::Lazy::new(Instant::now);

/// Time the preview generator, or print what it produces.
///
/// The point is to be measurable from outside: the generator with no thread,
/// channel or terminal around it, so it can be put beside `bat` directly.
/// Loading the syntax definitions is reported separately because the running
/// app pays it once, at startup, on the preview thread - unlike `bat`, which
/// pays it on every invocation.
fn preview_command(path: &std::path::Path, lines: usize, repeat: usize, show: bool) -> Result<()> {
    assert!(repeat > 0, "nothing to measure in zero runs");

    let load_start = Instant::now();
    let generator = preview::Generator::new();
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let is_dir = path.is_dir();
    let width = 120;

    if show {
        print!(
            "{}",
            to_ansi(&generator.generate(path, is_dir, width, lines))
        );
        return Ok(());
    }

    // One run outside the measurement: the first touch of a file is a page
    // fault or a disk read, and that is not what is being compared.
    let _ = generator.generate(path, is_dir, width, lines);

    let mut timings: Vec<f64> = (0..repeat)
        .map(|_| {
            let start = Instant::now();
            let text = generator.generate(path, is_dir, width, lines);
            std::hint::black_box(&text);
            start.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    timings.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));

    let produced = generator.generate(path, is_dir, width, lines).lines.len();
    println!("{}", path.display());
    println!("  syntax load   {:>8.2}ms  (once per process)", load_ms);
    println!(
        "  generate      {:>8.2}ms  median of {} runs, min {:.2}ms, max {:.2}ms",
        timings[timings.len() / 2],
        repeat,
        timings[0],
        timings[timings.len() - 1]
    );
    println!("  produced      {:>8} lines", produced);

    Ok(())
}

/// Styled text back out as ANSI, so `--show` can be eyeballed beside `bat`.
fn to_ansi(text: &ratatui::text::Text<'_>) -> String {
    use ratatui::style::Color;

    let mut out = String::new();
    for line in &text.lines {
        for span in &line.spans {
            match span.style.fg {
                Some(Color::Rgb(r, g, b)) => out.push_str(&format!(
                    "\x1b[38;2;{};{};{}m{}\x1b[0m",
                    r, g, b, span.content
                )),
                _ => out.push_str(&span.content),
            }
        }
        out.push('\n');
    }
    out
}

/// Generate a unique session ID using a random u64.
fn create_session_id() -> String {
    rand::random::<u64>().to_string()
}

fn main() -> Result<()> {
    // Start global timer at the very beginning. Forcing the Lazy here is what
    // pins it to process start rather than to whoever reads it first.
    let main_start = *PROCESS_START;

    // Parsed before logging is set up, because `--data-dir` decides where the
    // log goes. It used to be written to `~/.local/share/psychic` regardless,
    // while `internal analyze-perf` and `print-log` read it from the data
    // directory - so pointing psychic somewhere else split its own log from the
    // commands that read it, and quietly wrote into the default directory.
    let cli = Cli::parse();

    let data_dir = cli
        .data_dir
        .clone()
        .or_else(|| cli::get_default_data_dir().ok());

    // Generate session ID early so we can include it in all logs
    let session_id = create_session_id();

    // Initialize logger with fern to write to both file and memory
    let (log_tx, log_rx) = mpsc::channel();

    if let Some(dir) = &data_dir {
        let _ = std::fs::create_dir_all(dir);
        let log_file = dir.join("app.log");
        rotate_log_if_large(&log_file);

        // The session id is captured, not read from the environment on every
        // line. It also means nothing has to `set_var` before the threads
        // start, which was an `unsafe` block for no gain. (The `unsafe` that
        // remains is `tty_input.rs` calling libc on raw descriptors.)
        let session = session_id.clone();
        fern::Dispatch::new()
            .format(move |out, message, record| {
                out.finish(format_args!(
                    "[{} {} {} {}] {}",
                    jiff::Timestamp::now().strftime("%Y-%m-%d %H:%M:%S"),
                    record.level(),
                    record.target(),
                    session,
                    message
                ))
            })
            .level(log::LevelFilter::Debug)
            .chain(fern::log_file(&log_file).expect("Failed to open log file"))
            .chain(log_tx)
            .apply()
            .expect("Failed to initialize logger");
    }

    // Handle subcommands
    if let Some(command) = cli.command {
        let data_dir = data_dir
            .clone()
            .context("No data directory: pass --data-dir, or set HOME")?;

        match command {
            Commands::GenerateFeatures { format } => {
                // Create data directory if it doesn't exist
                std::fs::create_dir_all(&data_dir)?;

                // Determine output paths
                let default_filename = match format {
                    OutputFormat::Csv => "features.csv",
                    OutputFormat::Json => "features.json",
                };
                let output_path = data_dir.join(default_filename);
                let schema_path = data_dir.join("feature_schema.json");
                let db_path = db::Database::get_db_path(&data_dir);

                // Convert CLI format to features format
                let features_format = match format {
                    OutputFormat::Csv => features::OutputFormat::Csv,
                    OutputFormat::Json => features::OutputFormat::Json,
                };

                let format_str = match features_format {
                    features::OutputFormat::Csv => "CSV",
                    features::OutputFormat::Json => "JSON",
                };

                println!(
                    "Generating features ({}) from DB at {:?} and writing to {:?}",
                    format_str, db_path, output_path
                );
                let summary = features::generate_features(
                    &db_path,
                    &output_path,
                    &schema_path,
                    features_format,
                )?;

                println!(
                    "Generated {} rows, {} of them clicked, at {:?}",
                    summary.rows, summary.positives, output_path
                );
                println!("Generated feature schema at {:?}", schema_path);
                println!("Done.");
                return Ok(());
            }
            Commands::Retrain => {
                println!("Retraining model with data directory: {:?}", data_dir);
                // When called from CLI, print to stdout (no log file)
                ranker::retrain_model(&data_dir, None)?;
                println!("Done.");
                return Ok(());
            }
            Commands::Zsh => {
                // Output zsh integration script
                print!("{}", include_str!("../shell/psychic.zsh"));
                return Ok(());
            }
            Commands::Hidden { command } => {
                let db_path = db::Database::get_db_path(&data_dir);
                let db = db::Database::new(&db_path)?;

                match command {
                    cli::HiddenCommands::List => {
                        let hidden = db.get_hidden_prefixes()?;
                        if hidden.is_empty() {
                            println!("No hidden directories.");
                        } else {
                            for path in hidden {
                                println!("{}", path.display());
                            }
                        }
                    }
                    cli::HiddenCommands::Add { path } => {
                        // Canonicalized so it matches the paths in the file
                        // registry, which are canonical. A directory that is
                        // already gone can still be hidden - falling back to
                        // the path as given is better than refusing.
                        let full_path = path.canonicalize().unwrap_or(path);
                        db.hide_prefix(&full_path)?;
                        println!("Hidden: {}", full_path.display());
                    }
                    cli::HiddenCommands::Remove { path } => {
                        let full_path = path.canonicalize().unwrap_or(path);
                        if db.unhide_prefix(&full_path)? {
                            println!("No longer hidden: {}", full_path.display());
                        } else {
                            println!("Not hidden: {}", full_path.display());
                        }
                    }
                }
                return Ok(());
            }
            Commands::TrackVisit { path } => {
                // Canonicalize the path to get the absolute path
                let full_path = path
                    .canonicalize()
                    .with_context(|| format!("Failed to resolve path: {}", path.display()))?;

                // Only track directories
                if !full_path.is_dir() {
                    anyhow::bail!("Path is not a directory: {}", full_path.display());
                }

                // Open database
                let db_path = db::Database::get_db_path(&data_dir);
                let db = db::Database::new(&db_path)?;

                // Generate a session ID for this tracking event
                let session_id = create_session_id();

                // Get file metadata
                let metadata = std::fs::metadata(&full_path)?;
                let mtime = metadata.mtime_as_secs();
                let atime = metadata.atime_as_secs();

                // Log the directory visit
                db.log_event(db::EventData {
                    query: "",
                    file_path: full_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .as_ref(),
                    full_path: full_path.to_string_lossy().as_ref(),
                    mtime,
                    atime,
                    file_size: None, // directories don't have meaningful sizes
                    subsession_id: 0,
                    action: db::UserInteraction::StartupVisit,
                    session_id: &session_id,
                    episode_queries: None,
                    rank: None,         // not an impression
                    is_dir: Some(true), // a visit is always to a directory
                })?;

                return Ok(());
            }
            Commands::Internal { command } => match command {
                InternalCommands::AnalyzePerf => {
                    let log_path = data_dir.join("app.log");
                    analyze_perf::analyze_perf(&log_path)?;
                    return Ok(());
                }
                InternalCommands::PrintLog => {
                    let log_path = data_dir.join("app.log");
                    let contents =
                        std::fs::read_to_string(&log_path).context("Failed to read log file")?;
                    print!("{}", contents);
                    return Ok(());
                }
                InternalCommands::ClearLog => {
                    let log_path = data_dir.join("app.log");
                    if log_path.exists() {
                        std::fs::remove_file(&log_path).with_context(|| {
                            format!("Failed to delete log file {}", log_path.display())
                        })?;
                        println!("Deleted log file {}", log_path.display());
                    } else {
                        println!("No log file found at {}", log_path.display());
                    }
                    return Ok(());
                }
                InternalCommands::Preview {
                    path,
                    lines,
                    repeat,
                    show,
                } => {
                    preview_command(&path, lines, repeat, show)?;
                    return Ok(());
                }
                InternalCommands::SummarizeEvents => {
                    let db_path = db::Database::get_db_path(&data_dir);
                    let db = db::Database::new(&db_path)?;
                    let summary = db.summarize_events()?;

                    println!("Event Summary:");
                    println!("{:<20} {:>10}", "Action", "Count");
                    println!("{}", "-".repeat(32));
                    for (action, count) in summary {
                        println!("{:<20} {:>10}", action, count);
                    }
                    return Ok(());
                }
            },
        }
    }

    log::info!("=== STARTUP TIMING ===");

    // Get current working directory and canonicalize once
    let root_start = Instant::now();
    let root = env::current_dir()?.canonicalize()?;
    log::info!(
        "TIMING {{\"op\":\"get_canonicalize_root\",\"ms\":{}}}",
        root_start.elapsed().as_secs_f64() * 1000.0
    );

    let data_dir = data_dir.context("No data directory: pass --data-dir, or set HOME")?;

    // Create unified event channel - all events (worker, input, tick) flow through this
    let (event_tx, event_rx) = mpsc::channel::<AppEvent>();

    // Thread 1: terminal input. It sleeps in the kernel until a key arrives, a
    // resize happens, or we ask it to stop so a child process can have the
    // terminal. See tty_input.rs for why that is harder than it sounds.
    let input = tty_input::spawn(event_tx.clone()).context("Failed to start the input thread")?;

    // Thread 2: preview generation. Kept off the UI thread because how long a
    // preview takes depends on the file, and the pane must never be what the
    // rest of the frame is waiting for.
    let preview_tx = preview::spawn(event_tx.clone());

    // Start background retraining in a new thread
    let retrain_start = Instant::now();
    let data_dir_clone = data_dir.clone();
    let training_log_path = data_dir.join("training.log");
    let retrain_event_tx = event_tx.clone();
    std::thread::spawn(move || {
        let _ = retrain_event_tx.send(AppEvent::Retrain(true)); // Signal retraining started
        log::info!("Starting background model retraining");
        if let Err(e) = ranker::retrain_model(&data_dir_clone, Some(training_log_path)) {
            log::error!("Background retraining failed: {}", e);
        } else {
            log::info!("Background retraining completed successfully");
        }
        let _ = retrain_event_tx.send(AppEvent::Retrain(false)); // Signal retraining completed
    });
    log::info!(
        "TIMING {{\"op\":\"spawn_retrain_thread\",\"ms\":{}}}",
        retrain_start.elapsed().as_secs_f64() * 1000.0
    );

    // Convert CLI filter to internal FilterType
    let initial_filter = match cli.filter {
        Some(FilterArg::None) | None => search_worker::FilterType::None,
        Some(FilterArg::Cwd) => search_worker::FilterType::OnlyCwd,
        Some(FilterArg::Direct) => search_worker::FilterType::DirectCwd,
        Some(FilterArg::Dirs) => search_worker::FilterType::OnlyDirs,
        Some(FilterArg::Files) => search_worker::FilterType::OnlyFiles,
    };

    // Initialize app
    let app_new_start = Instant::now();
    let bootstrap = AppBootstrap {
        session_id,
        event_tx: event_tx.clone(),
        input,
        preview_tx,
    };

    // Detect editor from environment, with fallback chain
    let editor = env::var("EDITOR")
        .or_else(|_| env::var("VISUAL"))
        .unwrap_or_else(|_| "vi".to_string());

    let options = AppOptions {
        on_dir_click: cli.on_dir_click.clone(),
        on_cwd_visit: cli.on_cwd_visit.clone(),
        initial_filter,
        no_preview: cli.no_preview,
        no_click_loading: cli.no_click_loading,
        no_model: cli.no_model,
        no_click_logging: cli.no_click_logging,
        respect_gitignore: !cli.no_ignore,
        editor,
    };

    let mut app = App::new(root.clone(), &data_dir, bootstrap, options)?;
    log::info!(
        "TIMING {{\"op\":\"app_new\",\"ms\":{}}}",
        app_new_start.elapsed().as_secs_f64() * 1000.0
    );

    // Thread 2: Tick timer for periodic tasks (spawned after app creation to access tick_paused)
    let tick_tx = event_tx.clone();
    let tick_paused = app.tick_paused.clone();
    let something_animates = app.something_animates.clone();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(200));

            // Paused during editor and shell suspension, and quiet whenever
            // nothing on screen is moving. A tick is a full redraw, so sending
            // one with nothing to animate costs a redraw five times a second
            // for as long as psychic sits open.
            let idle = !something_animates.load(std::sync::atomic::Ordering::Relaxed);
            let paused = tick_paused.load(std::sync::atomic::Ordering::Relaxed);
            if paused || idle {
                continue;
            }
            if tick_tx.send(AppEvent::Tick).is_err() {
                break; // Main thread died, exit
            }
        }
    });

    let log_session_start = Instant::now();
    log::info!(
        "Started psychic in directory {}, session {}",
        root.display(),
        app.analytics.session_id()
    );
    log::info!(
        "TIMING {{\"op\":\"session_log_message\",\"ms\":{}}}",
        log_session_start.elapsed().as_secs_f64() * 1000.0
    );

    // Gather context in background thread and log initial directory click
    let context_spawn_start = Instant::now();
    let session_id_clone = app.analytics.session_id().to_string();
    let data_dir_clone = data_dir.clone();
    let root_clone = root.clone();
    std::thread::spawn(move || {
        let context = context::gather_context();
        let db_path = db::Database::get_db_path(&data_dir_clone);
        match db::Database::new(&db_path) {
            Ok(db) => {
                // Log the initial directory as a startup visit (with empty query)
                // This happens in background thread so it doesn't block startup
                let dir_name = root_clone
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(".");

                // Get metadata for the directory
                let metadata = std::fs::metadata(&root_clone).ok();
                let mtime = metadata.as_ref().and_then(|m| m.mtime_as_secs());
                let atime = metadata.as_ref().and_then(|m| m.atime_as_secs());
                let file_size = metadata.as_ref().map(|m| m.len() as i64);

                match db.log_event(EventData {
                    query: "",
                    file_path: dir_name,
                    full_path: &root_clone.to_string_lossy(),
                    mtime,
                    atime,
                    file_size,
                    subsession_id: 0, // Initial event, before any query
                    action: db::UserInteraction::StartupVisit,
                    session_id: &session_id_clone,
                    episode_queries: None,
                    rank: None,         // not an impression
                    is_dir: Some(true), // a visit is always to a directory
                }) {
                    Ok(_) => log::info!("Logged startup visit for {}", root_clone.display()),
                    Err(e) => log::error!("Failed to log startup visit: {:?}", e),
                }

                // Continue with context logging
                if let Err(e) = db.log_session(&session_id_clone, &context) {
                    log::error!("Failed to log session context: {:?}", e);
                }
            }
            Err(e) => {
                log::error!("Failed to open database for startup visit: {:?}", e);
            }
        }
    });
    log::info!(
        "TIMING {{\"op\":\"spawn_context_thread\",\"ms\":{}}}",
        context_spawn_start.elapsed().as_secs_f64() * 1000.0
    );

    // Setup terminal
    let terminal_setup_start = Instant::now();
    // Use /dev/tty directly so shell integration can redirect stdout
    let mut tty = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")?;

    enable_raw_mode()?;
    execute!(tty, EnterAlternateScreen)?;
    execute!(tty, crossterm::event::EnableMouseCapture)?;
    // Enable enhanced keyboard protocol for better modifier key detection
    execute!(
        tty,
        PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
    )?;
    let backend = CrosstermBackend::new(tty);
    let mut terminal = Terminal::new(backend)?;
    log::info!(
        "TIMING {{\"op\":\"terminal_setup\",\"ms\":{}}}",
        terminal_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    log::info!(
        "TIMING {{\"op\":\"main_setup_total\",\"ms\":{}}}",
        main_start.elapsed().as_secs_f64() * 1000.0
    );

    // Run the app
    let result = run_app(&mut terminal, &mut app, event_rx, main_start, &log_rx);

    // Shutdown. The one rule: **the log sink must outlive everything that logs.**
    // Background threads log as they wind down, and several of them are only
    // told to stop by `app` being dropped, so if `app` owned the receiving end
    // of the logging channel it would take the sink with it - and fern reports
    // a dead channel by printing the whole record to stderr, over the terminal
    // we are in the middle of restoring.
    //
    // `log_rx` is owned by `main` for exactly this reason, and dropped last.
    // Joining threads one at a time only ever fixed the thread being joined;
    // the retraining and context threads are detached and can log at any moment.
    let worker_tx = std::mem::replace(&mut app.worker_tx, mpsc::channel().0);
    // Ask the worker to stop, rather than relying on this being the last
    // sender: the walker holds a clone so it can send into the same channel,
    // and it blocks forever waiting for its next command, so dropping this one
    // leaves the worker waiting on a channel that never disconnects.
    let _ = worker_tx.send(WorkerRequest::Shutdown);
    drop(worker_tx);

    if let Some(handle) = app.worker_handle.take() {
        let _ = handle.join();
    }

    // Not for the logging - that is handled above - but so that nothing is
    // still reading the terminal while we put it back the way we found it.
    app.input.shutdown();

    drop(app);

    // Terminal cleanup, through the same function every other hand-back uses.
    input::leave_tui(&mut terminal)?;

    // Everything that could have logged has now been told to stop, and the
    // terminal is back. Anything still arriving goes nowhere, quietly.
    drop(log_rx);

    result
}

/// Move `app.log` aside once it gets big, keeping one generation.
///
/// Checked at startup rather than on every write: `fern` has no rotation, and a
/// size check per line would put a `stat` in front of every log call on the
/// worker thread. Once per launch is enough for a file that grows by a few
/// hundred kilobytes a day.
///
/// One generation, not many. The log is a debugging aid read by
/// `internal analyze-perf` and `print-log`, both of which want the current
/// session; the previous file is there for the case where something went wrong
/// last time and psychic has since been restarted.
fn rotate_log_if_large(log_file: &std::path::Path) {
    const MAX_LOG_BYTES: u64 = 8 * 1024 * 1024;

    let Ok(metadata) = std::fs::metadata(log_file) else {
        return; // no log yet, which is the common case on a first launch
    };
    if metadata.len() < MAX_LOG_BYTES {
        return;
    }

    let previous = log_file.with_extension("log.1");
    if let Err(e) = std::fs::rename(log_file, &previous) {
        // Not fatal: logging to an oversized file beats not starting.
        eprintln!("Could not rotate {}: {}", log_file.display(), e);
    }
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
    app: &mut App,
    event_rx: Receiver<AppEvent>,
    main_start: Instant,
    log_rx: &Receiver<String>,
) -> Result<()> {
    let mut first_render_logged = false;
    // Captured inside the draw closure, applied to app once the borrow ends.
    let mut first_paint_ms: Option<f64> = None;
    let mut first_query_complete_logged = false;
    let mut first_full_render_logged = false;

    let marquee_delay = Duration::from_millis(500); // 0.5s pause at ends
    let marquee_speed = Duration::from_millis(80); // scroll every 80ms

    loop {
        // Log impressions for this subsession if >200ms old
        let _ = app.check_and_log_impressions(false);

        // Draw UI
        let draw_start = Instant::now();
        let mut frame_layout = None;
        let mut help_scroll_max = None;
        let mut preview_pane = None;
        terminal.draw(|f| {
            // Log first render
            if !first_render_logged {
                let elapsed = main_start.elapsed().as_secs_f64() * 1000.0;
                log::info!("TIMING {{\"op\":\"first_render\",\"ms\":{}}}", elapsed);
                first_paint_ms = Some(elapsed);
                first_render_logged = true;
            }

            // If in history mode, render history-specific UI
            if app.ui_state.history_mode {
                let filtered_history = app.get_filtered_history();
                let history_ctx = render::HistoryRenderContext {
                    filtered_history: &filtered_history,
                    history_selected: app.history_selected,
                    total_history_items: app.history.items_for_display().len(),
                    preview: &app.preview,
                    query: &app.query,
                };
                preview_pane = Some(render::render_history_mode(f, history_ctx));

                // The help screen is reachable from every mode, so it is drawn
                // last, over whichever mode is underneath.
                if app.ui_state.help_visible {
                    help_scroll_max =
                        Some(render::render_help_overlay(f, app.ui_state.help_scroll));
                }
                return;
            }

            // Render normal mode UI using render module. It reads `&App` and
            // writes nothing back; what it works out from the geometry comes
            // back in the layout.
            let layout = render::render_normal_mode(f, app);
            preview_pane = layout.preview_pane;
            frame_layout = Some(layout);

            if app.ui_state.help_visible {
                help_scroll_max = Some(render::render_help_overlay(f, app.ui_state.help_scroll));
            }
        })?;

        if let Some(ms) = first_paint_ms.take() {
            app.timings.first_paint_ms = Some(ms);
        }

        // Scrolling the help screen is clamped to what actually overflowed the
        // last frame, which only the renderer knows.
        if let Some(max) = help_scroll_max {
            app.ui_state.help_scroll_max = max;
            app.ui_state.help_scroll = app.ui_state.help_scroll.min(max);
        }

        // Now that the layout is known, ask for the preview of whatever is
        // selected. Generating it during the draw would put a file read, and
        // once a process spawn, in front of every frame.
        if let Some(pane) = preview_pane {
            app.update_preview(pane);
        }

        // Take what only the layout knew, now that the frame is drawn.
        if let Some(layout) = frame_layout {
            app.path_bar_width = layout.path_bar_width;
            app.visible_list_height = layout.visible_list_height;
            app.something_animates.store(
                layout.path_bar_overflows,
                std::sync::atomic::Ordering::Relaxed,
            );
            app.update_scroll(layout.visible_list_height, layout.file_list_scroll);
        }

        // Log draw time and check for first full render (with data)
        let draw_time = draw_start.elapsed().as_secs_f64() * 1000.0;
        if !first_full_render_logged && app.total_results > 0 {
            log::info!(
                "TIMING {{\"op\":\"first_full_render_complete\",\"ms\":{}}}",
                main_start.elapsed().as_secs_f64() * 1000.0
            );
            log::info!(
                "TIMING {{\"op\":\"first_full_render_draw_time\",\"ms\":{}}}",
                draw_time
            );
            first_full_render_logged = true;
        }

        // Check for startup complete: walker done + we have results + UI rendered
        if !app.startup_complete_logged && app.walker_done && app.total_results > 0 {
            log::info!(
                "TIMING {{\"op\":\"startup_complete\",\"ms\":{}}}",
                main_start.elapsed().as_secs_f64() * 1000.0
            );
            app.startup_complete_logged = true;
        }

        // Collect new log messages (non-blocking)
        while let Ok(log_msg) = log_rx.try_recv() {
            // fern adds a newline to each message sent via channel, so trim it
            app.recent_logs.push_back(log_msg.trim_end().to_string());
            // Keep only last 50 messages
            if app.recent_logs.len() > 50 {
                app.recent_logs.pop_front();
            }
        }

        // Block until ANY event arrives (worker, input, tick, retrain, log)
        let app_event = event_rx.recv()?;

        // Handle the event
        match app_event {
            AppEvent::Retrain(retraining_status) => {
                app.currently_retraining = retraining_status;
            }
            AppEvent::DbStats(stats) => {
                app.db_stats = Some(*stats);
            }
            AppEvent::Preview(generated) => {
                app.preview.ready(*generated);
            }
            AppEvent::Tick => {
                // The tick is what moves the marquee along. Drawing a frame used
                // to do it, which made the animation a side effect of rendering
                // and meant render could not take `&App`.
                app.advance_marquee(marquee_delay, marquee_speed);
            }
            AppEvent::Worker(response) => {
                // Handle worker response
                match response {
                    WorkerResponse::QueryUpdated {
                        query_id,
                        total_results,
                        total_files,
                        initial_page,
                        model_stats,
                        rank_ms,
                    } => {
                        let active_query_id = app.analytics.current_subsession_id();

                        // Only accept updates for queries that are newer than what we currently have.
                        // This handles out-of-order responses.
                        if query_id >= active_query_id {
                            // Clear page cache on query change
                            app.page_cache.clear();

                            // Insert initial page
                            let page = Page {
                                start_index: initial_page.start_index,
                                end_index: initial_page.end_index,
                                files: initial_page.files,
                            };
                            app.page_cache.insert(initial_page.page_num, page);

                            app.total_results = total_results;
                            app.total_files = total_files;
                            app.model_stats_cache = model_stats;

                            // Reset selection if needed
                            if app.selected_index >= total_results {
                                app.selected_index = 0;
                            }

                            // Log first query completion
                            if !first_query_complete_logged {
                                let elapsed = main_start.elapsed().as_secs_f64() * 1000.0;
                                log::info!(
                                    "TIMING {{\"op\":\"first_query_complete\",\"ms\":{}}}",
                                    elapsed
                                );
                                app.timings.first_results_ms = Some(elapsed);
                                first_query_complete_logged = true;
                            }

                            app.timings.last_rank_ms = Some(rank_ms);
                            app.note_query_completed(query_id);

                            // Create subsession, using the query text from the app state
                            app.analytics.new_subsession(query_id, app.query.clone());
                        }
                    }
                    WorkerResponse::Page {
                        query_id,
                        page_data,
                    } => {
                        let active_query_id = app.analytics.current_subsession_id();

                        // Only accept page data for the currently active query
                        if query_id == active_query_id {
                            // Insert page into cache
                            let page = Page {
                                start_index: page_data.start_index,
                                end_index: page_data.end_index,
                                files: page_data.files,
                            };
                            app.page_cache.insert(page_data.page_num, page);
                        }
                    }
                    WorkerResponse::FilesChanged => {
                        // The worker detected file changes, so we trigger a refresh
                        // of the current query to get fresh results.
                        log::info!("Auto-refreshing query due to file changes.");
                        let query_id = app.next_query_id();
                        let _ = app.worker_tx.send(WorkerRequest::UpdateQuery(
                            search_worker::UpdateQueryRequest {
                                query: app.query.clone(),
                                query_id,
                                filter: app.current_filter,
                            },
                        ));
                    }
                    WorkerResponse::WalkerDone { walk_ms } => {
                        app.walker_done = true;
                        // The worker's own measurement, so the pane and the
                        // walker_complete log line are the same number.
                        app.timings.walk_complete_ms = Some(walk_ms);
                    }
                }
            }
            AppEvent::Input(event_read) => {
                // Delegate all input handling to input module
                match input::handle_input(app, event_read, terminal)? {
                    input::InputAction::Exit => return Ok(()),
                    input::InputAction::PrintAndExit(path) => {
                        println!("{}", path);
                        return Ok(());
                    }
                    input::InputAction::Continue => {}
                }
            }
        }

        // Checked here rather than trusted to be impossible: the worker owns
        // the file registry and the model, so if it stops, every keystroke
        // silently does nothing and the list on screen is frozen but still
        // redrawing. Saying so and leaving beats looking merely slow.
        if app.worker_has_died() {
            anyhow::bail!(
                "The search worker stopped unexpectedly. See the log at \
                 ~/.local/share/psychic/app.log"
            );
        }
    }
}

// Tests for path display functions moved to src/path_display.rs

#[cfg(test)]
mod log_rotation_tests {
    use super::*;

    fn dir(name: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("psychic-log-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn test_a_small_log_is_left_alone() {
        let dir = dir("small");
        let log = dir.join("app.log");
        std::fs::write(&log, "one line\n").unwrap();

        rotate_log_if_large(&log);

        assert_eq!(std::fs::read_to_string(&log).unwrap(), "one line\n");
        assert!(!dir.join("app.log.1").exists(), "nothing to rotate");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_a_big_log_moves_aside_and_the_next_launch_starts_clean() {
        let dir = dir("big");
        let log = dir.join("app.log");
        std::fs::write(&log, vec![b'x'; 9 * 1024 * 1024]).unwrap();

        rotate_log_if_large(&log);

        assert!(!log.exists(), "the current log is out of the way");
        assert_eq!(
            std::fs::metadata(dir.join("app.log.1")).unwrap().len(),
            9 * 1024 * 1024,
            "and is kept as one previous generation"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_rotating_twice_keeps_only_the_previous_one() {
        // One generation, not a growing pile: the older file is what gets
        // overwritten, which is the point of bounding the log at all.
        let dir = dir("twice");
        let log = dir.join("app.log");

        std::fs::write(&log, vec![b'a'; 9 * 1024 * 1024]).unwrap();
        rotate_log_if_large(&log);
        std::fs::write(&log, vec![b'b'; 9 * 1024 * 1024]).unwrap();
        rotate_log_if_large(&log);

        let kept = std::fs::read(dir.join("app.log.1")).unwrap();
        assert_eq!(kept[0], b'b', "the newer of the two is the one kept");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_a_missing_log_is_not_an_error() {
        let dir = dir("missing");
        rotate_log_if_large(&dir.join("app.log")); // a first launch
        std::fs::remove_dir_all(&dir).ok();
    }
}
