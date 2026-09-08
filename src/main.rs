mod analytics;
mod analyze_perf;
mod app;
mod cli;
mod context;
mod db;
mod episode;
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
    cursor::Show,
    event::{
        Event, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use db::EventData;
use metadata_ext::MetadataExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use search_worker::{WorkerRequest, WorkerResponse};
use std::{
    env,
    path::PathBuf,
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

/// Generate a unique session ID using a random u64.
fn create_session_id() -> String {
    rand::random::<u64>().to_string()
}

fn main() -> Result<()> {
    // Start global timer at the very beginning. Forcing the Lazy here is what
    // pins it to process start rather than to whoever reads it first.
    let main_start = *PROCESS_START;

    // Generate session ID early so we can include it in all logs
    let session_id = create_session_id();

    // Set as environment variable so all threads can access it
    // SAFETY: We set this once at the very beginning of main() before any other threads exist
    unsafe {
        std::env::set_var("PSYCHIC_SESSION_ID", &session_id);
    }

    // Initialize logger with fern to write to both file and memory
    let (log_tx, log_rx) = mpsc::channel();

    if let Ok(home) = std::env::var("HOME") {
        let log_dir = PathBuf::from(&home)
            .join(".local")
            .join("share")
            .join("psychic");
        let _ = std::fs::create_dir_all(&log_dir);
        let log_file = log_dir.join("app.log");

        fern::Dispatch::new()
            .format(|out, message, record| {
                let session =
                    std::env::var("PSYCHIC_SESSION_ID").unwrap_or_else(|_| "unknown".to_string());
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
            .chain(fern::log_file(log_file).expect("Failed to open log file"))
            .chain(log_tx)
            .apply()
            .expect("Failed to initialize logger");
    }

    let cli = Cli::parse();

    // Handle subcommands
    if let Some(command) = cli.command {
        // Get data directory (global option)
        let data_dir = cli.data_dir.unwrap_or_else(|| {
            cli::get_default_data_dir().expect("Failed to get default data directory")
        });

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
                features::generate_features(&db_path, &output_path, &schema_path, features_format)?;

                println!("Generated features at {:?}", output_path);
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

    // Get data directory for main app
    let data_dir = cli.data_dir.unwrap_or_else(|| {
        cli::get_default_data_dir().expect("Failed to get default data directory")
    });

    // Create unified event channel - all events (worker, input, tick) flow through this
    let (event_tx, event_rx) = mpsc::channel::<AppEvent>();

    // Thread 1: terminal input. It sleeps in the kernel until a key arrives, a
    // resize happens, or we ask it to stop so a child process can have the
    // terminal. See tty_input.rs for why that is harder than it sounds.
    let input = tty_input::spawn(event_tx.clone()).context("Failed to start the input thread")?;

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
        log_receiver: log_rx,
        event_tx: event_tx.clone(),
        input,
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
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(200));

            // Skip sending tick if paused (e.g., during editor/shell suspension)
            if !tick_paused.load(std::sync::atomic::Ordering::Relaxed)
                && tick_tx.send(AppEvent::Tick).is_err()
            {
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
    let result = run_app(&mut terminal, &mut app, event_rx, main_start);

    // Shutdown sequence: extract and drop worker_tx to signal the worker to stop,
    // then wait for the worker thread to finish, THEN drop app (which drops log_rx).
    // This ensures the worker can log its shutdown message before the logging channel closes.
    let worker_tx = std::mem::replace(&mut app.worker_tx, mpsc::channel().0);
    drop(worker_tx);

    if let Some(handle) = app.worker_handle.take() {
        let _ = handle.join();
    }

    // The input thread logs as it exits, and it only starts exiting when its
    // handle is dropped - which, left to `drop(app)`, happens *after* the field
    // holding the logging channel. Same reasoning as the worker above.
    app.input.shutdown();

    // Now it's safe to drop app, which will close the logging channel
    drop(app);

    // Terminal cleanup
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        PopKeyboardEnhancementFlags,
        crossterm::event::DisableMouseCapture,
        LeaveAlternateScreen,
        Show
    )?;

    result
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<std::fs::File>>,
    app: &mut App,
    event_rx: Receiver<AppEvent>,
    main_start: Instant,
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
        let mut render_updates = None;
        let mut help_scroll_max = None;
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
                    preview_scroll_position: app.preview.scroll_position() as u16,
                    query: &app.query,
                };
                render::render_history_mode(f, history_ctx);

                // The help screen is reachable from every mode, so it is drawn
                // last, over whichever mode is underneath.
                if app.ui_state.help_visible {
                    help_scroll_max =
                        Some(render::render_help_overlay(f, app.ui_state.help_scroll));
                }
                return;
            }

            // Render normal mode UI using render module
            let normal_ctx = render::NormalRenderContext {
                selected_index: app.selected_index,
                file_list_scroll: app.file_list_scroll,
                total_results: app.total_results,
                total_files: app.total_files,
                current_filter: app.current_filter,
                no_preview: app.options.no_preview,
                preview: &app.preview,
                currently_retraining: app.currently_retraining,
                model_stats_cache: app.model_stats_cache.as_ref(),
                timings: &app.timings,
                db_stats: app.db_stats.as_ref(),
                page_cache: &app.page_cache,
                ui_state: &app.ui_state,
                recent_logs: &app.recent_logs,
                last_path_bar_update: app.last_path_bar_update,
                path_bar_scroll: app.path_bar_scroll,
                path_bar_scroll_direction: app.path_bar_scroll_direction,
                cwd: app.cwd.as_path(),
                query: &app.query,
                status_message: app.status_message.as_deref(),
            };
            let updates = render::render_normal_mode(f, normal_ctx, marquee_delay, marquee_speed);
            render_updates = Some(updates);

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

        // Apply render updates to app state after rendering is complete
        if let Some(mut updates) = render_updates {
            let visible_height = updates.visible_list_height;
            let scroll_override = updates.file_list_scroll;
            updates.visible_list_height = None;
            updates.apply_to(app);
            if let Some(height) = visible_height {
                app.update_scroll(height, scroll_override);
            }
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
        while let Ok(log_msg) = app.log_receiver.try_recv() {
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
            AppEvent::Tick => {
                // Tick event - just triggers a redraw for marquee animation
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
    }
}

// Tests for path display functions moved to src/path_display.rs
