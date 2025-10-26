mod analytics;
mod analyze_perf;
mod app;
mod cli;
mod context;
mod db;
mod feature_defs;
mod features;
mod history;
mod input;
mod path_display;
mod preview;
mod ranker;
mod render;
mod search_worker;
mod ui_state;
mod walker;

use anyhow::Result;
use clap::Parser;
use crossterm::{
    cursor::Show,
    event::{self, Event},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use db::EventData;
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
}

impl From<WorkerResponse> for AppEvent {
    fn from(response: WorkerResponse) -> Self {
        AppEvent::Worker(response)
    }
}

// Import app types from dedicated module
use app::{App, Page};

// Import CLI types from dedicated module
use cli::{Cli, Commands, FilterArg, InternalCommands, OutputFormat};

fn main() -> Result<()> {
    // Start global timer at the very beginning
    let main_start = Instant::now();

    // Generate session ID early so we can include it in all logs
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};
    let mut hasher = RandomState::new().build_hasher();
    Instant::now().hash(&mut hasher);
    std::process::id().hash(&mut hasher);
    let session_id = hasher.finish().to_string();

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
            Commands::Internal { command } => match command {
                InternalCommands::AnalyzePerf => {
                    let log_path = data_dir.join("app.log");
                    analyze_perf::analyze_perf(&log_path)?;
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

    // Create control channel for pausing/resuming crossterm thread
    let (input_control_tx, input_control_rx) = crossbeam::channel::unbounded::<bool>();

    // Thread 1: Pauseable crossterm event forwarder
    let input_tx = event_tx.clone();
    std::thread::spawn(move || {
        loop {
            // Use crossbeam::select! to wait on both control channel and check for events
            crossbeam::select! {
                recv(input_control_rx) -> msg => {
                    match msg {
                        Ok(false) => {
                            // Paused - wait for resume signal
                            loop {
                                match input_control_rx.recv() {
                                    Ok(true) => break,  // Resumed
                                    Ok(false) => continue,  // Still paused
                                    Err(_) => return,  // Channel closed, exit thread
                                }
                            }
                        }
                        Ok(true) => {
                            // Already running, continue
                        }
                        Err(_) => break,  // Channel closed, exit thread
                    }
                }
                default(Duration::from_millis(10)) => {
                    // Check for crossterm events (non-blocking poll)
                    if event::poll(Duration::ZERO).unwrap_or(false) {
                        match event::read() {
                            Ok(evt) => {
                                if input_tx.send(AppEvent::Input(evt)).is_err() {
                                    break;  // Main thread died, exit
                                }
                            }
                            Err(_) => break,  // Error reading events, exit
                        }
                    }
                }
            }
        }
    });

    // Thread 2: Tick timer for periodic tasks
    let tick_tx = event_tx.clone();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(200));
            if tick_tx.send(AppEvent::Tick).is_err() {
                break; // Main thread died, exit
            }
        }
    });

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
    let mut app = App::new(
        root.clone(),
        &data_dir,
        log_rx,
        cli.on_dir_click.clone(),
        cli.on_cwd_visit.clone(),
        initial_filter,
        cli.no_preview,
        cli.no_click_loading,
        cli.no_model,
        cli.no_click_logging,
        event_tx.clone(),
        input_control_tx.clone(),
    )?;
    log::info!(
        "TIMING {{\"op\":\"app_new\",\"ms\":{}}}",
        app_new_start.elapsed().as_secs_f64() * 1000.0
    );

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
        if let Ok(db) = db::Database::new(&data_dir_clone) {
            // Log the initial directory as a startup visit (with empty query)
            // This happens in background thread so it doesn't block startup
            let dir_name = root_clone
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(".");

            // Get metadata for the directory
            let metadata = std::fs::metadata(&root_clone).ok();
            let mtime = metadata.as_ref().and_then(|m| {
                m.modified().ok().and_then(|t| {
                    t.duration_since(std::time::UNIX_EPOCH)
                        .ok()
                        .map(|d| d.as_secs() as i64)
                })
            });
            let atime = metadata.as_ref().and_then(|m| {
                m.accessed().ok().and_then(|t| {
                    t.duration_since(std::time::UNIX_EPOCH)
                        .ok()
                        .map(|d| d.as_secs() as i64)
                })
            });
            let file_size = metadata.as_ref().map(|m| m.len() as i64);

            let _ = db.log_event(EventData {
                query: "",
                file_path: dir_name,
                full_path: &root_clone.to_string_lossy(),
                mtime,
                atime,
                file_size,
                subsession_id: 0, // Initial event, before any query
                action: db::UserInteraction::StartupVisit,
                session_id: &session_id_clone,
            });

            // Continue with context logging
            let _ = db.log_session(&session_id_clone, &context);
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

    // Now it's safe to drop app, which will close the logging channel
    drop(app);

    // Terminal cleanup
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
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
        terminal.draw(|f| {
            // Log first render
            if !first_render_logged {
                log::info!(
                    "TIMING {{\"op\":\"first_render\",\"ms\":{}}}",
                    main_start.elapsed().as_secs_f64() * 1000.0
                );
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
                return;
            }

            // Render normal mode UI using render module
            let normal_ctx = render::NormalRenderContext {
                selected_index: app.selected_index,
                file_list_scroll: app.file_list_scroll,
                total_results: app.total_results,
                total_files: app.total_files,
                current_filter: app.current_filter,
                no_preview: app.no_preview,
                preview: &app.preview,
                currently_retraining: app.currently_retraining,
                model_stats_cache: app.model_stats_cache.as_ref(),
                page_cache: &app.page_cache,
                ui_state: &app.ui_state,
                recent_logs: &app.recent_logs,
                last_path_bar_update: app.last_path_bar_update,
                path_bar_scroll: app.path_bar_scroll,
                path_bar_scroll_direction: app.path_bar_scroll_direction,
                cwd: app.cwd.as_path(),
                query: &app.query,
            };
            let updates = render::render_normal_mode(f, normal_ctx, marquee_delay, marquee_speed);
            render_updates = Some(updates);
        })?;

        // Apply render updates to app state after rendering is complete
        if let Some(updates) = render_updates {
            updates.apply_to(app);
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
                                log::info!(
                                    "TIMING {{\"op\":\"first_query_complete\",\"ms\":{}}}",
                                    main_start.elapsed().as_secs_f64() * 1000.0
                                );
                                first_query_complete_logged = true;
                            }

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
                        let query_id = app.analytics.next_subsession_id();
                        let _ = app.worker_tx.send(WorkerRequest::UpdateQuery(
                            search_worker::UpdateQueryRequest {
                                query: app.query.clone(),
                                query_id,
                                filter: app.current_filter,
                            },
                        ));
                    }
                    WorkerResponse::WalkerDone => {
                        app.walker_done = true;
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
