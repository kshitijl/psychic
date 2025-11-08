use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::env;
use std::path::PathBuf;

/// For feature generation output format
#[derive(Debug, Clone, ValueEnum)]
pub enum OutputFormat {
    Csv,
    Json,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Generate features for training from collected events
    GenerateFeatures {
        /// Output format
        #[arg(short, long, value_enum, default_value = "csv")]
        format: OutputFormat,
    },
    /// Retrain the ranking model using collected events. This does everything,
    /// including feature gen.
    Retrain,
    /// Output shell integration script for zsh
    Zsh,
    /// Track a directory visit (for shell integration hooks)
    TrackVisit {
        /// Directory path to track
        path: PathBuf,
    },
    /// Internal development and debugging commands
    Internal {
        #[command(subcommand)]
        command: InternalCommands,
    },
}

#[derive(Subcommand, Debug)]
pub enum InternalCommands {
    /// Analyze performance timings from the latest psychic run
    AnalyzePerf,
    /// Print the application log file
    PrintLog,
    /// Delete the current application log file
    ClearLog,
    /// Summarize events in the database by action type
    SummarizeEvents,
}

/// Filter type for initial filter
#[derive(ValueEnum, Clone, Debug)]
pub enum FilterArg {
    None,
    Cwd,
    Direct,
    Dirs,
    Files,
}

/// Action to take when user presses Enter on a directory
#[derive(ValueEnum, Clone, Debug, PartialEq)]
pub enum OnDirClickAction {
    /// Navigate into the directory (default)
    Navigate,
    /// Print path to stdout and exit
    PrintToStdout,
    /// Drop into a shell in that directory
    DropIntoShell,
}

/// Action to take when user presses Ctrl-J on current directory
#[derive(ValueEnum, Clone, Debug, PartialEq)]
pub enum OnCwdVisitAction {
    /// Print path to stdout and exit (for shell integration)
    PrintToStdout,
    /// Drop into a shell in the current directory
    DropIntoShell,
}

/// A terminal-based file browser that learns which files you want to see.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Data directory for database and training files (default: ~/.local/share/psychic)
    #[arg(long, global = true, value_name = "DIR")]
    pub data_dir: Option<PathBuf>,

    /// Action when pressing Enter on a directory
    #[arg(long, value_enum, default_value = "navigate")]
    pub on_dir_click: OnDirClickAction,

    /// Action when pressing Ctrl-J (current directory visit)
    #[arg(long, value_enum, default_value = "drop-into-shell")]
    pub on_cwd_visit: OnCwdVisitAction,

    /// Set initial filter (none, cwd, dirs, files)
    #[arg(long, value_enum)]
    pub filter: Option<FilterArg>,

    /// Disable preview pane (show blank preview to improve startup time)
    #[arg(long)]
    pub no_preview: bool,

    /// Disable loading click history and historical files (improves startup time)
    #[arg(long)]
    pub no_click_loading: bool,

    /// Disable ML ranking model (sort by mtime instead, improves startup time)
    #[arg(long)]
    pub no_model: bool,

    /// Disable logging clicks/scrolls/impressions to database
    #[arg(long)]
    pub no_click_logging: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

/// Get the default data directory (~/.local/share/psychic)
pub fn get_default_data_dir() -> Result<PathBuf> {
    let home = env::var("HOME").context("HOME environment variable not set")?;
    Ok(PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("psychic"))
}
