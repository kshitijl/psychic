use std::path::PathBuf;
use std::sync::mpsc::Sender;
use walkdir::WalkDir;

const IGNORED_DIRS: &[&str] = &[".git", "node_modules", ".venv", "target"];

pub fn start_file_walker(root: PathBuf, tx: Sender<PathBuf>) {
    std::thread::spawn(move || {
        for entry in WalkDir::new(&root)
            .into_iter()
            .filter_entry(|e| {
                // Skip ignored directories
                if e.file_type().is_dir()
                    && let Some(name) = e.file_name().to_str() {
                    return !IGNORED_DIRS.contains(&name);
                }
                true
            })
            .filter_map(|e| e.ok())
        {
            // Only send files, not directories
            if entry.file_type().is_file() {
                let _ = tx.send(entry.path().to_path_buf());
            }
        }
    });
}
