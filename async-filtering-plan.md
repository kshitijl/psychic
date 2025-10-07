# Plan: Async Filtering/Ranking to Fix UI Lag

## Problem

Currently, every keystroke in the search bar blocks the UI thread while:
1. Filtering 5000 files
2. Getting file metadata for each match
3. Running ML model inference
4. Computing features for ranking

This causes noticeable lag when typing, especially with large file counts.

## Current Flow (Blocking)

```
User types 'a'
  → app.query.push('a')
  → app.update_filtered_files()  // BLOCKS HERE (100-500ms)
      → Filter all 5000 files
      → Get metadata for matches
      → Run ranker.rank_files() (ML inference + feature computation)
      → Update app.filtered_files and app.file_scores
  → Render UI
  → User sees update
```

## Solution: Background Thread with Debouncing

Move filtering/ranking to a background thread. Main thread stays responsive.

```
User types 'a'
  → app.query.push('a')
  → Send query to filter_tx channel
  → Render UI immediately (shows old results)
  → Check filter_rx for results
  → If available: update and re-render

Background thread:
  → Receive query from channel
  → Debounce: wait 50ms, check if newer query arrived
  → If still latest: run filter + rank
  → Send FilterResult back to main thread
```

## Implementation Details

### 1. New Types

```rust
struct FilterResult {
    query: String,
    filtered_files: Vec<FileEntry>,
    file_scores: Vec<ranker::FileScore>,
}

enum FilterRequest {
    Update(String),  // New query to process
    Shutdown,        // Tell thread to exit
}
```

### 2. Modify App Struct

```rust
struct App {
    // ... existing fields ...

    // New fields for async filtering
    filter_tx: Sender<FilterRequest>,
    filter_rx: Receiver<FilterResult>,
    pending_query: Option<String>,  // Track what's being processed
}
```

### 3. Setup in App::new()

```rust
fn new(root: PathBuf) -> Result<Self> {
    // ... existing setup ...

    // Create channels
    let (filter_tx, filter_task_rx) = mpsc::channel::<FilterRequest>();
    let (filter_result_tx, filter_rx) = mpsc::channel::<FilterResult>();

    // Clone data needed by background thread
    let files_clone = files.clone();
    let historical_clone = historical_files.clone();
    let root_clone = root.clone();
    let session_id_clone = session_id.clone();
    let ranker_clone = /* Need to make Ranker cloneable or use Arc */;

    // Spawn background filtering thread
    std::thread::spawn(move || {
        let mut last_query: Option<String> = None;

        loop {
            // Receive next filter request
            match filter_task_rx.recv() {
                Ok(FilterRequest::Shutdown) => break,
                Ok(FilterRequest::Update(query)) => {
                    // Debounce: wait 50ms for more updates
                    std::thread::sleep(Duration::from_millis(50));

                    // Check if newer query arrived
                    if let Ok(FilterRequest::Update(newer)) = filter_task_rx.try_recv() {
                        // Skip this one, process newer query
                        continue;
                    }

                    // Still latest, process it
                    let result = do_filtering_and_ranking(
                        &query,
                        &files_clone,
                        &historical_clone,
                        &root_clone,
                        &session_id_clone,
                        &ranker_clone,
                    );

                    let _ = filter_result_tx.send(result);
                }
                Err(_) => break,
            }
        }
    });

    Ok(App {
        // ... existing fields ...
        filter_tx,
        filter_rx,
        pending_query: None,
    })
}
```

### 4. Extract Filtering Logic

```rust
fn do_filtering_and_ranking(
    query: &str,
    files: &[PathBuf],
    historical_files: &[PathBuf],
    root: &Path,
    session_id: &str,
    ranker: &ranker::Ranker,
) -> FilterResult {
    // This is the current update_filtered_files() logic
    // Move it to a standalone function

    let query_lower = query.to_lowercase();
    let mut file_entries = Vec::new();
    let mut matching_files = Vec::new();

    // Filter files...
    // Get metadata...
    // Run ranker...

    FilterResult {
        query: query.to_string(),
        filtered_files: file_entries,
        file_scores: scored,
    }
}
```

### 5. Update Key Handling

```rust
// In main event loop
KeyCode::Char(c) => {
    app.query.push(c);
    // Send to background thread (non-blocking)
    let _ = app.filter_tx.send(FilterRequest::Update(app.query.clone()));
    app.pending_query = Some(app.query.clone());
    // UI renders immediately with old results
}

KeyCode::Backspace => {
    app.query.pop();
    let _ = app.filter_tx.send(FilterRequest::Update(app.query.clone()));
    app.pending_query = Some(app.query.clone());
}
```

### 6. Poll for Results in Main Loop

```rust
// After rendering, check for filter results
if let Ok(result) = app.filter_rx.try_recv() {
    // Only apply if this matches our pending query
    if app.pending_query.as_ref() == Some(&result.query) {
        app.filtered_files = result.filtered_files;
        app.file_scores = result.file_scores;
        app.pending_query = None;

        // Reset selection if needed
        if app.selected_index >= app.filtered_files.len() && !app.filtered_files.is_empty() {
            app.selected_index = 0;
        }

        // Re-render with new results (will happen on next iteration)
    }
}
```

### 7. Cleanup on Exit

```rust
// Before exiting main()
let _ = app.filter_tx.send(FilterRequest::Shutdown);
```

## Ranker Sharing Strategy

The ranker needs to be shared between main thread and filter thread. Options:

### Option 1: Arc<Ranker>
```rust
use std::sync::Arc;

struct App {
    ranker: Arc<ranker::Ranker>,
    // ...
}

// Clone Arc for background thread
let ranker_clone = Arc::clone(&ranker);
```

### Option 2: Make Ranker Clone (if lightweight)
```rust
// In ranker.rs, if model can be shared
#[derive(Clone)]
pub struct Ranker { /* ... */ }
```

**Recommendation: Use Arc<Ranker>** since LightGBM model is read-only after loading.

## Expected Performance

- **Before:** 100-500ms blocking on each keystroke
- **After:**
  - UI renders in <16ms (responsive immediately)
  - Results appear 50-150ms after user stops typing
  - Smooth typing experience even with 5000 files

## Testing Plan

1. Test with 5000 files, verify no lag when typing
2. Test rapid typing (ensure debouncing works)
3. Test that results match current implementation
4. Test edge cases: empty query, no matches, etc.

## Files to Modify

- `src/main.rs`: Main changes (channels, background thread, key handling)
- `src/ranker.rs`: Possibly add Clone or use Arc

## Notes

- Keep logging in background thread (already goes to app.log)
- Consider increasing debounce delay if 50ms isn't enough
- Could add visual indicator for "searching..." state
