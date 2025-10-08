# Async Filtering Implementation - Current State

## Design Overview

Worker thread owns all file data. Main thread only keeps what's visible (20 items).

### Thread Architecture:
1. **Main UI Thread**: Renders UI, handles events, polls for updates
2. **Walker Thread**: Discovers files, sends to worker
3. **Filter/Rank Worker Thread**: Owns data, does filtering/ranking

### Data Flow:
```
[Walker] → (path, mtime) → [Worker]
[Main] → UpdateQuery("foo") → [Worker]
[Worker] → QueryUpdated{visible_slice, total} → [Main]
[Main] → GetVisibleSlice{start, count} → [Worker]
[Worker] → VisibleSlice(vec) → [Main]
```

## Implementation Progress

### ✅ Completed:

1. **Created communication types** (lines 134-155):
```rust
struct DisplayFileInfo {
    file_id: FileId,
    display_name: String,
    score: f64,
}

enum WorkerRequest {
    UpdateQuery(String),
    GetVisibleSlice { start: usize, count: usize },
    Shutdown,
}

enum WorkerResponse {
    QueryUpdated {
        query: String,
        total_results: usize,
        visible_slice: Vec<DisplayFileInfo>,
    },
    VisibleSlice(Vec<DisplayFileInfo>),
}
```

2. **Created unsafe Send wrapper** for Ranker (lines 157-159):
```rust
struct SendRanker(ranker::Ranker);
unsafe impl Send for SendRanker {}
```
LightGBM Booster contains raw pointers, not Send by default. Safe for read-only ops.

3. **Created WorkerState** (lines 161-284):
   - Owns: file_registry, path_to_id, filtered_files, file_scores, ranker
   - Methods: new(), add_file(), filter_and_rank(), get_slice()
   - Moved ALL filtering/ranking logic from App

4. **Implemented worker_thread_loop** (lines 286-353):
   - Processes walker updates (non-blocking via try_recv)
   - Receives worker requests with 10ms timeout
   - Debounces queries (drains all pending, keeps latest)
   - Sends back results with first 20 items

5. **Updated App struct** (lines 355-377):
   - Removed: file_registry, path_to_id, filtered_files, file_scores, ranker
   - Added: visible_files (Vec<DisplayFileInfo>), total_results, worker_tx, worker_rx
   - Only keeps what's visible on screen

6. **Updated App::new()** (lines 380-457):
   - Creates worker/walker channels
   - Initializes WorkerState with ranker
   - Spawns worker thread
   - Spawns walker thread (directly to worker)

## ❌ Remaining Work (46 compilation errors)

### 1. Remove/Update Old Methods

**Files to modify:**

#### `reload_model()` (line ~459)
- Currently tries to access `self.ranker` (doesn't exist)
- **Solution**: Send UpdateQuery to worker to trigger re-filter with new model
  - Or: Add `ReloadModel` variant to WorkerRequest
  - Worker would reload ranker, re-filter current query

#### `reload_and_rerank()` (line ~468)
- Currently tries to access `self.ranker.clicks`
- **Solution**: Add `ReloadClicks` variant to WorkerRequest
  - Worker reloads clicks, re-filters current query

#### `update_filtered_files()` (line ~477)
- Entire method is obsolete
- **Solution**: Delete it entirely
  - All callers should send WorkerRequest::UpdateQuery instead

### 2. Update Event Handlers

**Keyboard input** (in main event loop):
```rust
// OLD:
KeyCode::Char(c) => {
    app.query.push(c);
    app.update_filtered_files();  // BLOCKS
}

// NEW:
KeyCode::Char(c) => {
    app.query.push(c);
    let _ = app.worker_tx.send(WorkerRequest::UpdateQuery(app.query.clone()));
    // Continue immediately, UI renders with old visible_files
}
```

**Backspace:**
```rust
// OLD:
KeyCode::Backspace => {
    app.query.pop();
    app.update_filtered_files();
}

// NEW:
KeyCode::Backspace => {
    app.query.pop();
    let _ = app.worker_tx.send(WorkerRequest::UpdateQuery(app.query.clone()));
}
```

**Scroll/Navigation:**
```rust
// OLD:
KeyCode::Down => {
    app.selected_index += 1;
    if app.selected_index >= app.filtered_files.len() {
        app.selected_index = app.filtered_files.len().saturating_sub(1);
    }
}

// NEW:
KeyCode::Down => {
    app.selected_index += 1;
    if app.selected_index >= app.total_results {
        app.selected_index = app.total_results.saturating_sub(1);
    }

    // Check if we need more visible data
    let visible_start = app.file_list_scroll as usize;
    let visible_end = visible_start + 20;
    if app.selected_index >= visible_end - 5 {
        // Request next slice
        let _ = app.worker_tx.send(WorkerRequest::GetVisibleSlice {
            start: visible_start,
            count: 40,  // Get more to avoid constant requests
        });
    }
}
```

### 3. Add Result Polling in Main Loop

**After event handling, before rendering:**
```rust
// Poll for worker responses (non-blocking)
while let Ok(response) = app.worker_rx.try_recv() {
    match response {
        WorkerResponse::QueryUpdated { query, total_results, visible_slice } => {
            // Only apply if query still matches
            if query == app.query {
                app.visible_files = visible_slice;
                app.total_results = total_results;

                // Reset selection if needed
                if app.selected_index >= total_results {
                    app.selected_index = 0;
                }

                // Create subsession
                app.current_subsession = Some(Subsession {
                    id: app.next_subsession_id,
                    query: query.clone(),
                    created_at: jiff::Timestamp::now(),
                    impressions_logged: false,
                });
                app.next_subsession_id += 1;
            }
        }
        WorkerResponse::VisibleSlice(slice) => {
            app.visible_files = slice;
        }
    }
}
```

### 4. Update UI Rendering

**File list rendering:**
```rust
// OLD:
let items: Vec<ListItem> = app.filtered_files
    .iter()
    .enumerate()
    .skip(scroll_offset)
    .take(visible_count)
    .map(|(i, file_id)| {
        let file_info = &app.file_registry[file_id.0];
        // ... render
    })
    .collect();

// NEW:
let items: Vec<ListItem> = app.visible_files
    .iter()
    .enumerate()
    .map(|(i, display_info)| {
        // display_info has: file_id, display_name, score
        // ... render using display_info
    })
    .collect();
```

**Status line:**
```rust
// OLD:
format!("{}/{} files", app.selected_index + 1, app.filtered_files.len())

// NEW:
format!("{}/{} files", app.selected_index + 1, app.total_results)
```

### 5. Update Check/Log Impressions

**Problem**: Main thread needs file paths to log impressions, but only has visible_files

**Solution A**: Add file_path to DisplayFileInfo
```rust
struct DisplayFileInfo {
    file_id: FileId,
    display_name: String,
    full_path: PathBuf,  // ADD THIS
    score: f64,
}
```

**Solution B**: Add GetFilePath request to worker
```rust
enum WorkerRequest {
    // ...
    GetFilePath(FileId),
}

enum WorkerResponse {
    // ...
    FilePath(FileId, PathBuf),
}
```

Recommendation: **Solution A** - include full_path in DisplayFileInfo since impressions need it.

### 6. Update Preview Loading

**Problem**: Preview loading needs full file path

**Solutions**: Same as impressions - add full_path to DisplayFileInfo

### 7. Handle Shutdown

**In main cleanup:**
```rust
// Before exiting
let _ = app.worker_tx.send(WorkerRequest::Shutdown);
```

## Files to Modify

1. **src/main.rs**:
   - Delete `update_filtered_files()` method
   - Update `reload_model()` to send worker request
   - Update `reload_and_rerank()` to send worker request
   - Update all keyboard handlers (search event loop for `app.query.push`, `app.query.pop`)
   - Add result polling in main loop
   - Update all `app.filtered_files` references to `app.visible_files`
   - Update all `app.file_registry` lookups (no longer exists)
   - Update rendering code
   - Update impression logging
   - Update preview loading
   - Add shutdown message

2. **src/main.rs** (DisplayFileInfo):
   - Add `full_path: PathBuf` field

3. **src/main.rs** (WorkerState::get_slice):
   - Include `full_path` when building DisplayFileInfo

## Search Patterns to Find All References

```bash
# Find all places that need updating:
rg "app\.filtered_files" src/main.rs
rg "app\.file_registry" src/main.rs
rg "app\.file_scores" src/main.rs
rg "app\.ranker" src/main.rs
rg "update_filtered_files" src/main.rs
rg "app\.query\.push" src/main.rs
rg "app\.query\.pop" src/main.rs
```

## Testing Plan

1. **Basic typing**: Type query, verify results appear (with slight delay)
2. **Fast typing**: Type quickly, verify debouncing works (only latest query processed)
3. **Scrolling**: Scroll through results, verify new slices load
4. **Empty query**: Clear query, verify shows all files
5. **No matches**: Type gibberish, verify empty result
6. **Retraining**: Trigger retrain, verify works with worker thread
7. **Preview**: Select file, verify preview loads

## Performance Expectations

**Before async:**
- Each keystroke blocks UI for 30-80ms
- Typing "hello" = 5 × 50ms = 250ms of lag

**After async:**
- Each keystroke: <1ms (just channel send)
- Results appear 50-150ms after typing stops
- Smooth, responsive typing even with 5000 files

## Known Issues / Limitations

1. **Stale results possible**: User might see old results briefly while new ones compute
   - This is acceptable and expected behavior
   - Alternative would be to show "Loading..." but that's more jarring

2. **No model reload in worker yet**: Need to add ReloadModel request variant

3. **Subsession timing**: Currently creates subsession when results arrive, not when query changes
   - Might want to track "pending query" separately

4. **Preview might fail**: If file is only in visible_files and user scrolls before preview loads
   - Need to ensure full_path is in DisplayFileInfo

## Next Steps

1. Add `full_path` to DisplayFileInfo
2. Delete `update_filtered_files()`
3. Update `reload_model()` and `reload_and_rerank()`
4. Find/replace all `app.filtered_files` → `app.visible_files`
5. Find/replace all `app.file_registry` access (need different approach)
6. Update keyboard handlers to use channels
7. Add result polling in main loop
8. Update UI rendering
9. Update impression logging
10. Test thoroughly
