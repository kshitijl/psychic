# Plan: Unified Event Loop for Instant Worker Response Rendering

## Goal
Wake up and redraw **instantly** when worker sends results, without waiting for the event poll timeout to expire.

## Problem Statement

### Current Architecture
```rust
loop {
    terminal.draw(...)?;                      // Always draw, even if nothing changed
    while let Ok(msg) = worker_rx.try_recv() { ... }  // Non-blocking check
    event::poll(Duration::from_millis(100))?;          // Sleep up to 100ms!
}
```

### Performance Issue
Even though the worker completes queries in ~6ms, the main thread can be sleeping in `event::poll()` for up to 100ms before it wakes up and redraws. This causes:

**Measured Impact:**
- Worker responds: 6ms
- Wait for poll timeout: 0-100ms (avg ~50ms)
- **Total time to first render: 6-106ms** (feels sluggish!)

From actual timing data:
```
first_query_complete: 6.06ms
first_full_render_complete: 107.5ms
Gap: 101ms of waiting!
```

## Solution: Unified Event Channel

Create a single event channel that ALL event sources write to (worker, keyboard/mouse, tick timer). Main loop blocks on this channel with `recv()` - when ANY event arrives, we wake instantly and handle it.

### Architecture

```
┌─────────────┐
│Worker Thread│──┐
└─────────────┘  │
                 │   ┌──────────────┐      ┌───────────┐
┌─────────────┐  ├──→│ event_tx (Sender) │─→│ event_rx  │──→ Main Loop
│Input Thread │──┤   │  (cloned 3x)       │  │ (Receiver)│    (recv blocks)
└─────────────┘  │   └──────────────┘      └───────────┘
                 │
┌─────────────┐  │
│Tick Thread  │──┘
└─────────────┘

All threads send to ONE channel → Main blocks on recv() → Instant wake!
```

## Why This Design?

### Alternatives Considered

**Option 1: Just reduce timeout (simplest)**
```rust
event::poll(Duration::from_millis(5))?;  // Reduce from 100ms to 5ms
```
- ❌ Still up to 5ms delay
- ❌ Wastes CPU (polls 200x/second when idle)
- ❌ Draws every loop even when nothing changed

**Option 2: Check worker before poll**
```rust
if let Ok(msg) = worker_rx.try_recv() {
    handle_worker(msg);
    terminal.draw()?;
}
event::poll(timeout)?;
```
- ✅ Instant worker response
- ❌ Still polls in a loop
- ❌ Higher CPU during startup
- ❌ Not the standard Ratatui pattern

**Option 3: Unified event channel with select! macro (crossbeam)**
```rust
select! {
    recv(worker_rx) -> msg => { ... }
    recv(event_rx) -> evt => { ... }
    recv(tick_rx) -> _ => { ... }
}
```
- ✅ Proper multiplexing
- ❌ Requires crossbeam dependency
- ❌ More complex (select! macro)
- ⚠️ Risk of starvation if worker floods

**Option 4: Unified event channel with single recv() (CHOSEN)**
```rust
enum AppEvent { Worker(...), Input(...), Tick }
let event = event_rx.recv()?;  // Blocks until ANY event
match event { ... }
```
- ✅ True blocking recv - zero CPU when idle
- ✅ Instant wake on any event (worker, keyboard, tick)
- ✅ Simple code - just recv() and match
- ✅ No extra dependencies (std::sync::mpsc)
- ✅ FIFO ordering - fair to all event sources
- ✅ Standard Ratatui pattern (event-driven)
- ✅ Only 2 new threads (lightweight)

### Why Separate Threads Instead of Async?

**Async would require:**
- Adding tokio dependency (~1MB binary, slower compile)
- Making worker async (but it does CPU work - blocking is better)
- Making walker async (but filesystem I/O doesn't benefit from async)
- Using async channels, `.await` everywhere
- Still need to spawn_blocking for worker/walker

**Thread-based approach:**
- Worker does CPU-bound work → best as blocking thread
- Walker does filesystem I/O → best as blocking thread
- Main does event handling → can be blocking recv()
- Only 2 tiny new threads (crossterm + tick) that mostly sleep
- No async complexity, no tokio dependency

**Modern systems handle dozens of threads easily. 7 threads (5 existing + 2 new) is totally fine.**

### Why stdlib mpsc Instead of crossbeam-channel?

We only need:
- Multiple senders → one receiver ✅ std::sync::mpsc supports this
- Clone sender ✅ std::sync::mpsc::Sender is Clone
- Blocking recv ✅ std::sync::mpsc::Receiver::recv() blocks

We DON'T need:
- select! macro (we use single channel)
- Multiple receivers (only main thread receives)
- Extreme performance (3 senders, not thousands)

**Result: stdlib is perfect, no extra dependencies needed!**

## Implementation Plan

### 1. Create AppEvent Enum (main.rs)

```rust
enum AppEvent {
    Worker(WorkerResponse),
    Input(crossterm::event::Event),
    Tick,
    Retrain(bool),  // Retraining status updates
    Log(String),    // Log messages
}
```

### 2. Modify search_worker::spawn() Signature

**Before:**
```rust
pub fn spawn(cwd: PathBuf, data_dir: &Path, ...)
    -> Result<(Sender<WorkerRequest>, Receiver<WorkerResponse>, JoinHandle<()>)>
```

**After:**
```rust
pub fn spawn(
    cwd: PathBuf,
    data_dir: &Path,
    event_tx: mpsc::Sender<AppEvent>,  // NEW: unified event channel
    no_click_loading: bool,
    no_model: bool,
) -> Result<(Sender<WorkerRequest>, JoinHandle<()>)>  // No more worker_rx!
```

### 3. Update Worker to Send to Unified Channel

In `search_worker.rs`, change all `result_tx.send(WorkerResponse::...)` to:

```rust
event_tx.send(AppEvent::Worker(WorkerResponse::...)).unwrap();
```

### 4. Create Event Collection Threads (main.rs)

**Thread 1: Crossterm events**
```rust
let input_tx = event_tx.clone();
thread::spawn(move || {
    loop {
        if event::poll(Duration::MAX).unwrap() {
            let evt = event::read().unwrap();
            input_tx.send(AppEvent::Input(evt)).unwrap();
        }
    }
});
```

**Thread 2: Tick timer**
```rust
let tick_tx = event_tx.clone();
thread::spawn(move || {
    loop {
        std::thread::sleep(Duration::from_millis(250));
        tick_tx.send(AppEvent::Tick).unwrap();
    }
});
```

### 5. Refactor Main Event Loop

**Before:**
```rust
loop {
    terminal.draw(...)?;
    while let Ok(msg) = worker_rx.try_recv() { ... }
    while let Ok(msg) = retrain_rx.try_recv() { ... }
    while let Ok(msg) = log_rx.try_recv() { ... }
    if event::poll(Duration::from_millis(100))? { ... }
}
```

**After:**
```rust
loop {
    // Block until ANY event arrives
    let event = event_rx.recv()?;

    // Handle event
    match event {
        AppEvent::Worker(response) => {
            handle_worker_response(&mut app, response, &mut first_query_logged, main_start);
        }
        AppEvent::Input(evt) => {
            if handle_input_event(&mut app, evt, &mut terminal)? {
                break;  // User quit
            }
        }
        AppEvent::Tick => {
            handle_tick(&mut app);
        }
        AppEvent::Retrain(status) => {
            app.currently_retraining = status;
        }
        AppEvent::Log(msg) => {
            app.recent_logs.push_back(msg.trim_end().to_string());
            if app.recent_logs.len() > 50 {
                app.recent_logs.pop_front();
            }
        }
    }

    // Redraw after handling event
    terminal.draw(|f| { render_ui(f, &mut app, ...) })?;
}
```

### 6. Extract Event Handlers

Move event handling logic into separate functions for clarity:
- `handle_worker_response()`
- `handle_input_event()`
- `handle_tick()`

### 7. Update App::new() Call

```rust
let (event_tx, event_rx) = mpsc::channel();

let (worker_tx, worker_handle) = search_worker::spawn(
    root.clone(),
    &data_dir,
    event_tx.clone(),  // Pass unified channel
    cli.no_click_loading,
    cli.no_model
)?;

let mut app = App::new(
    root.clone(),
    &data_dir,
    log_rx,  // Still used for now, will migrate to event_tx
    ...
)?;
```

### 8. Migrate Retrain Thread

Change retrain thread to send to unified channel:

```rust
let retrain_tx = event_tx.clone();
thread::spawn(move || {
    retrain_tx.send(AppEvent::Retrain(true)).unwrap();
    // ... do retraining ...
    retrain_tx.send(AppEvent::Retrain(false)).unwrap();
});
```

### 9. Migrate Log Receiver

Change fern logging to send to unified channel instead of separate mpsc.

## Expected Performance Impact

**Current:**
- Worker completes: 6ms
- Main thread sleeping in event::poll(): 0-100ms (avg ~50ms)
- **Total: 6-106ms to render** (avg ~56ms)

**After:**
- Worker completes: 6ms
- Worker sends AppEvent::Worker
- Main thread wakes instantly from recv()
- Handles event + redraws: ~2-4ms
- **Total: ~8-10ms to render**

**Result: 5-10x faster perceived startup!**

## Files to Modify

1. `src/main.rs` - Major refactor of event loop
2. `src/search_worker.rs` - Change spawn() signature, send to event_tx
3. `how-it-works.md` - Document new architecture

## Rollback Plan

If issues arise:
1. Git revert the commit
2. Previous architecture is well-tested and working

## Testing Strategy

1. `cargo build --release` - must compile
2. `cargo test` - all tests must pass
3. Manual testing:
   - Launch app, verify instant results display
   - Type queries, verify instant filtering
   - Navigate with arrow keys, verify responsiveness
   - Scroll preview with mouse, verify it works
   - Press Ctrl-H for history mode
   - Press Ctrl-F for filter picker
   - Navigate to directories
   - Quit with Ctrl-C

## Success Criteria

- ✅ App compiles without errors
- ✅ All tests pass
- ✅ First full render < 20ms (currently ~107ms)
- ✅ Keyboard input feels instant
- ✅ Worker results appear instantly
- ✅ No regressions in functionality
