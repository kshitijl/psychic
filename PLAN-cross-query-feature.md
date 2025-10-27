# Plan: New Feature `clicks_in_episode_with_query`

## Goal

Create a NEW feature (separate from existing `clicks_for_this_query`) that counts clicks on a file when the current query appeared anywhere in the episode, not just as the exact query at click time.

## Example

User types "tc" → "todo" → "todo-current" → clicks "todo-current.md"

**Existing feature (`clicks_for_this_query`):**
- Only "todo-current" gets +1 (exact match at click time)

**New feature (`clicks_in_episode_with_query`):**
- All three queries get +1: "tc", "todo", "todo-current"
- Next time user types "tc", this file will rank higher

## Database Schema Change

**Add episode_queries column to events table:**
```sql
ALTER TABLE events ADD COLUMN episode_queries TEXT;
```

**Storage format:**
- JSON array string: `["tc", "todo", "todo-current"]`
- Only populated on click events
- NULL for impressions/scrolls (not needed)
- Easy to parse with serde_json in Rust

## Implementation Steps

### 1. Schema Migration (db.rs)

In `Database::new()`, add column if not exists:
```sql
ALTER TABLE events ADD COLUMN IF NOT EXISTS episode_queries TEXT
```

No data migration needed initially (NULLs are fine).

### 2. Track Episode Queries at Runtime (main.rs or event logging)

Add to app state:
```rust
episode_queries: Vec<String>  // Track queries in current episode
```

Logic:
- On impression: `if !episode_queries.contains(&query) { episode_queries.push(query) }`
- On click: serialize to JSON and pass to event logging
  ```rust
  let episode_queries_json = serde_json::to_string(&app.episode_queries)?;
  db.log_event_with_episode_queries(..., episode_queries_json);
  ```
- On episode end (click/scroll/session change): `episode_queries.clear()`

### 3. Backfill During Feature Generation (features.rs)

Accumulator already tracks `episode_queries` during feature generation.

After recording click, UPDATE the database:
```sql
UPDATE events
SET episode_queries = ?
WHERE id = ?
```

Or batch updates for efficiency.

### 4. Load at Startup (ranker.rs)

Modify `load_clicks()` query:
```sql
SELECT full_path, timestamp, query, episode_queries
FROM events
WHERE action = 'click' AND timestamp >= ?
```

Build new index while processing results:
```rust
let mut clicks_by_episode_query_and_file: FxHashMap<(String, String), Vec<ClickEvent>> = FxHashMap::default();

for (path, ts, query, episode_queries_json) in rows {
    let click = ClickEvent { timestamp: ts };

    // Keep existing logic unchanged
    clicks_by_query_and_file.entry((query, path.clone())).push(click);

    // NEW: Parse episode queries and build cross-query index
    if let Some(json) = episode_queries_json {
        let queries: Vec<String> = serde_json::from_str(&json)?;
        for episode_query in queries {
            clicks_by_episode_query_and_file
                .entry((episode_query, path.clone()))
                .or_default()
                .push(click);
        }
    }
}
```

### 5. Add New Field to ClickData (ranker.rs)

```rust
pub struct ClickData {
    pub clicks_by_file: FxHashMap<String, Vec<ClickEvent>>,
    pub clicks_by_parent_dir: FxHashMap<PathBuf, Vec<ClickEvent>>,
    pub clicks_by_query_and_file: FxHashMap<(String, String), Vec<ClickEvent>>,
    pub clicks_by_episode_query_and_file: FxHashMap<(String, String), Vec<ClickEvent>>,  // NEW
}
```

### 6. Update FeatureInputs (schema.rs)

```rust
pub struct FeatureInputs<'a> {
    // ... existing fields ...
    pub clicks_by_episode_query_and_file: &'a FxHashMap<(String, String), Vec<ClickEvent>>,  // NEW
}
```

### 7. Implement New Feature (implementations.rs)

```rust
pub struct ClicksInEpisodeWithQuery;

impl Feature for ClicksInEpisodeWithQuery {
    fn name(&self) -> &'static str {
        "clicks_in_episode_with_query"
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }

    fn monotonicity(&self) -> Option<Monotonicity> {
        Some(Monotonicity::Increasing)
    }

    fn compute(&self, inputs: &FeatureInputs) -> Result<f64> {
        let full_path_str = inputs.full_path.to_string_lossy().to_string();
        let key = (inputs.query.to_string(), full_path_str);

        let clicks = inputs
            .clicks_by_episode_query_and_file
            .get(&key)
            .map(|clicks| clicks.len())
            .unwrap_or(0);

        Ok(clicks as f64)
    }
}
```

### 8. Register Feature (registry.rs)

Add to FEATURE_REGISTRY after ClicksForThisQuery:
```rust
Box::new(ClicksInEpisodeWithQuery),
```

### 9. Update All FeatureInputs Creation Sites

- features.rs: Pass `clicks_by_episode_query_and_file` when creating FeatureInputs
- ranker.rs: Pass `clicks_by_episode_query_and_file` when creating FeatureInputs

### 10. Testing

- Unit test: episode query tracking (add/clear logic)
- Unit test: JSON serialization/deserialization
- Unit test: feature computation with mock data
- Integration test: full flow with real database

## Performance Analysis

**Startup Query:**
- Single table scan of clicks (1,000-5,000 rows)
- JSON parsing: ~1-5 short strings per click
- Total: ~5,000 parses
- **Estimated: 30-80ms** (acceptable)

**Storage:**
- ~50-200 bytes per click (JSON array)
- For 5,000 clicks: ~250KB-1MB (negligible)

**Runtime:**
- O(1) HashMap lookup (no change from current)

**Why This is Fast:**
- No JOINs (avoid O(n²) comparisons)
- Single table scan (like current click loading)
- Trivial JSON parsing in Rust
- All data self-contained on click event

## Migration Path

1. Deploy schema change (add column)
2. App starts populating episode_queries on new clicks
3. Run `psychic generate-features` to backfill old clicks
4. Feature immediately useful for new data
5. Fully useful after backfill

## Key Benefits

✅ No expensive JOINs
✅ Fast startup (~50ms additional)
✅ Self-contained data (easy to debug)
✅ Human-readable in DB
✅ Separate from existing feature (can compare effectiveness)

## CSV Output

Training data will have TWO columns:
- `clicks_for_this_query`: Exact query match at click time
- `clicks_in_episode_with_query`: Any query in the episode

Model can learn the relative importance of each signal.
