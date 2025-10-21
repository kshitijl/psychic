## notes on performance

### could maybe do click aggregation in SQL rather than in the rust code

```
SELECT
    full_path,
    COUNT(*) AS clicks_last_30_days
  FROM events
  WHERE action = 'click'
    AND timestamp >= strftime('%s','now') - 30 * 24 * 60 * 60
  GROUP BY full_path
  ORDER BY clicks_last_30_days DESC;`
```

And cache these in sqlite as a table. Would need to refresh this table at startup, and also lock the table to do the refresh. sqlite doesn't have materialized views. Or just have all this be a view, maybe it's still worth it, idk, haven't measured.

### 2025-10-20

Trying to make startup faster. Time to fully rendered initial screen.

Try commenting out preview. Actually add an argument to turn off preview, so we can always measure startup time without preview.

Added arguments to turn off various features to see if they were the culprit.

The real culprit: a 100ms per frame sleep/timeout. Fixed by refactoring to use a single event stream, and fast polling done on just the input events.

### 2025-10-10

On 5000 files, `update_filtered_files` takes 150-200ms. All of that comes from `rank_files`. The majority of THAT is from feature computation, not actually from running the model.

Update: with some optimizations, model is actually more expensive than computing features.

Feature computation can be sped up in a bunch of ways:
* Divide features into those that depend on the query and those that don't. The ones that don't depend on query don't need to be recomputed on query change.
* Two of them iterate over clicks_by_file. We can probably move parent dir click aggregation, and last 30 days click counting, into a precomputation that happens once at db load time and computes click counts by iterating over all clicks once. Then the features just do lookups? Idk how to do this elegantly though, where adding each feature doesn't become a big chore at db load time. Ah! Maybe each feature also gets to define its own Agg type that can do stuff at db load time? 

Helix file picker is a lot faster. Maybe implement a mode that doesn't sort or run the model or compute features, to make sure it's somewhat as fast.

## now

- audit the whole codebase for modularity. can we refactor extract something into a module, which can then be expect tested? right now its a big ball of very IO heavy code that makes it difficult to test. maybe the overall state logic and keypress logic? maybe the page caching logic? maybe the logic that when walker is finished it sends an AllDone message? maybe the logic that historical files in cwd still need to shown in filter view?
- when history is filtered, suppose number of items becomes less than selected index, then selected index should become 0 so the top item is automatically becomes selected.
- display is broken if we scroll past a binary file and it gets previewed
- if we hit up while file walker is still walking then it shows loading and we end up in some strange middle of the results. instead we should remember our scroll position as -1 and reevaluate that when results are updated.
- pick some good keybindings for going to top, and paging up and down the results
- watch the cwd + all historical files; if mtime changes then update. more generally, our internal file data structure must be kept up-to-date with the filesystem. Right now this works because the file list view polls the filesytem for file metadata every frame or something awful like that. But the fixes below will break that.
- until filewalker is done, don't bother sorting and calculating features? idk. or really, make sure we don't recalculate features? hmm. maybe we want to divide features into query-dependent and query independent?
- watch the cwd. if new files added then add them.
- maybe add a slight linear term?
- try fitting a linear or logistic regressor esp on modified time and num clicks and last time clicked
- hit enter to open Preview or whatever default thing is configured
- can we preview PDFs and images in the terminal?
- make sure that subnet and gateway are being logged properly. generally, look at the db and see what's up, is it missing important data?
- is_in_dotdir would need to be logged at query time i think. can't be done at feature gen time because what if FS changes
- maybe try random forests?
- implement directory selection

## not now, maybe never
- use tracing subscriber crate so we can have nice spans of time and we can maybe visualize and optimize idk
