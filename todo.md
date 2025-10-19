## notes on performance

On 5000 files, `update_filtered_files` takes 150-200ms. All of that comes from `rank_files`. The majority of THAT is from feature computation, not actually from running the model.

Update: with some optimizations, model is actually more expensive than computing features.

Feature computation can be sped up in a bunch of ways:
* Divide features into those that depend on the query and those that don't. The ones that don't depend on query don't need to be recomputed on query change.
* Two of them iterate over clicks_by_file. We can probably move parent dir click aggregation, and last 30 days click counting, into a precomputation that happens once at db load time and computes click counts by iterating over all clicks once. Then the features just do lookups? Idk how to do this elegantly though, where adding each feature doesn't become a big chore at db load time. Ah! Maybe each feature also gets to define its own Agg type that can do stuff at db load time? 

Helix file picker is a lot faster. Maybe implement a mode that doesn't sort or run the model or compute features, to make sure it's somewhat as fast.

## now

- scrolling results is broken after the async refactor
- buggy log messages printed to console when exiting
- get rid of calls to get_file_metadata in main.rs
- get_file_metadata should return a struct, not a tuple
- implement Ctrl-p and Ctrl-n for up and down
- pick some good keybindings for going to top, and paging up and down the results
- watch the cwd + all historical files; if mtime changes then update. more generally, our internal file data structure must be kept up-to-date with the filesystem. Right now this works because the file list view polls the filesytem for file metadata every frame or something awful like that. But the fixes below will break that.
- deduplicate code in up-down scrolling in main.rs
- file walker should probably just send mtime as given by walkdir API. and make the type it sends not a tuple. include file size and atime in there.
- watch the cwd. if new files added then add them.
- monotonicity constraints on certain obvious features
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
