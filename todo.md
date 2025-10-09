## notes on performance

On 5000 files, `update_filtered_files` takes 150-200ms. All of that comes from `rank_files`. The majority of THAT is from feature computation, not actually from running the model.

Feature computation can be sped up in a bunch of ways:
* Divide features into those that depend on the query and those that don't. The ones that don't depend on query don't need to be recomputed on query change.
* Two of them iterate over clicks_by_file. We can probably move parent dir click aggregation, and last 30 days click counting, into a precomputation that happens once at db load time and computes click counts by iterating over all clicks once. Then the features just do lookups? Idk how to do this elegantly though, where adding each feature doesn't become a big chore at db load time. Ah! Maybe each feature also gets to define its own Agg type that can do stuff at db load time? 

## now

- scrolling results is broken after the async refactor
- monotonicity constraints on certain obvious features
- maybe add a slight linear term?
- try fitting a linear or logistic regressor esp on modified time and num clicks and last time clicked
- hit enter to open Preview or whatever default thing is configured
- can we preview PDFs and images in the terminal?
- make sure that subnet and gateway are being logged properly. generally, look at the db and see what's up, is it missing important data?
- is_in_dotdir would need to be logged at query time i think. can't be done at feature gen time because what if FS changes
- implement the streaming async plan for updating ui and sorting etc in a different thread
- but also, measure which parts are slow. is it model? sorting? getting mtime? computing features? want aggregate cost of each of those.
- maybe try random forests?
- implement directory selection
- things go kinda wrong when i call it from my root homedir. too many files. it freezes. now it no longer freezes but typing is laggy

## not now, maybe never
- use tracing subscriber crate so we can have nice spans of time and we can maybe visualize and optimize idk
