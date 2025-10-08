## now

- make sure that subnet and gateway are being logged properly. generally, look at the db and see what's up, is it missing important data?
- is_in_dotdir would need to be logged at query time i think. can't be done at feature gen time because what if FS changes
- implement the streaming async plan for updating ui and sorting etc in a different thread
- but also, measure which parts are slow. is it model? sorting? getting mtime? computing features? want aggregate cost of each of those.
- maybe try random forests?
- implement directory selection
- things go kinda wrong when i call it from my root homedir. too many files. it freezes. now it no longer freezes but typing is laggy

## not now, maybe never
- use tracing subscriber crate so we can have nice spans of time and we can maybe visualize and optimize idk
