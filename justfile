# Run psychic in release mode
run *ARGS:
    cargo run --release -- {{ARGS}}

# Build release binary
build:
    cargo build --release && cargo doc

# Run tests
test:
    cargo test

# Run clippy linter
lint:
    cargo clippy

# Analyze performance of latest psychic run
analyze-perf:
    @cargo run --release -- internal analyze-perf

# Clear performance logs and run fresh measurement
measure-perf *ARGS:
    @echo "Clearing old logs..."
    @> ~/.local/share/psychic/app.log
    @echo "Running psychic..."
    @cargo run --release -- {{ARGS}} || true
    @sleep 0.5
    @echo ""
    @cargo run --release -- internal analyze-perf
