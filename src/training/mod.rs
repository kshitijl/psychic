//! Training the ranking model, in-process.
//!
//! This used to be `train.py`, run through `uv`, which made Python and a
//! network connection runtime dependencies of a terminal file browser. It is
//! now a port of that script onto the LightGBM that is already linked into the
//! binary for prediction. `train.py` stays in the repo as the reference: it is
//! what `bench/train_parity.py` checks this against, and it still draws the
//! plots.
//!
//! The port is meant to be exact, not approximate. Every step below says which
//! part of `train.py` (or of pandas, numpy or the LightGBM Python package) it
//! reproduces, because a step that is only nearly the same trains a different
//! model and nothing errors.
//!
//! The interface is [`train`]: a features CSV in, `model.txt` and
//! `model_stats.json` out.

mod lgbm;

use anyhow::{Context, Result, bail};
use rustc_hash::FxHashMap;
use std::fmt;
use std::io::Read;
use std::ops::Range;
use std::path::Path;
use std::time::Instant;

/// How long it takes for a row to count half as much as a fresh one. See
/// `HALF_LIFE_DAYS` in `train.py` for the measurements behind 180.
const HALF_LIFE_DAYS: f64 = 180.0;

/// Upper bound on trees in the validated fit; early stopping ends it long before.
const MAX_ROUNDS: usize = 1000;

/// Stop once validation has not improved for this many rounds.
const STOPPING_ROUNDS: usize = 50;

/// One model input: its column name and the monotone constraint LightGBM
/// should hold it to (1 increasing, -1 decreasing, 0 none).
#[derive(Debug, Clone, Copy)]
pub struct FeatureSpec<'a> {
    pub name: &'a str,
    pub monotone: i8,
}

/// The features this build computes, in model order.
pub fn registry_features() -> Vec<FeatureSpec<'static>> {
    crate::feature_defs::FEATURE_REGISTRY
        .iter()
        .map(|f| FeatureSpec {
            name: f.name(),
            monotone: match f.monotonicity() {
                None => 0,
                Some(crate::feature_defs::Monotonicity::Increasing) => 1,
                Some(crate::feature_defs::Monotonicity::Decreasing) => -1,
            },
        })
        .collect()
}

/// The rows training learns from: every episode with a click in it, grouped
/// by episode, in episode order.
///
/// This is `load_data` and `prepare_features` from `train.py`. Once parsed,
/// the invariants hold by construction: rows are sorted by episode, and every
/// episode has at least one positive.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingSet {
    /// Row-major, `labels.len()` rows of `num_features` columns. A blank cell
    /// is NaN, which is what pandas reads it as.
    features: Vec<f64>,
    num_features: usize,
    labels: Vec<f32>,
    episodes: Vec<i64>,
    timestamps: Vec<f64>,
}

impl TrainingSet {
    /// Parse a features CSV, as `generate_features` writes it.
    pub fn from_csv(reader: impl Read, features: &[FeatureSpec]) -> Result<Self> {
        let mut csv = csv::Reader::from_reader(reader);
        let headers = csv.headers()?.clone();
        let column = |name: &str| {
            headers
                .iter()
                .position(|h| h == name)
                .with_context(|| format!("features CSV has no `{name}` column"))
        };
        let label_col = column("label")?;
        let episode_col = column("episode_id")?;
        let timestamp_col = column("timestamp")?;
        let feature_cols: Vec<usize> = features
            .iter()
            .map(|f| column(f.name))
            .collect::<Result<_>>()?;

        let mut rows: Vec<ParsedRow> = Vec::new();
        for (line, record) in csv.records().enumerate() {
            let record = record?;
            let field = |col: usize| record.get(col).unwrap_or("");
            let context = || format!("features CSV row {}", line + 2);
            let label = match field(label_col) {
                "0" => 0.0,
                "1" => 1.0,
                other => bail!("{}: label must be 0 or 1, got {other:?}", context()),
            };
            let episode: i64 = field(episode_col).parse().with_context(context)?;
            let timestamp = field(timestamp_col).parse::<i64>().with_context(context)? as f64;
            let values = feature_cols
                .iter()
                .map(|&col| match field(col) {
                    "" => Ok(f64::NAN),
                    text => text.parse::<f64>(),
                })
                .collect::<Result<Vec<f64>, _>>()
                .with_context(context)?;
            rows.push(ParsedRow {
                label,
                episode,
                timestamp,
                values,
            });
        }

        // lambdarank learns from the order within an episode, which needs
        // something clicked to order. `load_data` drops the rest.
        let mut positives: FxHashMap<i64, f32> = FxHashMap::default();
        for row in &rows {
            *positives.entry(row.episode).or_default() += row.label;
        }
        rows.retain(|row| positives[&row.episode] > 0.0);

        // Stable, like `sort_values(kind="stable")`: within an episode the
        // rows keep their CSV order, which decides how LightGBM breaks ties.
        rows.sort_by_key(|row| row.episode);

        let mut set = TrainingSet {
            features: Vec::with_capacity(rows.len() * features.len()),
            num_features: features.len(),
            labels: Vec::with_capacity(rows.len()),
            episodes: Vec::with_capacity(rows.len()),
            timestamps: Vec::with_capacity(rows.len()),
        };
        for row in rows {
            set.features.extend_from_slice(&row.values);
            set.labels.push(row.label);
            set.episodes.push(row.episode);
            set.timestamps.push(row.timestamp);
        }
        Ok(set)
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn positives(&self) -> usize {
        self.labels.iter().filter(|&&l| l == 1.0).count()
    }

    fn row_features(&self, rows: Range<usize>) -> &[f64] {
        &self.features[rows.start * self.num_features..rows.end * self.num_features]
    }
}

struct ParsedRow {
    label: f32,
    episode: i64,
    timestamp: f64,
    values: Vec<f64>,
}

/// The LightGBM parameters, as `make_params` in `train.py` has them, plus the
/// round count that `lgb.train` adds from its `num_boost_round` argument.
///
/// The string is what LightGBM parses; key order does not matter to it.
fn lightgbm_params(features: &[FeatureSpec], num_iterations: usize) -> String {
    let monotone: Vec<String> = features.iter().map(|f| f.monotone.to_string()).collect();
    [
        "objective=lambdarank".to_owned(),
        "metric=ndcg".to_owned(),
        "eval_at=1,5".to_owned(),
        "lambdarank_truncation_level=30".to_owned(),
        format!("monotone_constraints={}", monotone.join(",")),
        "boosting_type=gbdt".to_owned(),
        "num_leaves=15".to_owned(),
        "learning_rate=0.1".to_owned(),
        "feature_fraction=0.9".to_owned(),
        "bagging_fraction=0.8".to_owned(),
        "bagging_freq=5".to_owned(),
        "verbose=-1".to_owned(),
        "seed=42".to_owned(),
        "bagging_seed=42".to_owned(),
        "feature_fraction_seed=42".to_owned(),
        "data_random_seed=42".to_owned(),
        format!("num_iterations={num_iterations}"),
    ]
    .join(" ")
}

/// Weight each row by age, halving every `HALF_LIFE_DAYS`, measured from the
/// newest row. `recency_weights` in `train.py`.
fn recency_weights(timestamps: &[f64]) -> Vec<f64> {
    assert!(!timestamps.is_empty(), "there must be rows to weight");
    let newest = timestamps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    timestamps
        .iter()
        .map(|&t| {
            let age_days = (newest - t) / 86400.0;
            0.5f64.powf(age_days / HALF_LIFE_DAYS)
        })
        .collect()
}

/// The `q` quantile of sorted values, exactly as `pandas.Series.quantile`
/// computes it - which is numpy's default "linear" method, down to the
/// rounding in its interpolation. The split boundaries come from this, so an
/// almost-equal formula could move an episode across one.
fn numpy_linear_quantile(sorted: &[i64], q: f64) -> f64 {
    assert!(!sorted.is_empty(), "a quantile of nothing");
    assert!(sorted.is_sorted(), "the values must be sorted");
    let n = sorted.len();
    let virtual_index = (n - 1) as f64 * q;
    if virtual_index >= (n - 1) as f64 {
        return sorted[n - 1] as f64;
    }
    let previous = virtual_index.floor();
    let gamma = virtual_index - previous;
    let a = sorted[previous as usize] as f64;
    let b = sorted[previous as usize + 1] as f64;
    // numpy's `_lerp`, which interpolates from whichever end is nearer.
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// Train, validation and test rows: the first 80% of episodes, the next 10%,
/// the last 10%. `time_split` in `train.py`.
///
/// Rows are sorted by episode and episode ids are handed out in time order, so
/// each part is a contiguous range, and every episode lands whole in one.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Split {
    train: Range<usize>,
    validation: Range<usize>,
    test: Range<usize>,
}

fn time_split(episodes: &[i64]) -> Result<Split> {
    let q80 = numpy_linear_quantile(episodes, 0.8);
    let q90 = numpy_linear_quantile(episodes, 0.9);
    let train_end = episodes.partition_point(|&e| (e as f64) <= q80);
    let validation_end = episodes.partition_point(|&e| (e as f64) <= q90);
    let split = Split {
        train: 0..train_end,
        validation: train_end..validation_end,
        test: validation_end..episodes.len(),
    };
    if split.train.is_empty() || split.validation.is_empty() || split.test.is_empty() {
        bail!(
            "every split needs rows; episode ids {}..{} gave {}/{}/{}",
            episodes.first().unwrap_or(&0),
            episodes.last().unwrap_or(&0),
            split.train.len(),
            split.validation.len(),
            split.test.len()
        );
    }
    Ok(split)
}

/// Rows per episode, in order. Rows of an episode are contiguous because the
/// set is sorted by episode.
fn group_sizes(episodes: &[i64]) -> Vec<i32> {
    assert!(episodes.is_sorted(), "rows must be grouped by episode");
    episodes
        .chunk_by(|a, b| a == b)
        .map(|group| i32::try_from(group.len()).expect("an episode fits in i32 rows"))
        .collect()
}

/// Python LightGBM's `early_stopping` callback, for one validation set.
///
/// Each metric keeps its best score and the iteration it was reached at; a
/// score counts as better only if strictly higher. Training stops as soon as
/// any metric has gone `STOPPING_ROUNDS` iterations without improving, or at
/// the last iteration, and the answer is that metric's best iteration. Metrics
/// are checked in order, so the first metric wins ties.
#[derive(Debug)]
struct EarlyStopping {
    /// (best score, 0-based iteration it was reached at), per metric.
    best: Vec<(f64, usize)>,
}

impl EarlyStopping {
    fn new() -> Self {
        EarlyStopping { best: Vec::new() }
    }

    /// Record iteration `iteration`'s validation scores. Returns the number
    /// of rounds to keep if training should stop here.
    fn observe(&mut self, iteration: usize, scores: &[f64]) -> Option<usize> {
        assert!(!scores.is_empty(), "early stopping needs a metric");
        let first_time = self.best.is_empty();
        if first_time {
            self.best = scores.iter().map(|&s| (s, iteration)).collect();
        }
        assert_eq!(
            self.best.len(),
            scores.len(),
            "the same metrics every round"
        );
        for (best, &score) in self.best.iter_mut().zip(scores) {
            if !first_time && score > best.0 {
                *best = (score, iteration);
            }
            if iteration - best.1 >= STOPPING_ROUNDS || iteration == MAX_ROUNDS - 1 {
                return Some(best.1 + 1);
            }
        }
        None
    }
}

/// The rows of one dataset, gathered into the shapes LightGBM takes.
struct Part {
    labels: Vec<f32>,
    weights: Vec<f32>,
    groups: Vec<i32>,
}

impl Part {
    fn new(set: &TrainingSet, weights: &[f64], rows: Range<usize>) -> Self {
        Part {
            labels: set.labels[rows.clone()].to_vec(),
            // Python hands LightGBM float32 weights, cast from these float64s.
            weights: weights[rows.clone()].iter().map(|&w| w as f32).collect(),
            groups: group_sizes(&set.episodes[rows]),
        }
    }
}

/// What came out of fitting: the model to ship and what it was measured at.
#[derive(Debug, Clone)]
pub struct Fitted {
    /// The model, in LightGBM's text format.
    pub model: String,
    /// Trees in the shipped model: where the validated fit stopped improving.
    pub best_iteration: usize,
    /// NDCG@1 and NDCG@5 on the validation set at `best_iteration`.
    pub validation_ndcg: Vec<f64>,
    /// Total split gain per feature in the shipped model, in feature order.
    pub gain_importance: Vec<f64>,
    pub train_rows: usize,
    pub validation_rows: usize,
    pub test_rows: usize,
}

/// Fit the model: find the round count on a time split, then refit on every
/// row for that many rounds. `train_model` and `refit_on_everything` in
/// `train.py`.
///
/// The shipped model has seen the most recent fortnight, which is the part
/// most like what the user is about to search for; the held-out fit only
/// decides how many trees it gets.
pub fn fit(set: &TrainingSet, features: &[FeatureSpec]) -> Result<Fitted> {
    assert_eq!(
        set.num_features,
        features.len(),
        "the set has these features"
    );
    let names: Vec<&str> = features.iter().map(|f| f.name).collect();
    let weights = recency_weights(&set.timestamps);
    let split = time_split(&set.episodes)?;

    let dataset = |part: &Part, rows: Range<usize>, params: &str, reference| {
        lgbm::Dataset::new(
            &lgbm::DatasetInput {
                features: set.row_features(rows),
                num_features: set.num_features,
                labels: &part.labels,
                weights: &part.weights,
                group_sizes: &part.groups,
                feature_names: &names,
            },
            params,
            reference,
        )
    };

    // The validated fit.
    let params = lightgbm_params(features, MAX_ROUNDS);
    let train_part = Part::new(set, &weights, split.train.clone());
    let validation_part = Part::new(set, &weights, split.validation.clone());
    let train_data = dataset(&train_part, split.train.clone(), &params, None)?;
    let validation_data = dataset(
        &validation_part,
        split.validation.clone(),
        &params,
        Some(&train_data),
    )?;
    let mut booster = lgbm::Booster::new(&train_data, &params, set.num_features)?;
    booster.add_valid(&validation_data)?;

    let mut stopping = EarlyStopping::new();
    let mut history: Vec<Vec<f64>> = Vec::new();
    let mut best_iteration = None;
    for iteration in 0..MAX_ROUNDS {
        booster.update_one_iter()?;
        let scores = booster.eval(1)?;
        let stop = stopping.observe(iteration, &scores);
        history.push(scores);
        if stop.is_some() {
            best_iteration = stop;
            break;
        }
    }
    let best_iteration = best_iteration.expect("early stopping always stops by the last round");
    let validation_ndcg = history[best_iteration - 1].clone();
    drop(booster);

    // The refit, on everything, with nothing held out to stop against.
    let params = lightgbm_params(features, best_iteration);
    let all = 0..set.len();
    let all_part = Part::new(set, &weights, all.clone());
    let all_data = dataset(&all_part, all, &params, None)?;
    let mut booster = lgbm::Booster::new(&all_data, &params, set.num_features)?;
    for _ in 0..best_iteration {
        booster.update_one_iter()?;
    }

    Ok(Fitted {
        model: booster.save_model_to_string()?,
        best_iteration,
        validation_ndcg,
        gain_importance: booster.gain_importance()?,
        train_rows: split.train.len(),
        validation_rows: split.validation.len(),
        test_rows: split.test.len(),
    })
}

/// What a training run did, for the training log.
#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub rows: usize,
    pub positives: usize,
    pub fitted_summary: FittedSummary,
    pub seconds: f64,
}

#[derive(Debug, Clone)]
pub struct FittedSummary {
    pub best_iteration: usize,
    pub validation_ndcg: Vec<f64>,
    pub train_rows: usize,
    pub validation_rows: usize,
    pub test_rows: usize,
}

impl fmt::Display for TrainingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.fitted_summary;
        writeln!(
            f,
            "Trained on {} rows ({} clicked): train {}, validation {}, test {}",
            self.rows, self.positives, s.train_rows, s.validation_rows, s.test_rows
        )?;
        let ndcg: Vec<String> = s
            .validation_ndcg
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect();
        writeln!(
            f,
            "Best iteration {}, validation NDCG@1,5 {}",
            s.best_iteration,
            ndcg.join(", ")
        )?;
        write!(f, "Took {:.2}s", self.seconds)
    }
}

/// Train on `features_csv` and write `model.txt` and `model_stats.json` into
/// `data_dir`, each atomically: the running TUI reloads them on its own
/// schedule and must never read half a file.
pub fn train(features_csv: &Path, data_dir: &Path) -> Result<TrainingReport> {
    let start = Instant::now();
    let features = registry_features();
    let file = std::fs::File::open(features_csv)
        .with_context(|| format!("Failed to open {}", features_csv.display()))?;
    let set = TrainingSet::from_csv(std::io::BufReader::new(file), &features)?;
    let fitted = fit(&set, &features)?;
    let seconds = start.elapsed().as_secs_f64();

    atomic_write(&data_dir.join("model.txt"), fitted.model.as_bytes())?;

    let mut ranked: Vec<crate::ranker::FeatureImportance> = features
        .iter()
        .zip(&fitted.gain_importance)
        .map(|(f, &importance)| crate::ranker::FeatureImportance {
            feature: f.name.to_owned(),
            importance,
        })
        .collect();
    ranked.sort_by(|a, b| b.importance.total_cmp(&a.importance));
    ranked.truncate(3);
    let stats = crate::ranker::ModelStats {
        trained_at: jiff::Timestamp::now().to_string(),
        training_duration_seconds: (seconds * 100.0).round() / 100.0,
        num_features: features.len(),
        num_total_examples: set.len(),
        num_positive_examples: set.positives(),
        num_negative_examples: set.len() - set.positives(),
        best_iteration: Some(fitted.best_iteration),
        top_3_features: ranked,
    };
    atomic_write(
        &data_dir.join("model_stats.json"),
        serde_json::to_string_pretty(&stats)?.as_bytes(),
    )?;

    Ok(TrainingReport {
        rows: set.len(),
        positives: set.positives(),
        fitted_summary: FittedSummary {
            best_iteration: fitted.best_iteration,
            validation_ndcg: fitted.validation_ndcg,
            train_rows: fitted.train_rows,
            validation_rows: fitted.validation_rows,
            test_rows: fitted.test_rows,
        },
        seconds,
    })
}

/// Write a temp file beside `target` and rename it over: on POSIX a reader
/// sees the whole old file or the whole new one, never a truncated one.
fn atomic_write(target: &Path, contents: &[u8]) -> Result<()> {
    let dir = target.parent().context("target has a directory")?;
    std::fs::create_dir_all(dir)?;
    let name = target.file_name().context("target has a name")?;
    let tmp = dir.join(format!(
        "{}.{}.tmp",
        name.to_string_lossy(),
        std::process::id()
    ));
    std::fs::write(&tmp, contents).with_context(|| format!("Failed to write {}", tmp.display()))?;
    std::fs::rename(&tmp, target)
        .with_context(|| format!("Failed to move {} into place", target.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO: [FeatureSpec<'static>; 2] = [
        FeatureSpec {
            name: "a",
            monotone: 0,
        },
        FeatureSpec {
            name: "b",
            monotone: 1,
        },
    ];

    #[test]
    fn test_from_csv_drops_unclicked_episodes_and_sorts_stably() {
        let csv = "\
label,episode_id,timestamp,query,b,a
0,7,100,x,1,10
1,3,90,x,2,20
0,5,95,x,3,30
1,7,101,x,4,
0,3,91,x,5,50
";
        let set = TrainingSet::from_csv(csv.as_bytes(), &TWO).unwrap();
        // Episode 5 had no click. Episode 3 comes first, its rows in CSV order.
        assert_eq!(set.episodes, vec![3, 3, 7, 7]);
        assert_eq!(set.labels, vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(set.timestamps, vec![90.0, 91.0, 100.0, 101.0]);
        // Columns come out in feature order, not CSV order; a blank is NaN.
        assert_eq!(&set.features[..6], &[20.0, 2.0, 50.0, 5.0, 10.0, 1.0]);
        assert!(set.features[6].is_nan());
        assert_eq!(set.features[7], 4.0);
    }

    #[test]
    fn test_from_csv_rejects_a_missing_feature() {
        let csv = "label,episode_id,timestamp,a\n1,1,1,1\n";
        let err = TrainingSet::from_csv(csv.as_bytes(), &TWO).unwrap_err();
        assert_eq!(err.to_string(), "features CSV has no `b` column");
    }

    #[test]
    fn test_numpy_linear_quantile() {
        // Values checked against `pd.Series(values).quantile(q)`.
        assert_eq!(
            numpy_linear_quantile(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.8),
            8.2
        );
        assert_eq!(
            numpy_linear_quantile(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9),
            9.1
        );
        assert_eq!(numpy_linear_quantile(&[1, 1, 1, 2, 2, 9], 0.8), 2.0);
        assert_eq!(numpy_linear_quantile(&[4], 0.9), 4.0);
    }

    #[test]
    fn test_time_split() {
        // Twenty rows, one per episode: q80 is 16.200000000000003 and q90 18.1.
        let episodes: Vec<i64> = (1..=20).collect();
        assert_eq!(
            time_split(&episodes).unwrap(),
            Split {
                train: 0..16,
                validation: 16..18,
                test: 18..20,
            }
        );
        // An episode is never cut in two, so one long episode can swallow
        // the validation set whole, and that is an error.
        let episodes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2];
        assert_eq!(
            time_split(&episodes).unwrap_err().to_string(),
            "every split needs rows; episode ids 1..2 gave 9/0/1"
        );
    }

    #[test]
    fn test_group_sizes() {
        assert_eq!(group_sizes(&[1, 1, 1, 4, 9, 9]), vec![3, 1, 2]);
    }

    #[test]
    fn test_recency_weights_halve_every_half_life() {
        let day = 86400.0;
        assert_eq!(
            recency_weights(&[0.0, 180.0 * day, 360.0 * day]),
            vec![0.25, 0.5, 1.0]
        );
    }

    #[test]
    fn test_early_stopping_waits_for_every_metric() {
        let mut stopping = EarlyStopping::new();
        // Metric 0 peaks at iteration 2; metric 1 keeps improving until 30.
        let mut stopped = None;
        for i in 0..MAX_ROUNDS {
            let first = if i <= 2 { i as f64 } else { 2.0 };
            let second = if i <= 30 { i as f64 } else { 30.0 };
            stopped = stopping.observe(i, &[first, second]);
            if stopped.is_some() {
                assert_eq!(i, 52, "metric 0 runs out of patience first");
                break;
            }
        }
        assert_eq!(stopped, Some(3));
    }

    #[test]
    fn test_early_stopping_ties_do_not_count_as_improvement() {
        let mut stopping = EarlyStopping::new();
        let stopped = (0..MAX_ROUNDS).find_map(|i| stopping.observe(i, &[0.5]));
        assert_eq!(stopped, Some(1));
    }

    #[test]
    fn test_early_stopping_at_the_last_round() {
        let mut stopping = EarlyStopping::new();
        let stopped = (0..MAX_ROUNDS).find_map(|i| stopping.observe(i, &[i as f64]));
        assert_eq!(stopped, Some(MAX_ROUNDS));
    }
}
