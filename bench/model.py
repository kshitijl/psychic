#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "lightgbm>=4.6.0",
#   "numpy>=1.26",
#   "pandas>=2.3.3",
#   "scikit-learn>=1.7.2",
#   # train.py is imported for its parameters, so its imports have to resolve.
#   "matplotlib>=3.10.6",
#   "seaborn>=0.13.2",
#   "shap>=0.48.0",
# ]
# ///
"""What a ranking change did to ranking quality.

    ./bench/model.py compare [rounds] [--seeds N]   # baseline vs current
    ./bench/model.py objectives   # two training objectives, one feature set
    ./bench/model.py seeds 8      # one feature set, N seeds: the noise floor

Speed benchmarks cannot see a feature: it changes what the model predicts, not
how fast it predicts it. This trains both versions the way train.py does and
scores them the way the user experiences them - did the file you opened come out
on top - over rolling-origin folds, so the model is always predicting the future
from the past.

Both feature sets are generated from one copy of the real events.db, so the only
difference between the two runs is the code that computed them.

The LightGBM parameters and the recency weighting are imported from train.py
rather than restated, because a benchmark that has drifted from what ships is
worse than no benchmark.
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
WORK = Path("/tmp/psychic-bench")
DEFAULT_DATA_DIR = Path.home() / ".local/share/psychic"

# The folds: train on everything up to `start`, early-stop on the next tenth,
# score the tenth after that. Three of them, so one lucky fortnight cannot
# decide whether a feature was worth adding.
FOLD_STARTS = (0.60, 0.70, 0.80)


def load_train_py():
    spec = importlib.util.spec_from_file_location("train_py", REPO / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_features(binary, out_dir):
    """Run `psychic generate-features` against a private copy of the database."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["sqlite3", str(DEFAULT_DATA_DIR / "events.db"), f".backup '{out_dir}/events.db'"],
        check=True,
    )
    subprocess.run(
        [str(binary), "generate-features", "--data-dir", str(out_dir)],
        check=True, capture_output=True,
    )
    return out_dir / "features.csv"


def evaluate(train_py, csv_path, schema_dir, fixed_rounds=None, params_override=None):
    """Rolling-origin folds over one feature set. Returns metrics and gains.

    With `fixed_rounds`, both sides train for the same number of rounds and
    early stopping is off. Early stopping watches pooled validation AUC, so a
    feature that moves that curve changes how long training runs - and then the
    comparison is partly two differently sized models rather than the feature.
    """
    feature_names, binary_features, monotonicity = train_py.load_schema(schema_dir)
    df = train_py.load_data(str(csv_path))
    prepared = train_py.prepare_features(df, feature_names, binary_features, monotonicity)

    X, y, episodes = prepared.X, prepared.y, prepared.episodes
    weights = train_py.recency_weights(prepared.timestamps)
    params = train_py.make_params(prepared.monotone_constraints)
    if params_override:
        params = {**params, **params_override}
        # A pooled objective wants no groups; a ranking one cannot do without.
        params.pop("eval_at", None) if params["objective"] != "lambdarank" else None

    folds = []
    for start in FOLD_STARTS:
        q_train = episodes.quantile(start)
        q_val = episodes.quantile(start + 0.10)
        q_test = episodes.quantile(start + 0.20)
        train = episodes <= q_train
        val = (episodes > q_train) & (episodes <= q_val)
        test = (episodes > q_val) & (episodes <= q_test)

        # Groups, for a ranking objective: which rows competed with each other.
        ranking = params["objective"] == "lambdarank"
        groups = train_py.group_sizes if ranking else (lambda _: None)
        data = lgb.Dataset(X[train], label=y[train], weight=weights[train],
                           group=groups(episodes[train]))
        valid = lgb.Dataset(X[val], label=y[val], weight=weights[val],
                            group=groups(episodes[val]), reference=data)
        if fixed_rounds:
            model = lgb.train(params, data, num_boost_round=fixed_rounds)
        else:
            model = lgb.train(
                params, data, num_boost_round=1000, valid_sets=[valid],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        folds.append(score(model, X[test], y[test], episodes[test]))

    # Gains come from a fit on everything, which is what ships.
    rounds = fixed_rounds or int(np.mean([f["rounds"] for f in folds]))
    full_groups = train_py.group_sizes(episodes) if params["objective"] == "lambdarank" else None
    full = lgb.train(params,
                     lgb.Dataset(X, label=y, weight=weights, group=full_groups),
                     num_boost_round=rounds)
    gains = dict(zip(full.feature_name(), full.feature_importance("gain")))

    metrics = {key: float(np.mean([f[key] for f in folds])) for key in
               ("auc", "top1", "mrr", "rmse", "rounds")}
    metrics["episodes"] = int(sum(f["episodes"] for f in folds))
    metrics["folds"] = folds
    return metrics, gains


def score(model, X, y, episodes):
    """AUC over rows; top-1 and MRR over episodes, which is what a user feels."""
    predictions = model.predict(X)
    labels, groups = y.values, episodes.values

    top1, reciprocal_rank = [], []
    for episode in np.unique(groups):
        rows = groups == episode
        ranked = labels[rows][np.argsort(-predictions[rows])]
        if ranked.max() != 1:
            continue                       # nothing was clicked in this episode
        first = int(np.argmax(ranked == 1)) + 1
        top1.append(first == 1)
        reciprocal_rank.append(1.0 / first)

    # AUC, top-1 and MRR are all rank-based and stay meaningful whatever scale
    # the objective emits. RMSE does not: against a 0/1 label it only means
    # something when the model outputs a probability.
    return {
        "auc": roc_auc_score(labels, predictions),
        "top1": float(np.mean(top1)),
        "mrr": float(np.mean(reciprocal_rank)),
        "rmse": float(np.sqrt(((labels - predictions) ** 2).mean())),
        "rounds": model.best_iteration,
        "episodes": len(top1),
    }


def seed_overrides(seed):
    return {
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
    }


def compare(fixed_rounds=None, seed_count=1):
    """Baseline binary against the current one, on one copy of the database.

    With `seed_count` above 1 each side is trained under that many seeds and the
    means are compared. Do that for anything smaller than a few points: bagging
    and feature sampling are random, adding a column perturbs which subsets each
    tree sees, and on this data one seed moves top-1 by up to 0.027 on its own -
    wider than most single features are worth.
    """
    train_py = load_train_py()
    baseline_binary = WORK / "old-src/target/release/psychic"
    current_binary = REPO / "target/release/psychic"
    if not baseline_binary.exists():
        sys.exit(f"No baseline binary at {baseline_binary}. Run: ./bench/run.py setup HEAD")

    results = {}
    with tempfile.TemporaryDirectory(prefix="psychic-model-") as tmp:
        for name, binary in (("before", baseline_binary), ("after", current_binary)):
            out_dir = Path(tmp) / name
            csv_path = generate_features(binary, out_dir)
            print(f"--- {name}: {binary} ---", flush=True)
            if seed_count > 1:
                runs = []
                for seed in range(1, seed_count + 1):
                    print(f"    seed {seed}", flush=True)
                    metrics, gains = evaluate(train_py, csv_path, out_dir, fixed_rounds,
                                              seed_overrides(seed))
                    runs.append(metrics)
                averaged = {key: float(np.mean([run[key] for run in runs]))
                            for key in ("auc", "top1", "mrr", "rmse", "rounds")}
                averaged["episodes"] = runs[0]["episodes"]
                averaged["folds"] = runs[0]["folds"]
                averaged["spread"] = {key: max(run[key] for run in runs)
                                      - min(run[key] for run in runs)
                                      for key in ("top1", "mrr")}
                averaged["runs"] = runs
                results[name] = (averaged, gains)
            else:
                results[name] = evaluate(train_py, csv_path, out_dir, fixed_rounds)

    (before, before_gains), (after, after_gains) = results["before"], results["after"]

    how = f", {fixed_rounds} rounds fixed" if fixed_rounds else ""
    if seed_count > 1:
        how += f", mean of {seed_count} seeds"
    print(f"\nrolling-origin folds at {FOLD_STARTS}{how}, "
          f"{before['episodes']} scored episodes\n")
    print(f"{'':<10}{'before':>10}{'after':>10}{'change':>10}")
    for key, label in (("auc", "AUC"), ("top1", "top-1"), ("mrr", "MRR"), ("rmse", "RMSE")):
        delta = after[key] - before[key]
        print(f"{label:<10}{before[key]:>10.4f}{after[key]:>10.4f}{delta:>+10.4f}")
    print(f"{'rounds':<10}{before['rounds']:>10.0f}{after['rounds']:>10.0f}")

    if seed_count > 1:
        # A change smaller than the seed spread is not evidence of anything.
        for key, label in (("top1", "top-1"), ("mrr", "MRR")):
            values_before = [run[key] for run in before["runs"]]
            values_after = [run[key] for run in after["runs"]]
            print(f"  {label} across seeds: before {min(values_before):.4f}-"
                  f"{max(values_before):.4f}, after {min(values_after):.4f}-"
                  f"{max(values_after):.4f}, "
                  f"sd of the mean {np.std(values_after) / np.sqrt(seed_count):.4f}")

    # Per fold as well as averaged. A mean that moves while the folds disagree
    # is one fold's luck, not a feature that works.
    print(f"\n{'fold':<10}{'top-1 before':>14}{'top-1 after':>13}{'rounds':>16}")
    for start, before_fold, after_fold in zip(FOLD_STARTS, before["folds"], after["folds"]):
        print(f"train<={start:<5.2f}{before_fold['top1']:>14.4f}{after_fold['top1']:>13.4f}"
              f"{before_fold['rounds']:>9.0f} ->{after_fold['rounds']:>4.0f}")

    new = [name for name in after_gains if name not in before_gains]
    gone = [name for name in before_gains if name not in after_gains]

    if not new and not gone:
        print("\nNote: both sides have the same features, and train.py is taken from the\n"
              "working tree for both arms - so a change to the training parameters cannot\n"
              "show up here. This mode compares feature sets. To compare parameters, train\n"
              "one feature set twice with different params, as `objectives` does.")
    if new or gone:
        print()
    for name in new:
        total = sum(after_gains.values()) or 1.0
        rank = sorted(after_gains, key=after_gains.get, reverse=True).index(name) + 1
        print(f"new feature {name!r}: gain {after_gains[name]:.0f} "
              f"({after_gains[name] / total * 100:.1f}% of total, rank {rank} of {len(after_gains)})")
    for name in gone:
        print(f"removed feature {name!r}")


def seeds(count):
    """The same feature set, trained under several seeds.

    This is the honest floor under every other number here. Bagging and feature
    sampling are random, and adding a column changes which subsets each tree
    sees - so part of any measured difference between two feature sets is the
    same perturbation a different seed would cause. If the spread here is as
    wide as the differences being acted on, those differences are not evidence.
    """
    train_py = load_train_py()

    with tempfile.TemporaryDirectory(prefix="psychic-seeds-") as tmp:
        out_dir = Path(tmp) / "current"
        csv_path = generate_features(REPO / "target/release/psychic", out_dir)
        runs = []
        for seed in range(1, count + 1):
            print(f"--- seed {seed} ---", flush=True)
            metrics, _ = evaluate(
                train_py, csv_path, out_dir,
                params_override={
                    "seed": seed,
                    "bagging_seed": seed,
                    "feature_fraction_seed": seed,
                    "data_random_seed": seed,
                },
            )
            runs.append(metrics)

    print(f"\n{count} seeds, one feature set, {runs[0]['episodes']} scored episodes\n")
    print(f"{'seed':<6}{'top-1':>10}{'MRR':>10}{'rounds':>9}")
    for seed, run in enumerate(runs, start=1):
        print(f"{seed:<6}{run['top1']:>10.4f}{run['mrr']:>10.4f}{run['rounds']:>9.0f}")

    for key, label in (("top1", "top-1"), ("mrr", "MRR")):
        values = [run[key] for run in runs]
        spread = max(values) - min(values)
        print(f"\n{label}: min {min(values):.4f}  max {max(values):.4f}  "
              f"spread {spread:.4f}  sd {np.std(values):.4f}")


def objectives():
    """The same features, trained two ways.

    The question this answers is whether the model is being trained on the
    question it is asked. lambdarank's gradient comes from swapping pairs within
    an episode; a pooled objective also learns the level of the scores, which
    nothing uses.

    RMSE is printed but means nothing for a ranking objective: it is a distance
    from a 0/1 label, and only a probability-shaped output lives on that scale.
    """
    train_py = load_train_py()
    binary = {"objective": "binary", "metric": "auc"}

    with tempfile.TemporaryDirectory(prefix="psychic-objective-") as tmp:
        out_dir = Path(tmp) / "current"
        csv_path = generate_features(REPO / "target/release/psychic", out_dir)
        print("--- pooled (binary) ---", flush=True)
        pooled, _ = evaluate(train_py, csv_path, out_dir, params_override=binary)
        print("--- ranking (lambdarank) ---", flush=True)
        ranking, _ = evaluate(train_py, csv_path, out_dir,
                              params_override={"objective": "lambdarank", "metric": "ndcg"})

    print(f"\nrolling-origin folds at {FOLD_STARTS}, "
          f"{pooled['episodes']} scored episodes, one feature set\n")
    print(f"{'':<10}{'binary':>10}{'lambdarank':>12}{'change':>10}")
    for key, label in (("auc", "AUC"), ("top1", "top-1"), ("mrr", "MRR")):
        print(f"{label:<10}{pooled[key]:>10.4f}{ranking[key]:>12.4f}"
              f"{ranking[key] - pooled[key]:>+10.4f}")
    print(f"{'rounds':<10}{pooled['rounds']:>10.0f}{ranking['rounds']:>12.0f}")
    print(f"\n{'fold':<10}{'binary':>10}{'lambdarank':>12}")
    for start, a, b in zip(FOLD_STARTS, pooled["folds"], ranking["folds"]):
        print(f"train<={start:<4.2f}{a['top1']:>10.4f}{b['top1']:>12.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # ./bench/model.py compare [rounds]
        args = sys.argv[2:]
        seed_count = 1
        if "--seeds" in args:
            index = args.index("--seeds")
            seed_count = int(args[index + 1])
            args = args[:index] + args[index + 2:]
        compare(int(args[0]) if args else None, seed_count)
    elif len(sys.argv) > 1 and sys.argv[1] == "objectives":
        objectives()
    elif len(sys.argv) > 1 and sys.argv[1] == "seeds":
        seeds(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    else:
        print(__doc__)
