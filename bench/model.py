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

    ./bench/model.py compare      # baseline binary vs current, same database

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


def evaluate(train_py, csv_path, schema_dir):
    """Rolling-origin folds over one feature set. Returns metrics and gains."""
    feature_names, binary_features, monotonicity = train_py.load_schema(schema_dir)
    df = train_py.load_data(str(csv_path))
    prepared = train_py.prepare_features(df, feature_names, binary_features, monotonicity)

    X, y, episodes = prepared.X, prepared.y, prepared.episodes
    weights = train_py.recency_weights(prepared.timestamps)
    params = train_py.make_params(prepared.monotone_constraints)

    folds = []
    for start in FOLD_STARTS:
        q_train = episodes.quantile(start)
        q_val = episodes.quantile(start + 0.10)
        q_test = episodes.quantile(start + 0.20)
        train = episodes <= q_train
        val = (episodes > q_train) & (episodes <= q_val)
        test = (episodes > q_val) & (episodes <= q_test)

        data = lgb.Dataset(X[train], label=y[train], weight=weights[train])
        valid = lgb.Dataset(X[val], label=y[val], weight=weights[val], reference=data)
        model = lgb.train(
            params, data, num_boost_round=1000, valid_sets=[valid],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        folds.append(score(model, X[test], y[test], episodes[test]))

    # Gains come from a fit on everything, which is what ships.
    full = lgb.train(params, lgb.Dataset(X, label=y, weight=weights),
                     num_boost_round=int(np.mean([f["rounds"] for f in folds])))
    gains = dict(zip(full.feature_name(), full.feature_importance("gain")))

    metrics = {key: float(np.mean([f[key] for f in folds])) for key in
               ("auc", "top1", "mrr", "rmse", "rounds")}
    metrics["episodes"] = int(sum(f["episodes"] for f in folds))
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

    return {
        "auc": roc_auc_score(labels, predictions),
        "top1": float(np.mean(top1)),
        "mrr": float(np.mean(reciprocal_rank)),
        "rmse": float(np.sqrt(((labels - predictions) ** 2).mean())),
        "rounds": model.best_iteration,
        "episodes": len(top1),
    }


def compare():
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
            results[name] = evaluate(train_py, csv_path, out_dir)

    (before, before_gains), (after, after_gains) = results["before"], results["after"]

    print(f"\nrolling-origin folds at {FOLD_STARTS}, "
          f"{before['episodes']} scored episodes\n")
    print(f"{'':<10}{'before':>10}{'after':>10}{'change':>10}")
    for key, label in (("auc", "AUC"), ("top1", "top-1"), ("mrr", "MRR"), ("rmse", "RMSE")):
        delta = after[key] - before[key]
        print(f"{label:<10}{before[key]:>10.4f}{after[key]:>10.4f}{delta:>+10.4f}")
    print(f"{'rounds':<10}{before['rounds']:>10.0f}{after['rounds']:>10.0f}")

    new = [name for name in after_gains if name not in before_gains]
    gone = [name for name in before_gains if name not in after_gains]
    if new or gone:
        print()
    for name in new:
        total = sum(after_gains.values()) or 1.0
        rank = sorted(after_gains, key=after_gains.get, reverse=True).index(name) + 1
        print(f"new feature {name!r}: gain {after_gains[name]:.0f} "
              f"({after_gains[name] / total * 100:.1f}% of total, rank {rank} of {len(after_gains)})")
    for name in gone:
        print(f"removed feature {name!r}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare()
    else:
        print(__doc__)
