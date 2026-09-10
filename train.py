#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "lightgbm>=4.6.0",
#   "matplotlib>=3.10.6",
#   "numpy>=1.26",
#   "pandas>=2.3.3",
#   "scikit-learn>=1.7.2",
#   "seaborn>=0.13.2",
#   "shap>=0.48.0",
# ]
# ///
"""
Train LightGBM ranking model on features generated from events.db, and generate evaluation visualizations.

Usage:
    python train.py features.csv output_prefix [--data-dir DIR]
    uv run train.py features.csv output_prefix [--data-dir DIR]

This is a PEP 723 script: the block above lists everything it needs, so
`uv run train.py` builds its own environment on any machine with uv, with no
virtualenv to set up and no dependence on the psychic checkout. That matters
because psychic embeds this file, writes it into the data directory, and runs it
from whatever directory the user happened to launch from - which may be an
unrelated project with its own pyproject.toml. Inline metadata takes precedence
over a surrounding project, so that case works too.

Every module imported below is listed above, including ones that would otherwise
arrive transitively (numpy), so a resolver change upstream cannot break this.
"""

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import shap
import json
from pathlib import Path
import os
import tempfile
import argparse
from typing import NamedTuple

sns.set_style("whitegrid")

# How long it takes for a row to count half as much as a fresh one.
#
# Measured, this costs a little: over three rolling-origin folds the sweep runs
# uniform 0.703 top-1, 365d 0.704, 180d 0.685, 120d 0.679, 60d 0.671, 30d 0.640 -
# monotone, the gentler the better, because recency is already carried by
# clicks_last_hour through clicks_last_30_days and decay only removes rows. With
# ~1.2k clicks in the whole history, positives are the scarce thing.
#
# 180 days is the deliberate trade: it keeps 83% of the effective rows and buys
# insurance against what the metrics cannot see yet - a week of unusual activity
# that would otherwise keep its full vote forever, and features prone to
# memorisation, whose grip on stale rows decays on its own. Raise this to 1000 to
# get uniform weighting back.
HALF_LIFE_DAYS = 180


def load_schema(data_dir):
    """Load feature schema from data directory."""
    import hashlib

    schema_path = Path(data_dir) / "feature_schema.json"
    if not schema_path.exists():
        print(f"Error: feature_schema.json not found at {schema_path}")
        print("Run: cargo run --release -- generate-features")
        print("This will generate both features.csv and feature_schema.json")
        sys.exit(1)

    # Hash the schema file to verify it's the same across runs
    with open(schema_path, "rb") as f:
        schema_hash = hashlib.md5(f.read()).hexdigest()[:8]

    with open(schema_path) as f:
        schema = json.load(f)

    # Extract feature names, types, and monotonicity from schema
    feature_names = [f["name"] for f in schema["features"]]
    binary_features = [f["name"] for f in schema["features"] if f["type"] == "binary"]
    monotonicity_map = {f["name"]: f.get("monotonicity") for f in schema["features"]}

    print(f"Loaded feature schema: {len(feature_names)} features")
    print(f"  Schema file hash: {schema_hash}")

    return feature_names, binary_features, monotonicity_map


def load_data(csv_path):
    """Load features CSV and prepare for training."""
    import hashlib

    # Hash the CSV file to verify it's the same across runs
    with open(csv_path, "rb") as f:
        csv_hash = hashlib.md5(f.read()).hexdigest()[:8]

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"  CSV file hash: {csv_hash}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(
        f"\nFeatures: {[c for c in df.columns if c not in ['label', 'episode_id', 'subsession_id', 'session_id', 'timestamp']]}"
    )

    # Use episode_id for LambdaRank grouping (each episode spans from one action to the next)
    df["episode"] = df["episode_id"].astype(int)

    # Filter out episodes with no positive labels (no clicks/scrolls)
    # LambdaRank needs at least one positive example per episode
    episodes_with_positives = df.groupby("episode")["label"].sum()
    valid_episodes = episodes_with_positives[episodes_with_positives > 0].index

    original_samples = len(df)
    original_episodes = df["episode"].nunique()

    df = df[df["episode"].isin(valid_episodes)]

    # Sort by episode. A ranking objective is handed group *sizes*, not group
    # ids, so it reads them off consecutive rows: rows of one episode have to be
    # together and the episodes in order. They very nearly are already - episode
    # ids are handed out in one pass over time-sorted events - but the
    # accumulator flushes whatever impressions are still pending at the end out
    # of a hash map, which is enough to break it.
    df = df.sort_values("episode", kind="stable").reset_index(drop=True)

    filtered_samples = original_samples - len(df)
    filtered_episodes = original_episodes - df["episode"].nunique()

    print(f"Loaded {df['episode'].nunique()} episodes (impression-to-action sequences)")
    print(
        f"  Filtered out {filtered_episodes} episodes ({filtered_samples} samples) with no positive labels"
    )
    return df


class Prepared(NamedTuple):
    """Everything training needs out of the CSV, with each part named."""

    X: pd.DataFrame
    y: pd.Series
    episodes: pd.Series
    timestamps: pd.Series
    categorical_features: list
    monotone_constraints: list


def prepare_features(df, feature_names, binary_features, monotonicity_map):
    """Convert features to numeric and prepare X, y, episodes, timestamps."""
    # Separate label and episode from features
    y = df["label"].astype(int)
    episodes = df["episode"]
    # Not a feature: training weights each row by how old it is.
    timestamps = df["timestamp"].astype(float)

    # Drop categorical features (query, file_path) since Rust lightgbm3 doesn't support them
    # Also drop metadata columns (including episode_id since it's used as episode)
    X = df.drop(
        columns=[
            "label",
            "episode",
            "episode_id",
            "subsession_id",
            "session_id",
            "timestamp",
            "query",
            "file_path",
        ]
    )

    # Ensure numeric features are correct type using schema
    for col in X.columns:
        if col in binary_features:
            # These are binary 0/1
            X[col] = X[col].astype(int)
        else:
            # Everything else should be numeric
            X[col] = X[col].astype(float)

    print(f"\nNumeric features: {list(X.columns)}")

    # Verify all features from schema are present
    missing_features = [f for f in feature_names if f not in X.columns]
    if missing_features:
        raise ValueError(f"Missing features from schema: {missing_features}")

    # Create the monotonicity constraints list from the map
    constraints = [monotonicity_map.get(f, 0) or 0 for f in X.columns]

    return Prepared(X, y, episodes, timestamps, [], constraints)  # No categorical features


def group_sizes(episodes):
    """Rows per episode, in the order they appear.

    lambdarank needs to know which rows compete with each other. `episode_id` is
    handed out in one pass over time-sorted events, so rows of one episode are
    already contiguous and ascending - asserted here rather than assumed,
    because getting it wrong silently trains on the wrong groups.
    """
    counts = episodes.value_counts().sort_index()
    assert counts.sum() == len(episodes), "every row belongs to exactly one episode"
    assert (episodes.values == episodes.sort_values().values).all(), (
        "rows must be grouped by episode and in ascending order"
    )
    return counts.values


def recency_weights(timestamps):
    """Weight each row by age, halving every HALF_LIFE_DAYS.

    Ages are measured from the newest row in the CSV rather than from now, so a
    model trained on a stale export is not uniformly discounted into noise.
    """
    assert len(timestamps) > 0, "there must be rows to weight"

    age_days = (timestamps.max() - timestamps) / 86400.0
    assert (age_days >= 0).all(), "no row can be newer than the newest row"

    weights = 0.5 ** (age_days / HALF_LIFE_DAYS)

    # How many equally weighted rows this is worth, which is the number to watch:
    # if it collapses, the half-life is throwing away most of the data.
    effective = weights.sum() ** 2 / (weights**2).sum()
    print(
        f"Recency weights: half-life {HALF_LIFE_DAYS} days, "
        f"span {age_days.max():.1f} days, "
        f"effective sample size {effective:.0f} of {len(weights)} rows"
    )
    return weights


def make_params(monotone_constraints):
    """The LightGBM parameters, shared by the validated fit and the refit.

    Both fits must see identical parameters: the first one decides how many
    rounds the second one runs for, and that number means nothing if the two
    models are shaped differently.

    **The objective is a ranking one.** psychic asks one question - of the files
    on screen, which is the one - and lambdarank is trained on exactly that: its
    gradient comes from swapping pairs *within* an episode, weighted by what the
    swap does to NDCG. A pooled objective instead learns the level of the scores
    as well as their order, and the level is not used for anything.
    """
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [1, 5],  # position 1 is what the user sees first
        "lambdarank_truncation_level": 30,  # a screenful, not 243 rows
        "monotone_constraints": monotone_constraints,  # Added monotonicity
        "boosting_type": "gbdt",
        # Smaller, shallower trees than the LightGBM defaults, because they are
        # both better here and cheaper to evaluate. Swept over three seeds and
        # three folds: 31 leaves at 0.05 gives 93 trees and 0.7754 top-1, while
        # 15 leaves at 0.1 gives 67 trees and 0.7960. Predict is proportional to
        # trees times depth and is most of the cost of ranking a query, so the
        # smaller model is the faster one as well as the better one - the usual
        # story when there are 1,200 positives to learn from.
        "num_leaves": 15,
        "learning_rate": 0.1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
        "bagging_seed": 42,
        "feature_fraction_seed": 42,
        "data_random_seed": 42,
    }


def train_model(
    X_train, y_train, episodes_train, w_train, X_val, y_val, episodes_val, w_val, categorical_features, monotone_constraints
):
    """Train LightGBM binary classification model with class weights and constraints."""
    # This function assumes X_train is a pandas DataFrame to get column names
    feature_names = list(X_train.columns)

    # --- NEW: Create the monotonicity constraints list ---
    # 1 for positive, -1 for negative, 0 for no constraint
    print(f"Applying monotonicity constraints: {monotone_constraints}")

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        weight=w_train,
        group=group_sizes(episodes_train),
        categorical_feature=categorical_features,
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        weight=w_val,
        group=group_sizes(episodes_val),
        categorical_feature=categorical_features,
        reference=train_data,
    )

    params = make_params(monotone_constraints)

    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result),
        ],
    )

    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score}")

    return model, evals_result


def ranking_quality(scores, labels, episodes):
    """Where the clicked row landed in each episode.

    top-1 is the share of episodes whose clicked row came out first, MRR the
    mean of 1/position. Both are per episode, which is how the tool is used: one
    list on screen, one file wanted.
    """
    labels = np.asarray(labels)
    groups = np.asarray(episodes)

    positions = []
    for episode in np.unique(groups):
        rows = groups == episode
        ranked = labels[rows][np.argsort(-scores[rows])]
        if ranked.max() != 1:
            continue
        positions.append(int(np.argmax(ranked == 1)) + 1)

    if not positions:
        return {"top1": 0.0, "mrr": 0.0, "episodes": 0, "positions": []}

    return {
        "top1": float(np.mean([p == 1 for p in positions])),
        "mrr": float(np.mean([1.0 / p for p in positions])),
        "episodes": len(positions),
        "positions": positions,
    }


def create_visualizations(
    model,
    X_train,
    y_train,
    episodes_train,
    X_test,
    y_test,
    episodes_test,
    evals_result,
    output_pdf,
):
    """Generate all visualizations and save to PDF."""
    print(f"\nGenerating visualizations to {output_pdf}...")

    with PdfPages(output_pdf) as pdf:
        # Page 1: Training curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Whatever metric the objective was trained on, rather than a hardcoded
        # name: this said "auc" and broke the moment the objective changed.
        metric = next(iter(evals_result["train"]))
        axes[0].plot(evals_result["train"][metric], label=f"Train {metric}", linewidth=2)
        axes[0].plot(evals_result["val"][metric], label=f"Validation {metric}", linewidth=2)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel(metric)
        axes[0].set_title(f"Training progress ({metric})")
        axes[0].legend()
        axes[0].grid(True)

        # Feature importance (Gain)
        importance = model.feature_importance(importance_type="gain")
        feature_names = model.feature_name()
        feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=True)

        axes[1].barh(
            feature_importance_df["feature"], feature_importance_df["importance"]
        )
        axes[1].set_xlabel("Importance (Gain)")
        axes[1].set_title("Feature Importance: Gain")
        axes[1].grid(True, axis="x")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 2: Feature Importance - Multiple Methods
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Split importance
        importance_split = model.feature_importance(importance_type="split")
        split_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance_split}
        ).sort_values("importance", ascending=True)
        axes[0, 0].barh(split_df["feature"], split_df["importance"])
        axes[0, 0].set_xlabel("Number of Splits")
        axes[0, 0].set_title("Feature Importance: Split Count")
        axes[0, 0].grid(True, axis="x")

        # Gain importance (repeated for comparison)
        axes[0, 1].barh(
            feature_importance_df["feature"], feature_importance_df["importance"]
        )
        axes[0, 1].set_xlabel("Total Gain")
        axes[0, 1].set_title("Feature Importance: Gain")
        axes[0, 1].grid(True, axis="x")

        # Permutation importance - correlation with target
        correlations = []
        for col in X_test.columns:
            if X_test[col].dtype in ["int64", "float64"]:
                corr = X_test[col].corr(y_test)
            else:
                # For categorical, use point-biserial (convert to numeric codes)
                corr = pd.Series(X_test[col].cat.codes).corr(y_test)
            correlations.append(abs(corr))

        corr_df = pd.DataFrame(
            {"feature": X_test.columns, "abs_correlation": correlations}
        ).sort_values("abs_correlation", ascending=True)

        axes[1, 0].barh(corr_df["feature"], corr_df["abs_correlation"])
        axes[1, 0].set_xlabel("Absolute Correlation with Label")
        axes[1, 0].set_title("Feature-Label Correlation")
        axes[1, 0].grid(True, axis="x")

        # Normalized comparison of all three
        norm_gain = (
            feature_importance_df.set_index("feature")["importance"]
            / feature_importance_df["importance"].max()
        )
        norm_split = (
            split_df.set_index("feature")["importance"] / split_df["importance"].max()
        )
        norm_corr = (
            corr_df.set_index("feature")["abs_correlation"]
            / corr_df["abs_correlation"].max()
        )

        comparison_df = pd.DataFrame(
            {"Gain": norm_gain, "Split": norm_split, "Correlation": norm_corr}
        )

        comparison_df.plot(kind="barh", ax=axes[1, 1], width=0.8)
        axes[1, 1].set_xlabel("Normalized Importance (0-1)")
        axes[1, 1].set_title("Feature Importance Comparison (Normalized)")
        axes[1, 1].legend(loc="lower right")
        axes[1, 1].grid(True, axis="x")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 3: SHAP Summary Plot
        print("Computing SHAP values (this may take a minute)...")
        explainer = shap.TreeExplainer(model)

        # Use a sample of test data for SHAP (can be slow on large datasets)
        sample_size = min(500, len(X_test))
        X_test_sample = X_test.sample(n=sample_size, random_state=42)

        # Compute SHAP values (keep categorical features as-is for LightGBM)
        shap_values = explainer.shap_values(X_test_sample)

        # For regression, shap_values is not a list (unlike classification)
        # So no need to extract a specific class

        # Convert categorical to numeric ONLY for visualization
        X_test_sample_numeric = X_test_sample.copy()
        for col in X_test_sample_numeric.columns:
            if X_test_sample_numeric[col].dtype.name == "category":
                X_test_sample_numeric[col] = X_test_sample_numeric[col].cat.codes

        fig, axes = plt.subplots(2, 1, figsize=(14, 12))

        # SHAP summary plot (bar)
        plt.sca(axes[0])
        shap.summary_plot(
            shap_values, X_test_sample_numeric, plot_type="bar", show=False
        )
        axes[0].set_title("SHAP Feature Importance (Mean |SHAP value|)")

        # SHAP summary plot (beeswarm)
        plt.sca(axes[1])
        shap.summary_plot(shap_values, X_test_sample_numeric, show=False)
        axes[1].set_title("SHAP Feature Impact (each dot is a sample)")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 4: SHAP Dependence Plots (top 4 features)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[-4:][::-1]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, feat_idx in enumerate(top_features_idx):
            feat_name = X_test_sample_numeric.columns[feat_idx]

            # Use numeric data for SHAP dependence plot
            shap.dependence_plot(
                feat_idx, shap_values, X_test_sample_numeric, show=False, ax=axes[i]
            )
            axes[i].set_title(f"SHAP Dependence: {feat_name}")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 5: Ranking quality and score distribution
        y_pred_scores = model.predict(X_test, num_iteration=model.best_iteration)

        # Ranking metrics, not regression ones. The objective emits an order,
        # not a probability, so a distance from a 0/1 label says nothing; what
        # matters is where the clicked row landed in its own episode.
        quality = ranking_quality(y_pred_scores, y_test, episodes_test)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Metrics summary
        metrics_text = (
            f"top-1: {quality['top1']:.4f}\n"
            f"MRR:   {quality['mrr']:.4f}\n"
            f"episodes: {quality['episodes']}"
        )
        axes[0, 0].text(
            0.5,
            0.5,
            metrics_text,
            ha="center",
            va="center",
            fontsize=16,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        axes[0, 0].set_xlim([0, 1])
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].set_title("Ranking quality (test set)")
        axes[0, 0].axis("off")

        # Score distribution for clicked vs not clicked
        axes[0, 1].hist(
            y_pred_scores[y_test == 0],
            bins=30,
            alpha=0.5,
            label="Not Clicked",
            color="blue",
        )
        axes[0, 1].hist(
            y_pred_scores[y_test == 1],
            bins=30,
            alpha=0.5,
            label="Clicked",
            color="red",
        )
        axes[0, 1].set_xlabel("Predicted Score")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Score Distribution by Label")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Where the clicked row actually landed, which is the thing being sold.
        axes[1, 0].hist(quality["positions"], bins=range(1, 22), align="left")
        axes[1, 0].set_xlabel("Position of the clicked row")
        axes[1, 0].set_ylabel("Episodes")
        axes[1, 0].set_title("Where the file the user wanted came out")
        axes[1, 0].grid(True, axis="y")

        # Score vs label scatter
        axes[1, 1].scatter(
            y_pred_scores[y_test == 0],
            np.random.normal(0, 0.05, sum(y_test == 0)),
            alpha=0.3,
            s=20,
            label="Not Clicked",
            color="blue",
        )
        axes[1, 1].scatter(
            y_pred_scores[y_test == 1],
            np.random.normal(1, 0.05, sum(y_test == 1)),
            alpha=0.8,
            s=40,
            label="Clicked",
            color="red",
        )
        axes[1, 1].set_xlabel("Predicted Score")
        axes[1, 1].set_ylabel("Label (jittered)")
        axes[1, 1].set_title("Scores by Label")
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_yticklabels(["Not Clicked", "Clicked"])
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Print ranking metrics
        print("\nRanking quality (test set):")
        print(f"  top-1: {quality['top1']:.4f}")
        print(f"  MRR:   {quality['mrr']:.4f}")
        print(f"  over {quality['episodes']} episodes")


def atomic_write(target_path, write_file):
    """Write a file by writing a temp file beside it and renaming over the target.

    psychic retrains in the background while the TUI is running, and the worker
    reloads model.txt and model_stats.json on its own schedule. A plain write
    truncates the file first, so a reload landing in that window reads an empty or
    half-written file - which looks exactly like "no model" and silently drops
    ranking back to the simple model until the next launch.

    os.replace is atomic on POSIX when source and destination are on the same
    filesystem, which is why the temp file goes in the target's own directory. A
    reader sees either the whole old file or the whole new one.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=target_path.parent, prefix=target_path.name + ".", suffix=".tmp"
    )
    os.close(fd)

    # mkstemp creates the file 0600. Match what a plain open() would have made it,
    # so writing atomically does not quietly turn the model owner-only.
    umask = os.umask(0o022)
    os.umask(umask)
    os.chmod(tmp_path, 0o666 & ~umask)

    try:
        write_file(tmp_path)
        os.replace(tmp_path, target_path)
    except BaseException:
        # Do not leave a stray temp file in the data directory on failure.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def save_model(model, output_prefix):
    """Save the LightGBM model where psychic will look for it.

    `output_prefix` is `<data_dir>/model`, so this writes `<data_dir>/model.txt`.
    It used to also write a second copy to a hardcoded ~/.local/share/psychic,
    which was the same file in the default case (writing 640KB twice, and widening
    the window where the model was truncated) and the *wrong* file whenever
    --data-dir pointed somewhere else.
    """
    model_path = f"{output_prefix}.txt"
    atomic_write(model_path, model.save_model)
    print(f"Model saved to: {model_path}")


def refit_on_everything(
    X, y, w, episodes, categorical_features, monotone_constraints, num_boost_round
):
    """Retrain on every row, for the number of rounds the validated fit settled on.

    The time split exists to answer one question - how many trees before this
    starts fitting noise - and it costs the last 20% of the data to answer it.
    Shipping that model would ship one that has never seen the most recent
    fortnight, which is the part most like what the user is about to search for.
    So the round count comes from the validated fit and the shipped model is
    grown on everything, with no early stopping because there is nothing held
    out to stop against.
    """
    assert num_boost_round > 0, "the validated fit must have produced some trees"

    print(f"\nRefitting on all {len(X)} rows for {num_boost_round} rounds")
    full_data = lgb.Dataset(
        X,
        label=y,
        weight=w,
        group=group_sizes(episodes),
        categorical_feature=categorical_features,
    )
    return lgb.train(
        make_params(monotone_constraints), full_data, num_boost_round=num_boost_round
    )


def time_split(episodes):
    """Split episode ids into train / validation / test by time.

    A random split leaks the future into the past: an episode from March 2 gets
    validated against a model that trained on March 3-30, so any feature that
    identifies a file is rewarded for memorising it, and early stopping keeps
    adding trees that memorise. `episode_id` is handed out in a single pass over
    time-sorted events (`features.rs`), so it is monotone in time and splitting
    on it splits on time. There is no timestamp column in the CSV to use instead.

    The three masks partition the rows, and every episode lands whole in one of
    them, because the boundary is drawn between episode ids.
    """
    q80, q90 = episodes.quantile(0.8), episodes.quantile(0.9)

    train_mask = episodes <= q80
    val_mask = (episodes > q80) & (episodes <= q90)
    test_mask = episodes > q90

    assert (train_mask.sum() + val_mask.sum() + test_mask.sum()) == len(episodes), (
        "the three splits must cover every row exactly once"
    )
    assert train_mask.any() and val_mask.any() and test_mask.any(), (
        f"every split needs rows; episode ids {episodes.min()}..{episodes.max()} "
        f"gave {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}"
    )
    return train_mask, val_mask, test_mask


def take(X, y, episodes, mask):
    """The rows a split mask selects, reindexed from zero."""
    return (
        X[mask].reset_index(drop=True),
        y[mask].reset_index(drop=True),
        episodes[mask].reset_index(drop=True),
    )


def describe_split(name, X_split, y_split, episodes_split, mask):
    """Print the size of a split and a hash of exactly which rows it holds."""
    import hashlib

    rows = np.flatnonzero(mask.to_numpy())
    split_hash = hashlib.md5(str(rows.tolist()).encode()).hexdigest()[:8]

    print(
        f"{name} set: {len(X_split)} samples, {episodes_split.nunique()} episodes "
        f"({y_split.sum()} positive), episodes {episodes_split.min()}-{episodes_split.max()}"
    )
    print(f"  {name} split hash: {split_hash}")


def main():
    import time

    training_start = time.time()

    parser = argparse.ArgumentParser(
        description="Train LightGBM ranking model on psychic feature data"
    )
    parser.add_argument("csv_path", help="Path to features CSV file")
    parser.add_argument(
        "output_prefix", help="Output prefix for model and visualizations"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.expanduser("~/.local/share/psychic"),
        help="Data directory for schema and outputs (default: ~/.local/share/psychic)",
    )

    args = parser.parse_args()

    csv_path = args.csv_path
    output_prefix = args.output_prefix
    output_pdf = f"{output_prefix}_viz.pdf"

    # Load schema
    feature_names, binary_features, monotonicity_map = load_schema(args.data_dir)

    # Load data
    df = load_data(csv_path)

    # Prepare features
    prepared = prepare_features(df, feature_names, binary_features, monotonicity_map)
    X, y, episodes = prepared.X, prepared.y, prepared.episodes
    categorical_features = prepared.categorical_features
    monotone_constraints = prepared.monotone_constraints

    # Weight rows by age, so the recent weeks lead and old bursts still count.
    weights = recency_weights(prepared.timestamps)

    # Split by time: the first 80% of episodes train, the next 10% validates,
    # the last 10% is the test set.
    train_mask, val_mask, test_mask = time_split(episodes)

    X_train, y_train, episodes_train = take(X, y, episodes, train_mask)
    X_val, y_val, episodes_val = take(X, y, episodes, val_mask)
    X_test, y_test, episodes_test = take(X, y, episodes, test_mask)
    w_train = weights[train_mask].reset_index(drop=True)
    w_val = weights[val_mask].reset_index(drop=True)

    print("")
    describe_split("Train", X_train, y_train, episodes_train, train_mask)
    describe_split("Validation", X_val, y_val, episodes_val, val_mask)
    describe_split("Test", X_test, y_test, episodes_test, test_mask)

    # Train model. This fit is the evaluation: it is the one with data held out.
    model, evals_result = train_model(
        X_train, y_train, episodes_train, w_train, X_val, y_val, episodes_val, w_val, categorical_features, monotone_constraints
    )

    # Ship a model grown on every row, for as many rounds as the validated fit
    # found worth growing.
    best_iteration = model.best_iteration
    final_model = refit_on_everything(
        X, y, weights, episodes, categorical_features, monotone_constraints, best_iteration
    )

    save_model(final_model, output_prefix)

    # Generate visualizations
    create_visualizations(
        model,
        X_train,
        y_train,
        episodes_train,
        X_test,
        y_test,
        episodes_test,
        evals_result,
        output_pdf,
    )

    # Calculate training duration
    training_duration = time.time() - training_start

    # Get feature importance (top 3) from the model that ships, not the one
    # that was only there to find the round count.
    importance = final_model.feature_importance(importance_type="gain")
    feature_names_list = final_model.feature_name()
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names_list, "importance": importance}
    ).sort_values("importance", ascending=False)
    top_3_features = feature_importance_df.head(3)[["feature", "importance"]].to_dict(
        "records"
    )

    # Write model stats to JSON
    import datetime

    stats = {
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_duration_seconds": round(training_duration, 2),
        "num_features": len(feature_names),
        "num_total_examples": len(df),
        "num_positive_examples": int(y.sum()),
        "num_negative_examples": int(len(y) - y.sum()),
        "best_iteration": best_iteration,
        "top_3_features": top_3_features,
    }

    stats_path = Path(args.data_dir) / "model_stats.json"

    def write_stats(path):
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    # Same reasoning as the model: the TUI reads this while training runs.
    atomic_write(stats_path, write_stats)
    print(f"  - Stats: {stats_path}")

    print("\n✓ Training complete!")
    print(f"  - Model: {output_prefix}.txt")
    print(f"  - Visualizations: {output_pdf}")


if __name__ == "__main__":
    main()
