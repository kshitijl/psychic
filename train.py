#!/usr/bin/env python3
# /// script
# dependencies = [
#   "lightgbm>=4.6.0",
#   "matplotlib>=3.10.6",
#   "pandas>=2.3.3",
#   "scikit-learn>=1.7.2",
#   "seaborn>=0.13.2",
#   "shap>=0.48.0",
# ]
# ///
"""
Train LightGBM ranking model on psychic feature data and generate evaluation visualizations.

Usage:
    python train.py features.csv output_prefix [--data-dir DIR]
"""

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    ndcg_score,
    label_ranking_average_precision_score,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import shap
import json
from pathlib import Path
import os
import argparse

sns.set_style("whitegrid")


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
    with open(schema_path, 'rb') as f:
        schema_hash = hashlib.md5(f.read()).hexdigest()[:8]

    with open(schema_path) as f:
        schema = json.load(f)

    # Extract feature names and types from schema
    feature_names = [f["name"] for f in schema["features"]]
    binary_features = [f["name"] for f in schema["features"] if f["type"] == "binary"]
    print(f"Loaded feature schema: {len(feature_names)} features")
    print(f"  Schema file hash: {schema_hash}")

    return feature_names, binary_features


def load_data(csv_path):
    """Load features CSV and prepare for training."""
    import hashlib

    # Hash the CSV file to verify it's the same across runs
    with open(csv_path, 'rb') as f:
        csv_hash = hashlib.md5(f.read()).hexdigest()[:8]

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"  CSV file hash: {csv_hash}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"\nFeatures: {[c for c in df.columns if c not in ['label', 'group_id', 'subsession_id', 'session_id']]}")

    # Use group_id as query groups (each group spans from one click/scroll to the next)
    df['query_group'] = df['group_id'].astype(int)

    # Filter out query groups with no positive labels (no clicks/scrolls)
    # LambdaRank needs at least one positive example per query group
    groups_with_positives = df.groupby('query_group')['label'].sum()
    valid_groups = groups_with_positives[groups_with_positives > 0].index

    original_samples = len(df)
    original_groups = df['query_group'].nunique()

    df = df[df['query_group'].isin(valid_groups)].reset_index(drop=True)

    filtered_samples = original_samples - len(df)
    filtered_groups = original_groups - df['query_group'].nunique()

    print(f"Loaded {df['query_group'].nunique()} query groups (click/scroll sequences)")
    print(f"  Filtered out {filtered_groups} groups ({filtered_samples} samples) with no positive labels")
    return df


def prepare_features(df, feature_names, binary_features):
    """Convert features to numeric and prepare X, y, query_groups."""
    # Separate label and query_group from features
    y = df["label"].astype(int)
    query_groups = df["query_group"]

    # Drop categorical features (query, file_path) since Rust lightgbm3 doesn't support them
    # Also drop metadata columns (including group_id since it's used as query_group)
    X = df.drop(columns=["label", "query_group", "group_id", "subsession_id", "session_id", "query", "file_path"])

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

    return X, y, query_groups, []  # No categorical features


def train_model(X_train, y_train, groups_train, X_val, y_val, groups_val, categorical_features):
    """Train LightGBM ranking model."""
    print("\nTraining LightGBM ranking model...")

    # Compute group sizes for LambdaRank
    # LightGBM needs to know how many items per query group
    train_group_sizes = groups_train.value_counts().sort_index().values
    val_group_sizes = groups_val.value_counts().sort_index().values

    # Create LightGBM datasets with group info
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=train_group_sizes,
        categorical_feature=categorical_features
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        group=val_group_sizes,
        categorical_feature=categorical_features,
        reference=train_data
    )

    # LightGBM LambdaRank parameters
    # Seeds are critical for deterministic behavior with small sample sizes
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        # Deterministic seeds
        "seed": 42,
        "bagging_seed": 42,
        "feature_fraction_seed": 42,
        "data_random_seed": 42,
    }

    # Train with early stopping
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


def create_visualizations(model, X_train, y_train, groups_train, X_test, y_test, groups_test, evals_result, output_pdf):
    """Generate all visualizations and save to PDF."""
    print(f"\nGenerating visualizations to {output_pdf}...")

    with PdfPages(output_pdf) as pdf:
        # Page 1: Training curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # NDCG curves - plot NDCG@5 as primary metric
        train_ndcg = evals_result["train"]["ndcg@5"]
        val_ndcg = evals_result["val"]["ndcg@5"]
        axes[0].plot(train_ndcg, label="Train NDCG@5", linewidth=2)
        axes[0].plot(val_ndcg, label="Validation NDCG@5", linewidth=2)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("NDCG@5")
        axes[0].set_title("Training Progress (Ranking Quality)")
        axes[0].legend()
        axes[0].grid(True)

        # Feature importance (Gain)
        importance = model.feature_importance(importance_type="gain")
        feature_names = model.feature_name()
        feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=True)

        axes[1].barh(feature_importance_df["feature"], feature_importance_df["importance"])
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
        axes[0, 1].barh(feature_importance_df["feature"], feature_importance_df["importance"])
        axes[0, 1].set_xlabel("Total Gain")
        axes[0, 1].set_title("Feature Importance: Gain")
        axes[0, 1].grid(True, axis="x")

        # Permutation importance - correlation with target
        correlations = []
        for col in X_test.columns:
            if X_test[col].dtype in ['int64', 'float64']:
                corr = X_test[col].corr(y_test)
            else:
                # For categorical, use point-biserial (convert to numeric codes)
                corr = pd.Series(X_test[col].cat.codes).corr(y_test)
            correlations.append(abs(corr))

        corr_df = pd.DataFrame({
            "feature": X_test.columns,
            "abs_correlation": correlations
        }).sort_values("abs_correlation", ascending=True)

        axes[1, 0].barh(corr_df["feature"], corr_df["abs_correlation"])
        axes[1, 0].set_xlabel("Absolute Correlation with Label")
        axes[1, 0].set_title("Feature-Label Correlation")
        axes[1, 0].grid(True, axis="x")

        # Normalized comparison of all three
        norm_gain = feature_importance_df.set_index("feature")["importance"] / feature_importance_df["importance"].max()
        norm_split = split_df.set_index("feature")["importance"] / split_df["importance"].max()
        norm_corr = corr_df.set_index("feature")["abs_correlation"] / corr_df["abs_correlation"].max()

        comparison_df = pd.DataFrame({
            "Gain": norm_gain,
            "Split": norm_split,
            "Correlation": norm_corr
        })

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

        # SHAP uses [negative_class, positive_class], we want positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Convert categorical to numeric ONLY for visualization
        X_test_sample_numeric = X_test_sample.copy()
        for col in X_test_sample_numeric.columns:
            if X_test_sample_numeric[col].dtype.name == 'category':
                X_test_sample_numeric[col] = X_test_sample_numeric[col].cat.codes

        fig, axes = plt.subplots(2, 1, figsize=(14, 12))

        # SHAP summary plot (bar)
        plt.sca(axes[0])
        shap.summary_plot(shap_values, X_test_sample_numeric, plot_type="bar", show=False)
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
                feat_idx,
                shap_values,
                X_test_sample_numeric,
                show=False,
                ax=axes[i]
            )
            axes[i].set_title(f"SHAP Dependence: {feat_name}")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 5: Ranking Quality Metrics and Score Distribution
        y_pred_scores = model.predict(X_test, num_iteration=model.best_iteration)

        # Compute NDCG for test set
        # Group by query and compute NDCG per query
        test_df = pd.DataFrame({
            'group': groups_test.values,
            'label': y_test.values,
            'score': y_pred_scores
        })

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # NDCG at different k values
        ndcg_at_k = {}
        for k in [1, 3, 5, 10]:
            ndcgs = []
            for group_id in test_df['group'].unique():
                group_data = test_df[test_df['group'] == group_id]
                # Only compute NDCG if group has at least k items and at least one positive label
                if len(group_data) > k and group_data['label'].sum() > 0:
                    y_true = group_data['label'].values.reshape(1, -1)
                    y_score = group_data['score'].values.reshape(1, -1)
                    ndcg = ndcg_score(y_true, y_score, k=k)
                    ndcgs.append(ndcg)
            ndcg_at_k[k] = np.mean(ndcgs) if ndcgs else 0

        axes[0, 0].bar(ndcg_at_k.keys(), ndcg_at_k.values())
        axes[0, 0].set_xlabel("k")
        axes[0, 0].set_ylabel("NDCG@k")
        axes[0, 0].set_title("NDCG at Different Cutoffs")
        axes[0, 0].grid(True, axis='y')
        axes[0, 0].set_ylim([0, 1])

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

        # Rank position analysis - where do clicked items appear?
        rank_positions = []
        for group_id in test_df['group'].unique():
            group_data = test_df[test_df['group'] == group_id].sort_values('score', ascending=False)
            group_data['rank'] = range(1, len(group_data) + 1)
            clicked_ranks = group_data[group_data['label'] == 1]['rank'].tolist()
            rank_positions.extend(clicked_ranks)

        if rank_positions:
            axes[1, 0].hist(rank_positions, bins=range(1, max(rank_positions) + 2), edgecolor='black')
            axes[1, 0].set_xlabel("Rank Position")
            axes[1, 0].set_ylabel("Count of Clicked Items")
            axes[1, 0].set_title("Distribution of Clicked Item Ranks")
            axes[1, 0].grid(True, axis='y')

        # Score vs label scatter
        axes[1, 1].scatter(
            y_pred_scores[y_test == 0],
            np.random.normal(0, 0.05, sum(y_test == 0)),
            alpha=0.3,
            s=20,
            label="Not Clicked",
            color="blue"
        )
        axes[1, 1].scatter(
            y_pred_scores[y_test == 1],
            np.random.normal(1, 0.05, sum(y_test == 1)),
            alpha=0.8,
            s=40,
            label="Clicked",
            color="red"
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
        print(f"\nRanking Metrics (Test Set):")
        print(f"  NDCG@1:  {ndcg_at_k.get(1, 0):.4f}")
        print(f"  NDCG@3:  {ndcg_at_k.get(3, 0):.4f}")
        print(f"  NDCG@5:  {ndcg_at_k.get(5, 0):.4f}")
        print(f"  NDCG@10: {ndcg_at_k.get(10, 0):.4f}")
        if rank_positions:
            print(f"\nClicked items average rank: {np.mean(rank_positions):.2f}")
            print(f"Clicked items median rank: {np.median(rank_positions):.1f}")


def save_model(model, output_prefix):
    """Save LightGBM model to file."""
    import os

    # Save to current directory with given prefix
    model_path = f"{output_prefix}.txt"
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")

    # Also save to ~/.local/share/psychic/model.txt
    home = os.path.expanduser("~")
    psychic_dir = os.path.join(home, ".local", "share", "psychic")
    os.makedirs(psychic_dir, exist_ok=True)
    sg_model_path = os.path.join(psychic_dir, "model.txt")
    model.save_model(sg_model_path)
    print(f"Model also saved to: {sg_model_path}")


def main():
    import time
    training_start = time.time()

    parser = argparse.ArgumentParser(
        description="Train LightGBM ranking model on psychic feature data"
    )
    parser.add_argument("csv_path", help="Path to features CSV file")
    parser.add_argument("output_prefix", help="Output prefix for model and visualizations")
    parser.add_argument("--data-dir", type=str,
                       default=os.path.expanduser("~/.local/share/psychic"),
                       help="Data directory for schema and outputs (default: ~/.local/share/psychic)")

    args = parser.parse_args()

    csv_path = args.csv_path
    output_prefix = args.output_prefix
    output_pdf = f"{output_prefix}_viz.pdf"

    # Load schema
    feature_names, binary_features = load_schema(args.data_dir)

    # Load data
    df = load_data(csv_path)

    # Prepare features
    X, y, query_groups, categorical_features = prepare_features(df, feature_names, binary_features)

    # Split data by query groups (so each query stays together)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=query_groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    groups_train = query_groups.iloc[train_idx].reset_index(drop=True)
    groups_test = query_groups.iloc[test_idx].reset_index(drop=True)

    # Compute hashes to verify deterministic splits
    import hashlib
    train_hash = hashlib.md5(str(sorted(train_idx)).encode()).hexdigest()[:8]
    test_hash = hashlib.md5(str(sorted(test_idx)).encode()).hexdigest()[:8]

    print(
        f"\nTrain set: {len(X_train)} samples, {groups_train.nunique()} query groups ({y_train.sum()} positive)"
    )
    print(f"  Train split hash: {train_hash}")
    print(
        f"Test set: {len(X_test)} samples, {groups_test.nunique()} query groups ({y_test.sum()} positive)"
    )
    print(f"  Test split hash: {test_hash}")

    # Further split train into train/val for early stopping
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx2, val_idx = next(splitter_val.split(X_train, y_train, groups=groups_train))

    X_val = X_train.iloc[val_idx].reset_index(drop=True)
    y_val = y_train.iloc[val_idx].reset_index(drop=True)
    groups_val = groups_train.iloc[val_idx].reset_index(drop=True)

    X_train = X_train.iloc[train_idx2].reset_index(drop=True)
    y_train = y_train.iloc[train_idx2].reset_index(drop=True)
    groups_train = groups_train.iloc[train_idx2].reset_index(drop=True)

    # Print validation split hash
    val_hash = hashlib.md5(str(sorted(val_idx)).encode()).hexdigest()[:8]
    print(f"Validation set: {len(X_val)} samples, {groups_val.nunique()} query groups ({y_val.sum()} positive)")
    print(f"  Validation split hash: {val_hash}")

    # Train model
    model, evals_result = train_model(X_train, y_train, groups_train, X_val, y_val, groups_val, categorical_features)

    # Save model
    save_model(model, output_prefix)

    # Generate visualizations
    create_visualizations(model, X_train, y_train, groups_train, X_test, y_test, groups_test, evals_result, output_pdf)

    # Calculate training duration
    training_duration = time.time() - training_start

    # Get feature importance (top 3)
    importance = model.feature_importance(importance_type="gain")
    feature_names_list = model.feature_name()
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names_list, "importance": importance}
    ).sort_values("importance", ascending=False)
    top_3_features = feature_importance_df.head(3)[["feature", "importance"]].to_dict('records')

    # Write model stats to JSON
    import datetime
    stats = {
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_duration_seconds": round(training_duration, 2),
        "num_features": len(feature_names),
        "num_total_examples": len(df),
        "num_positive_examples": int(y.sum()),
        "num_negative_examples": int(len(y) - y.sum()),
        "top_3_features": top_3_features
    }

    stats_path = Path(args.data_dir) / "model_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  - Stats: {stats_path}")

    print(f"\n✓ Training complete!")
    print(f"  - Model: {output_prefix}.txt")
    print(f"  - Visualizations: {output_pdf}")


if __name__ == "__main__":
    main()
