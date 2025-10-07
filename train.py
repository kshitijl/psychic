#!/usr/bin/env python3
"""
Train LightGBM model on psychic feature data and generate evaluation visualizations.

Usage:
    python train.py /tmp/features.csv output.pdf
"""

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import shap
import json

sns.set_style("whitegrid")


def load_data(csv_path):
    """Load features CSV and prepare for training."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"\nFeatures: {[c for c in df.columns if c != 'label']}")
    return df


def prepare_features(df):
    """Convert features to numeric and prepare X, y."""
    # Separate label from features
    y = df["label"].astype(int)
    X = df.drop(columns=["label"])

    # Track categorical features (keep as category dtype for LightGBM)
    categorical_features = []

    for col in X.columns:
        if X[col].dtype == "object":
            # Convert to category dtype - LightGBM handles this natively
            X[col] = X[col].astype("category")
            categorical_features.append(col)
        elif col in ["filename_starts_with_query", "modified_today"]:
            # These are binary 0/1 but might be read as int
            X[col] = X[col].astype(int)

    print(f"\nCategorical features: {categorical_features}")
    return X, y, categorical_features


def train_model(X_train, y_train, X_val, y_val, categorical_features):
    """Train LightGBM classifier."""
    print("\nTraining LightGBM model...")

    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_features
    )
    val_data = lgb.Dataset(
        X_val, label=y_val, categorical_feature=categorical_features, reference=train_data
    )

    # LightGBM parameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
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


def create_visualizations(model, X_train, y_train, X_test, y_test, evals_result, output_pdf):
    """Generate all visualizations and save to PDF."""
    print(f"\nGenerating visualizations to {output_pdf}...")

    with PdfPages(output_pdf) as pdf:
        # Page 1: Training curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # AUC curves
        train_auc = evals_result["train"]["auc"]
        val_auc = evals_result["val"]["auc"]
        axes[0].plot(train_auc, label="Train AUC", linewidth=2)
        axes[0].plot(val_auc, label="Validation AUC", linewidth=2)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("AUC")
        axes[0].set_title("Training Progress")
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

        # Page 5: ROC Curve and Precision-Recall
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROC curve
        axes[0].plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve")
        axes[0].legend()
        axes[0].grid(True)

        # Precision-Recall curve
        axes[1].plot(recall, precision, linewidth=2, label=f"AP = {avg_precision:.3f}")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall Curve")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 6: Confusion Matrix and Predicted Distribution
        y_pred = (y_pred_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0],
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"],
        )
        axes[0].set_title("Confusion Matrix (threshold=0.5)")

        # Predicted probability distribution
        axes[1].hist(
            y_pred_proba[y_test == 0],
            bins=50,
            alpha=0.5,
            label="Actual 0",
            color="blue",
        )
        axes[1].hist(
            y_pred_proba[y_test == 1],
            bins=50,
            alpha=0.5,
            label="Actual 1",
            color="red",
        )
        axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1, label="Threshold")
        axes[1].set_xlabel("Predicted Probability")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Predicted Probability Distribution")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 7: Calibration and predicted vs actual
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Calibration curve (binned predicted prob vs actual positive rate)
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(y_pred_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_sums = np.bincount(bin_indices, weights=y_test, minlength=n_bins)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_true = np.divide(
            bin_sums, bin_counts, out=np.zeros_like(bin_sums), where=bin_counts != 0
        )

        axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        axes[0].plot(bin_centers, bin_true, "o-", linewidth=2, label="Model")
        axes[0].set_xlabel("Predicted Probability")
        axes[0].set_ylabel("Actual Positive Rate")
        axes[0].set_title("Calibration Curve")
        axes[0].legend()
        axes[0].grid(True)

        # Predicted vs actual scatter (jittered for visibility)
        jitter = 0.05
        y_test_jittered = y_test + np.random.uniform(-jitter, jitter, len(y_test))
        axes[1].scatter(
            y_pred_proba, y_test_jittered, alpha=0.3, s=10, c=y_test, cmap="coolwarm"
        )
        axes[1].set_xlabel("Predicted Probability")
        axes[1].set_ylabel("Actual Label (jittered)")
        axes[1].set_title("Predicted vs Actual (scatter)")
        axes[1].grid(True)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        print(f"\nClassification Report (threshold=0.5):")
        # Handle case where test set might not have both classes
        labels_in_test = sorted(y_test.unique())
        if len(labels_in_test) == 2:
            print(classification_report(y_test, y_pred, target_names=["Not Clicked", "Clicked"], labels=[0, 1]))
        else:
            print(f"Warning: Test set only contains class {labels_in_test}. Cannot generate full classification report.")
            print(f"Test set distribution: {y_test.value_counts().to_dict()}")
            print(f"Predictions distribution: {pd.Series(y_pred).value_counts().to_dict()}")


def save_model(model, output_prefix):
    """Save LightGBM model to file."""
    model_path = f"{output_prefix}.txt"
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python train.py <features.csv> <output_prefix>")
        print("  Example: python train.py /tmp/features.csv model")
        print("  Outputs: model.txt (LightGBM model), model_viz.pdf")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_prefix = sys.argv[2]
    output_pdf = f"{output_prefix}_viz.pdf"

    # Load data
    df = load_data(csv_path)

    # Prepare features
    X, y, categorical_features = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"\nTrain set: {len(X_train)} samples ({y_train.sum()} positive, {(~y_train.astype(bool)).sum()} negative)"
    )
    print(
        f"Test set: {len(X_test)} samples ({y_test.sum()} positive, {(~y_test.astype(bool)).sum()} negative)"
    )

    # Further split train into train/val for early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Train model
    model, evals_result = train_model(X_train, y_train, X_val, y_val, categorical_features)

    # Save model
    save_model(model, output_prefix)

    # Generate visualizations
    create_visualizations(model, X_train, y_train, X_test, y_test, evals_result, output_pdf)

    print(f"\n✓ Training complete!")
    print(f"  - Model: {output_prefix}.txt")
    print(f"  - Visualizations: {output_pdf}")


if __name__ == "__main__":
    main()
