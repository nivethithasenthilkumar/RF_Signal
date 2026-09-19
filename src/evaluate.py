"""
Evaluation and visualization module for RF Signal Level Classification.
Computes comprehensive classification metrics and generates publication-grade
diagnostic plots (Confusion Matrix, Multiclass ROC curves, Precision-Recall, Feature Importance).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from src.config import REPORTS_DIR, SIGNAL_CLASSES
from src.utils import ensure_directories, save_json, setup_logger

logger = setup_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    class_labels = classes or SIGNAL_CLASSES
    
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": {},
    }

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0)
    for label in class_labels:
        if label in report:
            metrics["per_class"][label] = {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1-score": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }

    # Multiclass ROC AUC if probabilities are provided
    if y_proba is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=class_labels)
            roc_auc_ovr = float(roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="macro"))
            metrics["roc_auc_ovr_macro"] = roc_auc_ovr
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generates dual confusion matrix (raw counts and normalized percentages)."""
    labels = classes or SIGNAL_CLASSES
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm_raw.astype("float") / np.maximum(cm_raw.sum(axis=1)[:, np.newaxis], 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Raw Counts
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
        cbar=True,
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("True Signal Level", fontsize=11)
    axes[0].set_xlabel("Predicted Signal Level", fontsize=11)

    # Normalized Rates
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        cbar=True,
    )
    axes[1].set_title("Normalized Confusion Matrix (%)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("True Signal Level", fontsize=11)
    axes[1].set_xlabel("Predicted Signal Level", fontsize=11)

    plt.tight_layout()
    save_file = output_path or (REPORTS_DIR / "confusion_matrix.png")
    fig.savefig(save_file, dpi=300)
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to {save_file}")
    return save_file


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generates Multiclass One-vs-Rest ROC curves for each signal level class."""
    labels = classes or SIGNAL_CLASSES
    y_true_bin = label_binarize(y_true, classes=labels)
    n_classes = len(labels)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc_score = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        ax.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f"{labels[i]} (AUC = {auc_score:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Multiclass One-vs-Rest ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_file = output_path or (REPORTS_DIR / "roc_curves.png")
    fig.savefig(save_file, dpi=300)
    plt.close(fig)
    logger.info(f"Saved ROC curves plot to {save_file}")
    return save_file


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generates Multiclass Precision-Recall curves."""
    labels = classes or SIGNAL_CLASSES
    y_true_bin = label_binarize(y_true, classes=labels)
    n_classes = len(labels)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        ax.plot(
            recall,
            precision,
            color=colors[i % len(colors)],
            lw=2,
            label=f"{labels[i]} (AP = {ap:.3f})",
        )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Multiclass Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_file = output_path or (REPORTS_DIR / "pr_curves.png")
    fig.savefig(save_file, dpi=300)
    plt.close(fig)
    logger.info(f"Saved PR curves plot to {save_file}")
    return save_file


def plot_feature_importance(
    pipeline: Pipeline,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    feature_names: Optional[List[str]] = None,
    top_n: int = 15,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Extracts and plots feature importance using tree MDI or permutation importance."""
    classifier = pipeline.named_steps.get("classifier")

    # Method 1: Check for native tree feature_importances_
    if hasattr(classifier, "feature_importances_") and feature_names is not None:
        importances = classifier.feature_importances_
        if len(importances) == len(feature_names):
            df_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances,
            }).sort_values(by="Importance", ascending=True).tail(top_n)
            title = f"Top {len(df_imp)} Most Influential RF Features (MDI)"
        else:
            df_imp = None
    else:
        df_imp = None

    # Method 2: Permutation Importance (robust, model-agnostic out-of-sample evaluation)
    if df_imp is None and X_test is not None and y_test is not None:
        # Sample subset for fast computation
        sample_size = min(len(X_test), 500)
        X_sub = X_test.iloc[:sample_size]
        y_sub = y_test.iloc[:sample_size]
        
        result = permutation_importance(
            pipeline, X_sub, y_sub, n_repeats=5, random_state=42, n_jobs=-1
        )
        col_names = list(X_test.columns)
        df_imp = pd.DataFrame({
            "Feature": col_names,
            "Importance": result.importances_mean,
        }).sort_values(by="Importance", ascending=True).tail(top_n)
        title = f"Top {len(df_imp)} Features (Permutation Importance on Test Set)"

    if df_imp is None:
        logger.warning("Could not compute feature importance.")
        return None

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(df_imp["Feature"], df_imp["Importance"], color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_file = output_path or (REPORTS_DIR / "feature_importance.png")
    fig.savefig(save_file, dpi=300)
    plt.close(fig)
    logger.info(f"Saved feature importance plot to {save_file}")
    return save_file


def evaluate_and_generate_reports(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs complete evaluation pipeline, computes metrics, and generates all artifact plots."""
    ensure_directories(REPORTS_DIR)
    # Align classes directly with pipeline.classes_ for exact predict_proba column correspondence
    labels = list(pipeline.classes_) if hasattr(pipeline, "classes_") else (classes or SIGNAL_CLASSES)

    # Predictions and Probabilities
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    # Compute Metrics
    metrics = compute_metrics(y_test.values, y_pred, y_proba=y_proba, classes=labels)
    
    # Generate Visualizations
    plot_confusion_matrix(y_test.values, y_pred, classes=labels)
    if y_proba is not None:
        plot_roc_curves(y_test.values, y_proba, classes=labels)
        plot_precision_recall_curves(y_test.values, y_proba, classes=labels)
    plot_feature_importance(pipeline, X_test=X_test, y_test=y_test, feature_names=feature_names)

    # Save Metrics Summary
    from src.config import METRICS_PATH
    save_json(metrics, METRICS_PATH)
    logger.info(f"Saved metrics summary to {METRICS_PATH}")

    return metrics
