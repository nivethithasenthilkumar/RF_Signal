"""
Training module for RF Signal Level Prediction.
Implements model comparison, hyperparameter optimization, model calibration,
and pipeline serialization.
"""

from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_FEATURES,
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    SIGNAL_CLASSES,
    TARGET_COLUMN,
)
from src.data_generator import generate_and_save_data
from src.evaluate import evaluate_and_generate_reports
from src.preprocessor import (
    RFFeatureEngineeringTransformer,
    RawImputerTransformer,
    build_full_preprocessing_pipeline,
    build_preprocessor_pipeline,
    get_feature_names_out,
    split_data,
)
from src.utils import ensure_directories, setup_logger

logger = setup_logger(__name__)


def build_candidate_pipeline(classifier) -> Pipeline:
    """Creates an end-to-end pipeline with imputation, feature engineering, preprocessing, and classifier."""
    return Pipeline([
        ("imputer", RawImputerTransformer()),
        ("fe", RFFeatureEngineeringTransformer()),
        ("preprocessor", build_preprocessor_pipeline(ALL_NUMERICAL_FEATURES, CATEGORICAL_FEATURES)),
        ("classifier", classifier),
    ])


def train_baseline_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, float]:
    """Evaluates multiple candidate models to establish baselines and select optimal architecture."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=16, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_depth=12, random_state=RANDOM_STATE
        ),
    }

    results = {}
    logger.info("--- Baseline Model Comparison ---")
    for name, clf in models.items():
        pipe = build_candidate_pipeline(clf)
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        acc = float(accuracy_score(y_val, val_pred))
        f1 = float(f1_score(y_val, val_pred, average="macro"))
        results[name] = acc
        logger.info(f"Model: {name:22s} | Val Accuracy: {acc:.4f} ({acc*100:.2f}%) | Val Macro F1: {f1:.4f}")

    return results


def build_and_tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    perform_grid_search: bool = False,
) -> Pipeline:
    """Constructs the optimized Pipeline with tuned Gradient Boosting Classifier.
    
    Gradient Boosting is the gold standard for non-linear physical systems with
    interacting environmental terms, providing superior boundary discrimination
    and probabilistic calibration.
    """
    base_clf = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.08,
        max_depth=12,
        min_samples_leaf=15,
        l2_regularization=0.01,
        random_state=RANDOM_STATE,
    )

    full_pipeline = build_candidate_pipeline(base_clf)

    if perform_grid_search:
        logger.info("Executing hyperparameter optimization via Stratified 3-Fold CV...")
        param_grid = {
            "classifier__max_iter": [180, 240],
            "classifier__learning_rate": [0.06, 0.08],
            "classifier__max_depth": [10, 14],
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        logger.info(f"Optimal Hyperparameters: {grid_search.best_params_}")
        logger.info(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
        best_pipeline = grid_search.best_estimator_
    else:
        logger.info("Fitting base optimized pipeline directly...")
        full_pipeline.fit(X_train, y_train)
        best_pipeline = full_pipeline

    return best_pipeline


def train_pipeline(
    data_path: Optional[str] = None,
    model_save_path: Optional[str] = None,
    tune_hyperparameters: bool = False,
    n_samples: int = 12000,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """End-to-end training and evaluation orchestration function."""
    csv_file = data_path or str(DATASET_PATH)
    export_path = model_save_path or str(MODEL_PATH)

    # 1. Load or generate data
    import os
    if not os.path.exists(csv_file):
        logger.info(f"Dataset not found at {csv_file}. Generating {n_samples} physics-grounded synthetic samples...")
        df = generate_and_save_data(filepath=csv_file, n_samples=n_samples)
    else:
        logger.info(f"Loading existing RF dataset from {csv_file}...")
        df = pd.read_csv(csv_file)

    # 2. Strict featurization ordering: split before fitting preprocessor
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # 3. Baseline comparison
    train_baseline_comparison(X_train, y_train, X_val, y_val)

    # 4. Train and optimize selected model
    logger.info("Training and tuning primary Gradient Boosting classifier...")
    pipeline = build_and_tune_model(X_train, y_train, perform_grid_search=tune_hyperparameters)

    # 5. Validation Check
    val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    logger.info(f"Validation Set Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

    # 6. Fit feature names out for feature importance plot
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())

    # 7. Final Test Evaluation & Diagnostic Reports
    logger.info("Running comprehensive evaluation on held-out Test set...")
    metrics = evaluate_and_generate_reports(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        classes=SIGNAL_CLASSES,
    )

    test_acc = metrics["accuracy"]
    logger.info(f"Test Set Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    logger.info(f"Test Set Macro F1: {metrics['f1_macro']:.4f}")
    logger.info(f"Test Set Weighted F1: {metrics['f1_weighted']:.4f}")

    # 8. Serialize trained production pipeline
    ensure_directories(MODEL_PATH.parent)
    joblib.dump(pipeline, export_path)
    logger.info(f"Trained model pipeline serialized successfully to {export_path}")

    return pipeline, metrics


if __name__ == "__main__":
    train_pipeline(tune_hyperparameters=False)
