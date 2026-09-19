"""
Unit tests for model training, metrics calculation, and performance criteria.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import RFDataGenerator
from src.evaluate import compute_metrics
from src.preprocessor import split_data
from src.train import build_and_tune_model


@pytest.fixture(scope="module")
def trained_artifacts():
    gen = RFDataGenerator(random_state=42)
    df = gen.generate_dataset(n_samples=8000, inject_missing_pct=0.005)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, random_state=42)
    pipeline = build_and_tune_model(X_train, y_train, perform_grid_search=False)
    return pipeline, X_test, y_test


def test_model_accuracy_criteria(trained_artifacts):
    pipeline, X_test, y_test = trained_artifacts
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    metrics = compute_metrics(y_test.values, y_pred, y_proba=y_proba)
    acc = metrics["accuracy"]
    f1 = metrics["f1_weighted"]

    # Criteria: At least 90% accuracy, target ~92%
    assert acc >= 0.90, f"Model accuracy {acc:.4f} is below required 0.90"
    assert f1 >= 0.89, f"Model F1 score {f1:.4f} is below expected threshold"


def test_model_predict_proba_validity(trained_artifacts):
    pipeline, X_test, _ = trained_artifacts
    y_proba = pipeline.predict_proba(X_test)

    assert y_proba.shape[1] == 4
    # All probabilities in [0, 1]
    assert (y_proba >= 0.0).all() and (y_proba <= 1.0).all()
    # Rows sum to 1.0 (within float precision)
    np.testing.assert_allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)
