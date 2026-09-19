"""
Unit tests for Preprocessing Pipeline and Data Splitting.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import ALL_FEATURES, TARGET_COLUMN
from src.data_generator import RFDataGenerator
from src.preprocessor import build_preprocessor_pipeline, split_data


@pytest.fixture
def sample_dataset():
    gen = RFDataGenerator(random_state=42)
    return gen.generate_dataset(n_samples=600, inject_missing_pct=0.02)


def test_split_proportions_and_no_leakage(sample_dataset):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        sample_dataset,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    )
    total = len(sample_dataset)
    assert len(X_train) == int(np.round(total * 0.70))
    assert len(X_val) == int(np.round(total * 0.15))
    assert len(X_test) == int(np.round(total * 0.15))

    # Verify indices do not overlap
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_preprocessor_handles_missing_values(sample_dataset):
    X = sample_dataset[ALL_FEATURES]
    from src.preprocessor import build_full_preprocessing_pipeline
    pipeline = build_full_preprocessing_pipeline()

    # Fit only on training chunk
    X_train = X.iloc[:400]
    X_test = X.iloc[400:]

    pipeline.fit(X_train)
    transformed = pipeline.transform(X_test)

    # Ensure no NaN or infinite values remain post-transformation
    assert not np.isnan(transformed).any()
    assert not np.isinf(transformed).any()
