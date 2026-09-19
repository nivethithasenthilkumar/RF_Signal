"""
Unit tests for RFDataGenerator module.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    ALL_FEATURES,
    CONTINUOUS_SIGNAL_COLUMN,
    SIGNAL_CLASSES,
    TARGET_COLUMN,
    VALID_ENVIRONMENTS,
    VALID_WEATHER_CONDITIONS,
)
from src.data_generator import RFDataGenerator


@pytest.fixture
def generator():
    return RFDataGenerator(random_state=42)


def test_data_generation_shape_and_columns(generator):
    df = generator.generate_dataset(n_samples=500, inject_missing_pct=0.0)
    
    assert len(df) == 500
    for feat in ALL_FEATURES:
        assert feat in df.columns
    assert CONTINUOUS_SIGNAL_COLUMN in df.columns
    assert TARGET_COLUMN in df.columns


def test_categorical_values_valid(generator):
    df = generator.generate_dataset(n_samples=200, inject_missing_pct=0.0)
    
    assert set(df["environment"].unique()).issubset(set(VALID_ENVIRONMENTS))
    assert set(df["weather_condition"].unique()).issubset(set(VALID_WEATHER_CONDITIONS))
    assert set(df[TARGET_COLUMN].unique()).issubset(set(SIGNAL_CLASSES))


def test_physics_power_monotonicity_distance(generator):
    """Verifies that as distance increases (all else constant), received power monotonically decreases."""
    base_params = {
        "frequency_mhz": [1800.0, 1800.0, 1800.0, 1800.0],
        "distance_m": [50.0, 200.0, 800.0, 3000.0],
        "tx_power_dbm": [40.0, 40.0, 40.0, 40.0],
        "tx_gain_dbi": [10.0, 10.0, 10.0, 10.0],
        "rx_gain_dbi": [2.0, 2.0, 2.0, 2.0],
        "tx_height_m": [25.0, 25.0, 25.0, 25.0],
        "rx_height_m": [1.5, 1.5, 1.5, 1.5],
        "obstacle_count": [0, 0, 0, 0],
        "humidity_pct": [50.0, 50.0, 50.0, 50.0],
        "environment": ["Rural", "Rural", "Rural", "Rural"],
        "weather_condition": ["Clear", "Clear", "Clear", "Clear"],
    }
    df = pd.DataFrame(base_params)
    # Run multiple times with fixed seed to test expected mean
    gen = RFDataGenerator(random_state=123)
    p_rx, _ = gen.compute_received_power(df)
    
    # Distance increases -> Path Loss increases -> Received power decreases
    assert p_rx[0] > p_rx[1] > p_rx[2] > p_rx[3]


def test_reproducibility():
    gen1 = RFDataGenerator(random_state=99)
    df1 = gen1.generate_dataset(n_samples=100, inject_missing_pct=0.0)

    gen2 = RFDataGenerator(random_state=99)
    df2 = gen2.generate_dataset(n_samples=100, inject_missing_pct=0.0)

    pd.testing.assert_frame_equal(df1, df2)
