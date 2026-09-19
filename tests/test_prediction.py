"""
Unit tests for Prediction Engine, input validation, and edge case error handling.
"""

import pytest

from src.config import SIGNAL_CLASSES
from src.data_generator import RFDataGenerator
from src.predict import RFPredictor, RFValidationError, predict_signal_level
from src.preprocessor import split_data
from src.train import build_and_tune_model


@pytest.fixture(scope="module")
def initialized_predictor(tmp_path_factory):
    # Train lightweight model and set on predictor
    gen = RFDataGenerator(random_state=42)
    df = gen.generate_dataset(n_samples=1500, inject_missing_pct=0.0)
    X_train, _, _, y_train, _, _ = split_data(df, random_state=42)
    pipe = build_and_tune_model(X_train, y_train, perform_grid_search=False)

    model_file = tmp_path_factory.mktemp("models") / "test_model.joblib"
    import joblib
    joblib.dump(pipe, model_file)

    predictor = RFPredictor(model_path=model_file)
    return predictor


def test_valid_prediction(initialized_predictor):
    params = {
        "frequency_mhz": 2100.0,
        "distance_m": 500.0,
        "tx_power_dbm": 43.0,
        "tx_gain_dbi": 15.0,
        "rx_gain_dbi": 2.0,
        "tx_height_m": 30.0,
        "rx_height_m": 1.5,
        "environment": "Urban",
        "obstacle_count": 3,
        "humidity_pct": 50.0,
        "weather_condition": "Clear",
    }
    result = initialized_predictor.predict(params)

    assert result["predicted_signal_level"] in SIGNAL_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == 4
    assert len(result["interpretation"]) > 0


def test_validation_error_missing_feature(initialized_predictor):
    params = {
        "frequency_mhz": 2100.0,
        "distance_m": 500.0,
        # missing other fields
    }
    with pytest.raises(RFValidationError) as excinfo:
        initialized_predictor.predict(params)
    assert "Missing required RF parameters" in str(excinfo.value)


def test_validation_error_negative_distance(initialized_predictor):
    params = {
        "frequency_mhz": 2100.0,
        "distance_m": -20.0,  # invalid
        "tx_power_dbm": 43.0,
        "tx_gain_dbi": 15.0,
        "rx_gain_dbi": 2.0,
        "tx_height_m": 30.0,
        "rx_height_m": 1.5,
        "environment": "Urban",
        "obstacle_count": 3,
        "humidity_pct": 50.0,
        "weather_condition": "Clear",
    }
    with pytest.raises(RFValidationError) as excinfo:
        initialized_predictor.predict(params)
    assert "outside physically plausible range" in str(excinfo.value)


def test_validation_error_invalid_category(initialized_predictor):
    params = {
        "frequency_mhz": 2100.0,
        "distance_m": 200.0,
        "tx_power_dbm": 43.0,
        "tx_gain_dbi": 15.0,
        "rx_gain_dbi": 2.0,
        "tx_height_m": 30.0,
        "rx_height_m": 1.5,
        "environment": "MoonSurface",  # invalid
        "obstacle_count": 0,
        "humidity_pct": 50.0,
        "weather_condition": "Clear",
    }
    with pytest.raises(RFValidationError) as excinfo:
        initialized_predictor.predict(params)
    assert "Invalid environment" in str(excinfo.value)
