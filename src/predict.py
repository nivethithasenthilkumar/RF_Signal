"""
Production inference and prediction module for RF Signal Level Prediction.
Provides robust input validation, boundary checking, graceful error handling,
confidence scoring, and domain-informed explanations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_BOUNDS,
    MODEL_PATH,
    NUMERICAL_FEATURES,
    SIGNAL_CLASSES,
    VALID_ENVIRONMENTS,
    VALID_WEATHER_CONDITIONS,
)
from src.utils import setup_logger

logger = setup_logger(__name__)


class RFValidationError(ValueError):
    """Custom exception raised when RF input parameters violate physical constraints or types."""
    pass


QUALITY_INTERPRETATIONS: Dict[str, str] = {
    "Excellent": (
        "RSSI >= -70 dBm: Outstanding signal quality. Supports maximum modulation coding schemes "
        "(MCS), high throughput, low latency, and zero measurable packet loss."
    ),
    "Good": (
        "RSSI -85 to -70 dBm: Strong, stable RF connection. Reliable for HD streaming, VoIP, "
        "and heavy data transfer with very rare frame retransmissions."
    ),
    "Fair": (
        "RSSI -100 to -85 dBm: Marginal signal quality typical of cell edges or obstructed paths. "
        "Throughput is throttled; occasional buffering and packet retries occur."
    ),
    "Poor": (
        "RSSI < -100 dBm: Extremely degraded or unusable signal. High frame error rate, frequent "
        "call drops, and connection time-outs likely. Consider boosting Tx power or relocating Rx antenna."
    ),
}


class RFPredictor:
    """Production predictor for RF signal levels with built-in validation."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path or MODEL_PATH)
        self._pipeline: Optional[Pipeline] = None

    @property
    def pipeline(self) -> Pipeline:
        """Lazy-loads the serialized model pipeline."""
        if self._pipeline is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model artifact not found at '{self.model_path}'. "
                    f"Please run model training first via 'python main.py --train'."
                )
            logger.debug(f"Loading serialized model pipeline from {self.model_path}...")
            self._pipeline = joblib.load(self.model_path)
        return self._pipeline

    @staticmethod
    def validate_input(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validates input dictionary against schema, types, and physical boundaries.
        
        Raises:
            RFValidationError: When inputs are missing, invalid, or out of physical bounds.
        """
        if not isinstance(params, dict):
            raise RFValidationError(f"Expected input to be a dictionary, got {type(params).__name__}.")

        validated: Dict[str, Any] = {}

        # 1. Check for missing required features
        missing_keys = [f for f in ALL_FEATURES if f not in params or params[f] is None]
        if missing_keys:
            raise RFValidationError(f"Missing required RF parameters: {missing_keys}")

        # 2. Validate Numerical Features
        for key in NUMERICAL_FEATURES:
            val = params[key]
            # Type and numeric check
            try:
                numeric_val = float(val)
            except (ValueError, TypeError):
                raise RFValidationError(
                    f"Parameter '{key}' must be numeric, received: {val} (type {type(val).__name__})"
                )

            if np.isnan(numeric_val) or np.isinf(numeric_val):
                raise RFValidationError(f"Parameter '{key}' cannot be NaN or infinite.")

            # Physical boundary checks
            if key in FEATURE_BOUNDS:
                low, high = FEATURE_BOUNDS[key]
                if not (low <= numeric_val <= high):
                    raise RFValidationError(
                        f"Parameter '{key}'={numeric_val} is outside physically plausible range "
                        f"[{low}, {high}]."
                    )
            
            # Additional integer constraint for obstacles
            if key == "obstacle_count":
                if int(numeric_val) != numeric_val or numeric_val < 0:
                    raise RFValidationError(
                        f"Parameter 'obstacle_count' must be a non-negative integer, got {numeric_val}."
                    )
                numeric_val = int(numeric_val)

            validated[key] = numeric_val

        # 3. Validate Categorical Features
        env = str(params.get("environment", "")).strip()
        # Case-insensitive matching with normalization
        matching_envs = [e for e in VALID_ENVIRONMENTS if e.lower() == env.lower()]
        if not matching_envs:
            raise RFValidationError(
                f"Invalid environment '{env}'. Must be one of {VALID_ENVIRONMENTS}."
            )
        validated["environment"] = matching_envs[0]

        weather = str(params.get("weather_condition", "")).strip()
        matching_weather = [w for w in VALID_WEATHER_CONDITIONS if w.lower() == weather.lower()]
        if not matching_weather:
            raise RFValidationError(
                f"Invalid weather_condition '{weather}'. Must be one of {VALID_WEATHER_CONDITIONS}."
            )
        validated["weather_condition"] = matching_weather[0]

        return validated

    def predict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts RF signal level for a single input dictionary with complete diagnostics."""
        clean_params = self.validate_input(params)
        df_single = pd.DataFrame([clean_params])

        # Prediction
        pred_label = self.pipeline.predict(df_single)[0]
        
        # Probabilities and Confidence
        classes = list(self.pipeline.classes_)
        probabilities = {}
        confidence = 0.0

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(df_single)[0]
            probabilities = {cls: float(round(p, 4)) for cls, p in zip(classes, probs)}
            confidence = float(round(np.max(probs), 4))

        return {
            "predicted_signal_level": str(pred_label),
            "confidence": confidence,
            "probabilities": probabilities,
            "interpretation": QUALITY_INTERPRETATIONS.get(str(pred_label), ""),
            "input_parameters": clean_params,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs batch prediction over a pandas DataFrame with validation."""
        if not isinstance(df, pd.DataFrame):
            raise RFValidationError("Batch input must be a pandas DataFrame.")

        # Validate each row
        validated_rows = []
        for idx, row in df.iterrows():
            try:
                val_row = self.validate_input(row.to_dict())
                validated_rows.append(val_row)
            except RFValidationError as e:
                raise RFValidationError(f"Row {idx} failed validation: {e}")

        df_clean = pd.DataFrame(validated_rows)
        preds = self.pipeline.predict(df_clean)
        
        result_df = df.copy()
        result_df["predicted_signal_level"] = preds
        
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(df_clean)
            for i, cls_name in enumerate(self.pipeline.classes_):
                result_df[f"prob_{cls_name.lower()}"] = np.round(probs[:, i], 4)
            result_df["prediction_confidence"] = np.round(np.max(probs, axis=1), 4)

        return result_df


# Module-level convenience singleton and function
_default_predictor: Optional[RFPredictor] = None


def predict_signal_level(
    frequency_mhz: float,
    distance_m: float,
    tx_power_dbm: float,
    tx_gain_dbi: float = 14.0,
    rx_gain_dbi: float = 2.0,
    tx_height_m: float = 25.0,
    rx_height_m: float = 1.5,
    environment: str = "Suburban",
    obstacle_count: int = 2,
    humidity_pct: float = 50.0,
    weather_condition: str = "Clear",
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Demonstration function: Accepts RF parameters and returns predicted signal level.
    
    Example:
        >>> result = predict_signal_level(
        ...     frequency_mhz=2100.0,
        ...     distance_m=450.0,
        ...     tx_power_dbm=43.0,
        ...     environment="Urban",
        ...     obstacle_count=3,
        ... )
        >>> print(result["predicted_signal_level"])
        'Good'
    """
    global _default_predictor
    if _default_predictor is None or model_path is not None:
        _default_predictor = RFPredictor(model_path=model_path)

    input_data = {
        "frequency_mhz": frequency_mhz,
        "distance_m": distance_m,
        "tx_power_dbm": tx_power_dbm,
        "tx_gain_dbi": tx_gain_dbi,
        "rx_gain_dbi": rx_gain_dbi,
        "tx_height_m": tx_height_m,
        "rx_height_m": rx_height_m,
        "obstacle_count": obstacle_count,
        "humidity_pct": humidity_pct,
        "environment": environment,
        "weather_condition": weather_condition,
    }

    return _default_predictor.predict(input_data)
