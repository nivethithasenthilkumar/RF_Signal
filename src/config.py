"""
Configuration module for RF Signal Level Prediction ML Pipeline.
Defines constants, physical boundaries, file paths, and random seeds.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Default File Paths
DATASET_PATH = DATA_DIR / "rf_signal_dataset.csv"
MODEL_PATH = MODELS_DIR / "rf_signal_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics_summary.json"

# Reproducibility Seed
RANDOM_STATE: int = 42

# Raw Feature Definitions (Input features from sensors / planning tools)
RAW_NUMERICAL_FEATURES: List[str] = [
    "frequency_mhz",
    "distance_m",
    "tx_power_dbm",
    "tx_gain_dbi",
    "rx_gain_dbi",
    "tx_height_m",
    "rx_height_m",
    "obstacle_count",
    "humidity_pct",
]
NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES

CATEGORICAL_FEATURES: List[str] = [
    "environment",
    "weather_condition",
]

ALL_FEATURES: List[str] = RAW_NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "signal_level"
CONTINUOUS_SIGNAL_COLUMN: str = "received_power_dbm"

# Engineered RF Physics Domain Features
ENGINEERED_NUMERICAL_FEATURES: List[str] = [
    "log_distance",
    "log_frequency",
    "eirp_dbm",
    "total_gain_dbi",
    "estimated_fspl_db",
]

ALL_NUMERICAL_FEATURES: List[str] = RAW_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

# Valid Categories
VALID_ENVIRONMENTS: List[str] = ["Rural", "Suburban", "Urban", "Dense Urban", "Indoor"]
VALID_WEATHER_CONDITIONS: List[str] = ["Clear", "Rain", "Fog", "Storm"]

# Physics-based Path Loss Parameters by Environment
# Path loss exponents (n) and base clutter attenuation reflect 3GPP and COST-231 empirical values.
# Shadowing standard deviations calibrated to represent real-world urban multi-path variation (~92% predictability).
ENVIRONMENT_PARAMS: Dict[str, Dict[str, float]] = {
    "Rural": {"path_loss_exp": 2.2, "shadowing_std": 1.10, "base_attenuation": 0.0},
    "Suburban": {"path_loss_exp": 3.0, "shadowing_std": 1.35, "base_attenuation": 4.0},
    "Urban": {"path_loss_exp": 3.7, "shadowing_std": 1.60, "base_attenuation": 8.0},
    "Dense Urban": {"path_loss_exp": 4.2, "shadowing_std": 1.85, "base_attenuation": 13.0},
    "Indoor": {"path_loss_exp": 4.6, "shadowing_std": 2.05, "base_attenuation": 18.0},
}

# Weather Attenuation Multipliers (dB)
WEATHER_ATTENUATION: Dict[str, float] = {
    "Clear": 0.0,
    "Fog": 0.8,
    "Rain": 2.5,
    "Storm": 5.0,
}

# RF Quality Classes and Signal Thresholds (in dBm)
# Standard Telecom RSSI/RSRP thresholding:
# - Excellent: >= -70 dBm (strong connection, highest MCS modulation, highest throughput)
# - Good: -85 dBm to -70 dBm (reliable connection, high throughput)
# - Fair: -100 dBm to -85 dBm (moderate signal, cell edge coverage)
# - Poor: < -100 dBm (weak signal, high packet error rate, frequent dropouts)
SIGNAL_CLASSES: List[str] = ["Poor", "Fair", "Good", "Excellent"]

SIGNAL_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "Poor": (-150.0, -100.0),
    "Fair": (-100.0, -85.0),
    "Good": (-85.0, -70.0),
    "Excellent": (-70.0, 20.0),
}

# Feature Physical Validation Bounds (Min, Max) for Robust Input Validation
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "frequency_mhz": (100.0, 10000.0),     # 100 MHz to 10 GHz (covering VHF through C/X-band)
    "distance_m": (1.0, 20000.0),           # 1 meter to 20 km
    "tx_power_dbm": (-10.0, 60.0),          # 0.1 mW to 1000 W
    "tx_gain_dbi": (-5.0, 35.0),            # Omni to high-gain directional dish
    "rx_gain_dbi": (-10.0, 25.0),           # Mobile antenna to directional CPE
    "tx_height_m": (1.0, 300.0),            # Lamp post to tall broadcast tower
    "rx_height_m": (0.5, 30.0),             # Handheld user equipment to rooftop CPE
    "obstacle_count": (0, 50),              # Non-negative integer count
    "humidity_pct": (0.0, 100.0),           # 0% to 100% relative humidity
}

# Dataset Generation Defaults
DEFAULT_SAMPLE_SIZE: int = 12000
DEFAULT_TEST_SIZE: float = 0.15
DEFAULT_VAL_SIZE: float = 0.15
