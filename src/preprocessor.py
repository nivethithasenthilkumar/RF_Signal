"""
Preprocessing and feature engineering pipeline for RF Signal prediction.
Provides data splitting, imputation, scaling, and categorical encoding
following strict leak-free ML best practices.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_FEATURES,
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    RAW_NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.utils import setup_logger

logger = setup_logger(__name__)


class RawImputerTransformer(BaseEstimator, TransformerMixin):
    """Handles missing values in raw RF features prior to physics feature engineering.
    
    - Raw Numerical: Imputes with training set median
    - Raw Categorical: Imputes with training set most frequent value
    """

    def __init__(self):
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=ALL_FEATURES)
        self.num_imputer.fit(X[RAW_NUMERICAL_FEATURES])
        self.cat_imputer.fit(X[CATEGORICAL_FEATURES])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=ALL_FEATURES)
        X_num = self.num_imputer.transform(X[RAW_NUMERICAL_FEATURES])
        X_cat = self.cat_imputer.transform(X[CATEGORICAL_FEATURES])

        df_num = pd.DataFrame(X_num, columns=RAW_NUMERICAL_FEATURES, index=X.index)
        df_cat = pd.DataFrame(X_cat, columns=CATEGORICAL_FEATURES, index=X.index)
        return pd.concat([df_num, df_cat], axis=1)


class RFFeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Scikit-Learn Transformer for domain-specific RF feature engineering.
    
    Computes key wireless communication metrics:
    - Logarithmic Distance: log10(distance_m) to match log-distance path loss physics
    - Logarithmic Frequency: log10(frequency_mhz) to match free space path loss
    - EIRP (Effective Isotropic Radiated Power): tx_power_dbm + tx_gain_dbi
    - Total Antenna Gain: tx_gain_dbi + rx_gain_dbi
    - Estimated Free Space Path Loss (FSPL): 20*log10(f) + 20*log10(d) - 27.55
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=ALL_FEATURES)

        X_out = X.copy()

        # Handle numeric conversions safely
        dist = np.maximum(pd.to_numeric(X_out["distance_m"], errors="coerce").values, 1.0)
        freq = np.maximum(pd.to_numeric(X_out["frequency_mhz"], errors="coerce").values, 10.0)
        tx_p = pd.to_numeric(X_out["tx_power_dbm"], errors="coerce").values
        tx_g = pd.to_numeric(X_out["tx_gain_dbi"], errors="coerce").values
        rx_g = pd.to_numeric(X_out["rx_gain_dbi"], errors="coerce").values

        log_dist = np.log10(dist)
        log_freq = np.log10(freq)
        eirp = tx_p + tx_g
        total_gain = tx_g + rx_g
        fspl = 20.0 * log_freq + 20.0 * log_dist - 27.55

        X_out["log_distance"] = np.round(log_dist, 4)
        X_out["log_frequency"] = np.round(log_freq, 4)
        X_out["eirp_dbm"] = np.round(eirp, 2)
        X_out["total_gain_dbi"] = np.round(total_gain, 2)
        X_out["estimated_fspl_db"] = np.round(fspl, 2)

        return X_out


def build_preprocessor_pipeline(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Builds a Scikit-Learn ColumnTransformer for numerical & categorical features."""
    num_cols = numerical_features or ALL_NUMERICAL_FEATURES
    cat_cols = categorical_features or CATEGORICAL_FEATURES

    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_full_preprocessing_pipeline() -> Pipeline:
    """Chains Raw Imputation, Physics Feature Engineering, and Scaling/Encoding."""
    return Pipeline([
        ("imputer", RawImputerTransformer()),
        ("fe", RFFeatureEngineeringTransformer()),
        ("preprocessor", build_preprocessor_pipeline(ALL_NUMERICAL_FEATURES, CATEGORICAL_FEATURES)),
    ])


def split_data(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = RANDOM_STATE,
    stratify_col: Optional[str] = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Splits dataset into stratified Train, Validation, and Test sets.
    
    Split occurs BEFORE any transformation to guarantee zero data leakage.
    Default split ratio: 70% Train, 15% Validation, 15% Test.
    """
    # Drop rows where target is missing if any
    clean_df = df.dropna(subset=[stratify_col]).copy() if stratify_col in df.columns else df.copy()

    X = clean_df[ALL_FEATURES]
    y = clean_df[stratify_col]

    stratify_target = y if stratify_col is not None else None

    # First split: Train+Val vs Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Second split: Train vs Val (adjust relative proportion)
    val_relative_size = val_size / (1.0 - test_size)
    stratify_train_val = y_train_val if stratify_col is not None else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=stratify_train_val,
    )

    logger.info(
        f"Data split complete: Train={len(X_train)} ({len(X_train)/len(clean_df):.1%}), "
        f"Val={len(X_val)} ({len(X_val)/len(clean_df):.1%}), "
        f"Test={len(X_test)} ({len(X_test)/len(clean_df):.1%})"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_feature_names_out(pipeline: Pipeline) -> List[str]:
    """Retrieves post-transformation feature names from a fitted pipeline's preprocessor."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())
    return ALL_NUMERICAL_FEATURES + CATEGORICAL_FEATURES
