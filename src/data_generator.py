"""
RF Signal Data Generator Module.
Generates realistic synthetic radio frequency datasets grounded in wireless
propagation physics (Log-Distance Path Loss, Shadowing, and Environmental Factors).
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURES,
    CONTINUOUS_SIGNAL_COLUMN,
    ENVIRONMENT_PARAMS,
    RANDOM_STATE,
    SIGNAL_CLASSES,
    SIGNAL_THRESHOLDS,
    TARGET_COLUMN,
    VALID_ENVIRONMENTS,
    VALID_WEATHER_CONDITIONS,
    WEATHER_ATTENUATION,
)


class RFDataGenerator:
    """Physics-grounded generator for RF signal propagation data.
    
    Simulates received signal strength (P_rx in dBm) based on:
    - Free Space Path Loss at reference distance d0 = 1 m
    - Log-Distance Path Loss model with environment-dependent path loss exponents
    - Log-normal shadowing (large-scale fading)
    - Obstacle and wall penetration attenuation
    - Atmospheric and weather losses
    - Antenna height gain corrections
    - Multi-path / measurement fluctuation noise
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def generate_features(self, n_samples: int) -> pd.DataFrame:
        """Generates random feature vectors within realistic RF engineering distributions."""
        # Common frequency bands: 700MHz, 850MHz, 900MHz, 1800MHz, 2100MHz, 2400MHz, 2600MHz, 3500MHz, 5200MHz, 5800MHz
        freq_bands = np.array([700, 850, 900, 1800, 2100, 2400, 2600, 3500, 5000, 5800])
        band_indices = self.rng.choice(len(freq_bands), size=n_samples)
        jitter = self.rng.uniform(-30.0, 30.0, size=n_samples)
        frequency_mhz = np.clip(freq_bands[band_indices] + jitter, 600.0, 6000.0)

        # Distance: Log-uniform distribution from 10m to 5000m (denser near cell, sparse far away)
        distance_m = np.exp(self.rng.uniform(np.log(10.0), np.log(5000.0), size=n_samples))

        # Transmit power (dBm): Macro cells (40-46 dBm), Micro/Small cells (20-36 dBm), Wi-Fi (14-23 dBm)
        tx_power_dbm = self.rng.choice([20.0, 23.0, 30.0, 37.0, 43.0, 46.0], size=n_samples) + \
                       self.rng.normal(0, 1.5, size=n_samples)
        tx_power_dbm = np.clip(tx_power_dbm, 10.0, 50.0)

        # Antenna gains (dBi)
        tx_gain_dbi = self.rng.uniform(2.0, 18.0, size=n_samples)
        rx_gain_dbi = self.rng.uniform(0.0, 5.0, size=n_samples)

        # Antenna heights (m)
        tx_height_m = self.rng.uniform(10.0, 50.0, size=n_samples)
        rx_height_m = self.rng.uniform(1.2, 2.5, size=n_samples)

        # Environment
        environment = self.rng.choice(VALID_ENVIRONMENTS, size=n_samples, p=[0.20, 0.25, 0.25, 0.15, 0.15])

        # Obstacles count (dependent on environment)
        obstacle_count = np.zeros(n_samples, dtype=int)
        for i, env in enumerate(environment):
            if env == "Rural":
                obstacle_count[i] = self.rng.integers(0, 3)
            elif env == "Suburban":
                obstacle_count[i] = self.rng.integers(1, 6)
            elif env == "Urban":
                obstacle_count[i] = self.rng.integers(3, 10)
            elif env == "Dense Urban":
                obstacle_count[i] = self.rng.integers(5, 14)
            else:  # Indoor
                obstacle_count[i] = self.rng.integers(2, 8)

        # Weather and Atmospheric Conditions
        weather_condition = self.rng.choice(
            VALID_WEATHER_CONDITIONS, size=n_samples, p=[0.60, 0.20, 0.12, 0.08]
        )
        humidity_pct = np.clip(self.rng.normal(55.0, 20.0, size=n_samples), 10.0, 98.0)

        df = pd.DataFrame({
            "frequency_mhz": np.round(frequency_mhz, 2),
            "distance_m": np.round(distance_m, 2),
            "tx_power_dbm": np.round(tx_power_dbm, 2),
            "tx_gain_dbi": np.round(tx_gain_dbi, 2),
            "rx_gain_dbi": np.round(rx_gain_dbi, 2),
            "tx_height_m": np.round(tx_height_m, 2),
            "rx_height_m": np.round(rx_height_m, 2),
            "obstacle_count": obstacle_count,
            "humidity_pct": np.round(humidity_pct, 1),
            "environment": environment,
            "weather_condition": weather_condition,
        })
        return df

    def compute_received_power(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates received power P_rx (dBm) and assigns discrete signal level classes."""
        n_samples = len(df)
        p_rx = np.zeros(n_samples)

        # Free space path loss at reference distance d0 = 1m:
        # PL0 = 20*log10(f_MHz) - 27.55
        pl_0 = 20.0 * np.log10(df["frequency_mhz"].values) - 27.55

        # Antenna height correction (Hata-like clearance advantage)
        height_gain = (
            10.0 * np.log10(np.maximum(df["tx_height_m"].values / 10.0, 0.1))
            + 3.0 * np.log10(np.maximum(df["rx_height_m"].values / 1.5, 0.1))
        )

        for i in range(n_samples):
            env = df["environment"].iloc[i]
            params = ENVIRONMENT_PARAMS[env]
            n_exp = params["path_loss_exp"]
            base_att = params["base_attenuation"]

            # Log-distance path loss
            dist = max(df["distance_m"].iloc[i], 1.0)
            pl_dist = 10.0 * n_exp * np.log10(dist)

            # Obstacle attenuation (average 2.4 dB per obstacle/wall)
            obs_loss = df["obstacle_count"].iloc[i] * 2.4

            # Weather and humidity attenuation
            weather = df["weather_condition"].iloc[i]
            freq_ghz = df["frequency_mhz"].iloc[i] / 1000.0
            weather_loss = WEATHER_ATTENUATION[weather] * (freq_ghz / 2.0) ** 0.5 * (
                1.0 + (df["humidity_pct"].iloc[i] - 50.0) / 200.0
            )

            # Log-normal Shadowing (large-scale fading calibrated for realistic channel variation)
            shadowing = self.rng.normal(0.0, params["shadowing_std"])

            # Small-scale fading / measurement jitter
            fast_fading = self.rng.normal(0.0, 0.5)

            # Link Budget Calculation
            p_rx[i] = (
                df["tx_power_dbm"].iloc[i]
                + df["tx_gain_dbi"].iloc[i]
                + df["rx_gain_dbi"].iloc[i]
                - pl_0[i]
                - pl_dist
                - base_att
                - obs_loss
                - weather_loss
                + height_gain[i]
                + shadowing
                + fast_fading
            )

        # Categorize into signal levels
        signal_level = np.empty(n_samples, dtype=object)
        for label, (low, high) in SIGNAL_THRESHOLDS.items():
            mask = (p_rx >= low) & (p_rx < high)
            signal_level[mask] = label

        # Fallbacks for extreme outliers
        signal_level[p_rx >= -70.0] = "Excellent"
        signal_level[p_rx < -100.0] = "Poor"

        return np.round(p_rx, 2), signal_level

    def generate_dataset(
        self,
        n_samples: int = 12000,
        inject_missing_pct: float = 0.01,
    ) -> pd.DataFrame:
        """Generates a complete RF signal dataset with optional realistic missing values."""
        df = self.generate_features(n_samples)
        p_rx, signal_level = self.compute_received_power(df)

        df[CONTINUOUS_SIGNAL_COLUMN] = p_rx
        df[TARGET_COLUMN] = signal_level

        # Realistic edge-case simulation: Inject occasional missing values to test robust preprocessor
        if inject_missing_pct > 0:
            mask_num = self.rng.uniform(0, 1, size=(n_samples, len(df.columns) - 2)) < inject_missing_pct
            for col_idx, col in enumerate(df.columns[:-2]):
                df.loc[mask_num[:, col_idx], col] = np.nan

        return df


def generate_and_save_data(
    filepath: Optional[str] = None,
    n_samples: int = 12000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generates dataset and writes to disk."""
    generator = RFDataGenerator(random_state=random_state)
    df = generator.generate_dataset(n_samples=n_samples, inject_missing_pct=0.005)
    
    target_path = filepath or DATASET_PATH
    df.to_csv(target_path, index=False)
    return df


if __name__ == "__main__":
    from src.config import DATASET_PATH
    print(f"Generating dataset at {DATASET_PATH}...")
    df = generate_and_save_data()
    print(f"Generated dataset shape: {df.shape}")
    print("Class distribution:")
    print(df[TARGET_COLUMN].value_counts(dropna=False))
