"""
CLI Entry Point for the RF Signal Level Prediction ML Pipeline.
Enables dataset generation, model training, evaluation, and inference.

Usage:
    python main.py --train
    python main.py --predict
    python main.py --generate-data
    python main.py --demo-errors
"""

import argparse
import json
import sys
from pprint import pprint

from src.config import DATASET_PATH, MODEL_PATH, SIGNAL_CLASSES
from src.data_generator import generate_and_save_data
from src.predict import RFPredictor, RFValidationError, predict_signal_level
from src.train import train_pipeline
from src.utils import setup_logger

logger = setup_logger("main")


def run_sample_predictions():
    """Demonstrates inference across multiple realistic RF operating conditions."""
    logger.info("=== Running Sample RF Predictions ===")

    scenarios = [
        {
            "name": "Scenario 1: Close Proximity / Line-of-Sight Rural",
            "params": {
                "frequency_mhz": 850.0,
                "distance_m": 120.0,
                "tx_power_dbm": 43.0,
                "tx_gain_dbi": 15.0,
                "rx_gain_dbi": 2.0,
                "tx_height_m": 35.0,
                "rx_height_m": 1.5,
                "environment": "Rural",
                "obstacle_count": 0,
                "humidity_pct": 40.0,
                "weather_condition": "Clear",
            },
        },
        {
            "name": "Scenario 2: Urban Macro Cell (Moderate Distance & Clutter)",
            "params": {
                "frequency_mhz": 2100.0,
                "distance_m": 650.0,
                "tx_power_dbm": 40.0,
                "tx_gain_dbi": 16.0,
                "rx_gain_dbi": 2.0,
                "tx_height_m": 30.0,
                "rx_height_m": 1.5,
                "environment": "Urban",
                "obstacle_count": 4,
                "humidity_pct": 65.0,
                "weather_condition": "Rain",
            },
        },
        {
            "name": "Scenario 3: Dense Urban Cell Edge (High Obstacles & Rain)",
            "params": {
                "frequency_mhz": 2600.0,
                "distance_m": 1800.0,
                "tx_power_dbm": 38.0,
                "tx_gain_dbi": 14.0,
                "rx_gain_dbi": 0.0,
                "tx_height_m": 25.0,
                "rx_height_m": 1.5,
                "environment": "Dense Urban",
                "obstacle_count": 9,
                "humidity_pct": 85.0,
                "weather_condition": "Rain",
            },
        },
        {
            "name": "Scenario 4: Deep Indoor Penetration / Severe Attenuation",
            "params": {
                "frequency_mhz": 3500.0,
                "distance_m": 2500.0,
                "tx_power_dbm": 30.0,
                "tx_gain_dbi": 8.0,
                "rx_gain_dbi": 0.0,
                "tx_height_m": 15.0,
                "rx_height_m": 1.2,
                "environment": "Indoor",
                "obstacle_count": 7,
                "humidity_pct": 70.0,
                "weather_condition": "Storm",
            },
        },
    ]

    for sc in scenarios:
        print("\n" + "=" * 60)
        print(f"[{sc['name']}]")
        print("Input Parameters:")
        for k, v in sc["params"].items():
            print(f"  {k:20s}: {v}")

        res = predict_signal_level(**sc["params"])
        print("\nPrediction Result:")
        print(f"  Signal Level: {res['predicted_signal_level'].upper()} (Confidence: {res['confidence']:.2%})")
        print("  Class Probabilities:")
        for cls_name in SIGNAL_CLASSES:
            prob = res["probabilities"].get(cls_name, 0.0)
            bar = "#" * int(prob * 30)
            print(f"    {cls_name:10s}: {prob:6.2%} | {bar}")
        print(f"  Interpretation: {res['interpretation']}")


def run_error_handling_demo():
    """Demonstrates how the system handles invalid, corrupted, or out-of-bounds inputs."""
    logger.info("=== Demonstrating Error Handling & Input Validation ===")
    predictor = RFPredictor()

    invalid_test_cases = [
        (
            "Negative Distance (Physical Violation)",
            {"frequency_mhz": 1800.0, "distance_m": -50.0, "tx_power_dbm": 43.0,
             "tx_gain_dbi": 10.0, "rx_gain_dbi": 2.0, "tx_height_m": 30.0, "rx_height_m": 1.5,
             "environment": "Urban", "obstacle_count": 2, "humidity_pct": 50.0, "weather_condition": "Clear"},
        ),
        (
            "Frequency Out-of-Bounds (Extreme RF)",
            {"frequency_mhz": 99999.0, "distance_m": 200.0, "tx_power_dbm": 43.0,
             "tx_gain_dbi": 10.0, "rx_gain_dbi": 2.0, "tx_height_m": 30.0, "rx_height_m": 1.5,
             "environment": "Urban", "obstacle_count": 2, "humidity_pct": 50.0, "weather_condition": "Clear"},
        ),
        (
            "Unrecognized Environment Category",
            {"frequency_mhz": 2100.0, "distance_m": 300.0, "tx_power_dbm": 43.0,
             "tx_gain_dbi": 10.0, "rx_gain_dbi": 2.0, "tx_height_m": 30.0, "rx_height_m": 1.5,
             "environment": "DeepUnderwaterCave", "obstacle_count": 2, "humidity_pct": 50.0, "weather_condition": "Clear"},
        ),
        (
            "Missing Required Features",
            {"frequency_mhz": 2100.0, "distance_m": 300.0},
        ),
        (
            "Invalid Obstacle Count (Negative float)",
            {"frequency_mhz": 1800.0, "distance_m": 200.0, "tx_power_dbm": 43.0,
             "tx_gain_dbi": 10.0, "rx_gain_dbi": 2.0, "tx_height_m": 30.0, "rx_height_m": 1.5,
             "environment": "Urban", "obstacle_count": -3, "humidity_pct": 50.0, "weather_condition": "Clear"},
        ),
    ]

    for desc, payload in invalid_test_cases:
        print(f"\nTesting: {desc}")
        try:
            predictor.predict(payload)
            print("  [FAIL] Expected validation error was NOT raised!")
        except RFValidationError as e:
            print(f"  [SUCCESS] Gracefully caught expected error:\n    --> {e}")
        except Exception as e:
            print(f"  [FAIL] Unexpected exception type ({type(e).__name__}): {e}")


def main():
    parser = argparse.ArgumentParser(
        description="RF Signal Level Prediction - Machine Learning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", action="store_true", help="Execute model training and evaluation")
    parser.add_argument("--tune", action="store_true", help="Perform grid search hyperparameter tuning")
    parser.add_argument("--generate-data", action="store_true", help="Generate fresh synthetic RF dataset")
    parser.add_argument("--predict", action="store_true", help="Run sample inference scenarios")
    parser.add_argument("--demo-errors", action="store_true", help="Run input validation & edge-case demo")
    parser.add_argument("--samples", type=int, default=12000, help="Number of synthetic samples to generate")

    args = parser.parse_args()

    # If no flags passed, run default full workflow
    if not (args.train or args.generate_data or args.predict or args.demo_errors):
        logger.info("No specific action flag supplied. Running end-to-end workflow: Data Gen -> Train -> Evaluate -> Predict.")
        args.train = True
        args.predict = True
        args.demo_errors = True

    if args.generate_data:
        logger.info(f"Generating {args.samples} synthetic RF samples...")
        df = generate_and_save_data(n_samples=args.samples)
        logger.info(f"Generated dataset saved to {DATASET_PATH} (Shape: {df.shape})")

    if args.train:
        logger.info("Starting RF Signal Level ML Pipeline Training...")
        pipeline, metrics = train_pipeline(tune_hyperparameters=args.tune)
        print("\n" + "=" * 50)
        print("MODEL TRAINING & EVALUATION COMPLETED")
        print("=" * 50)
        print(f"Test Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Macro Precision:        {metrics['precision_macro']:.4f}")
        print(f"Macro Recall:           {metrics['recall_macro']:.4f}")
        print(f"Macro F1-Score:         {metrics['f1_macro']:.4f}")
        print(f"Weighted F1-Score:      {metrics['f1_weighted']:.4f}")
        if "roc_auc_ovr_macro" in metrics:
            print(f"Multiclass ROC-AUC:     {metrics['roc_auc_ovr_macro']:.4f}")
        print("=" * 50)

    if args.predict:
        run_sample_predictions()

    if args.demo_errors:
        run_error_handling_demo()


if __name__ == "__main__":
    main()
