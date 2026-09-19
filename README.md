# RF Signal Level Prediction - Machine Learning System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A production-ready Machine Learning system that predicts Radio Frequency (RF) signal quality levels (`Poor`, `Fair`, `Good`, `Excellent`) grounded in wireless propagation physics (Log-Distance Path Loss, Shadowing, Clutter, and Atmospheric Attenuation). 

The system achieves **~90.3% - 91.5% classification accuracy** and **0.982 multiclass ROC-AUC** with strict featurization ordering (zero data leakage), robust input boundary validation, and a Scikit-Learn pipeline ready for edge and cloud deployment.

---

## 📡 Wireless Propagation Physics & Architecture

### 1. Link Budget & Path Loss Formulation
The received signal power $P_{rx}$ (dBm) is calculated according to the empirical Log-Distance Path Loss model combined with log-normal shadowing and environmental clutter factors:

$$P_{rx} = P_{tx} + G_{tx} + G_{rx} - PL(d, f) - L_{env} - L_{obs} - L_{weather} + G_{height} + \chi_\sigma + \epsilon$$

Where:
- **$P_{tx}$**: Transmitter power (dBm)
- **$G_{tx}, G_{rx}$**: Transmitter and receiver antenna gains (dBi)
- **$PL(d, f)$**: Log-distance path loss:
  $$PL(d, f) = 20\log_{10}(f_{\text{MHz}}) + 10 \cdot n_{\text{env}} \cdot \log_{10}(d) - 27.55$$
  with environment-dependent path loss exponents $n_{\text{env}} \in [2.2, 4.6]$ (Rural, Suburban, Urban, Dense Urban, Indoor).
- **$L_{env}$**: Base clutter loss per morphology (e.g., +8 dB for Urban, +18 dB for Indoor).
- **$L_{obs}$**: Obstacle and penetration loss (~2.4 dB per physical obstruction/wall).
- **$L_{weather}$**: Frequency-dependent rain and fog attenuation scaled by relative humidity.
- **$G_{height}$**: Antenna elevation clearance correction (Hata-Okumura model).
- **$\chi_\sigma \sim \mathcal{N}(0, \sigma^2)$**: Log-normal shadowing (large-scale fading).
- **$\epsilon$**: Multi-path fluctuation / fast-fading jitter.

### 2. Signal Quality Classification Tiers
Standard telecom RSSI/RSRP thresholds:
- **`Excellent` ($\ge -70\text{ dBm}$)**: Peak spectral efficiency, maximum modulation coding schemes (MCS 28/64-QAM+), low latency, zero packet loss.
- **`Good` ($-85\text{ to } -70\text{ dBm}$)**: Stable high-throughput link, suitable for HD streaming and voice with minimal retransmissions.
- **`Fair` ($-100\text{ to } -85\text{ dBm}$)**: Cell-edge condition, adaptive modulation throttled, occasional packet drops.
- **`Poor` ($< -100\text{ dBm}$)**: Marginal / degraded link, high Frame Error Rate (FER), frequent disconnects.

---

## 📊 Model Performance & Diagnostics

The production model evaluates an optimized **Histogram-Based Gradient Boosting Classifier** (`HistGradientBoostingClassifier`) with automated RF domain feature engineering (`log_distance`, `log_frequency`, `eirp_dbm`, `estimated_fspl_db`):

| Metric | Score |
|---|---|
| **Test Accuracy** | **90.28%** |
| **Weighted F1-Score** | **0.9020** |
| **Macro Precision** | **0.8519** |
| **Macro Recall** | **0.8460** |
| **Multiclass ROC-AUC (OvR)** | **0.9821** |

### Per-Class Performance
| Signal Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Excellent** | 0.94 | 0.97 | 0.95 | 659 |
| **Good** | 0.75 | 0.73 | 0.74 | 248 |
| **Fair** | 0.75 | 0.73 | 0.74 | 216 |
| **Poor** | 0.97 | 0.96 | 0.96 | 677 |

### Evaluation Visualizations
Generated artifacts located in `reports/`:
- **`confusion_matrix.png`**: Dual raw counts and normalized percentage heatmaps.
- **`roc_curves.png`**: Multiclass One-vs-Rest ROC curves (AUC > 0.96 for all classes).
- **`pr_curves.png`**: Precision-Recall curves displaying Average Precision (AP) scores.
- **`feature_importance.png`**: Permutation feature importance on the held-out test set showing `distance_m`, `environment`, and `tx_power_dbm` as leading drivers.

---

## 🛠️ Project Structure

```
R_F SIGNAL/
├── data/
│   └── rf_signal_dataset.csv      # Generated RF dataset (12,000 samples)
├── models/
│   └── rf_signal_model.joblib     # Serialized end-to-end inference pipeline
├── reports/
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   ├── roc_curves.png             # Multiclass ROC curves
│   ├── pr_curves.png              # Precision-Recall curves
│   ├── feature_importance.png     # Feature importance bar plot
│   └── metrics_summary.json       # JSON export of test metrics
├── src/
│   ├── __init__.py                # Package exports
│   ├── config.py                  # Physical constants, feature boundaries, thresholds
│   ├── data_generator.py          # Physics-grounded RF dataset generator
│   ├── preprocessor.py            # Imputation, feature engineering & scaling transformers
│   ├── train.py                   # Baseline comparison, training & serialization
│   ├── evaluate.py                # Metric computation and visualization generators
│   ├── predict.py                 # Production inference engine with boundary validation
│   └── utils.py                   # Logging, JSON I/O, and directory management
├── tests/
│   ├── __init__.py
│   ├── test_data_generator.py     # Tests for synthetic data generation & physics
│   ├── test_preprocessor.py       # Tests for leak-free transformations & imputation
│   ├── test_model.py              # Tests for model accuracy & probability sums
│   └── test_prediction.py         # Tests for inference API & validation edge cases
├── main.py                        # Unified CLI runner
├── requirements.txt               # Pinned dependencies
└── README.md                      # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Clone or enter the project directory and set up a virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
Execute data generation, model training, metric evaluation, sample inference, and input validation test:

```bash
python main.py
```

### 3. Individual CLI Operations

```bash
# Retrain model and re-generate diagnostic reports
python main.py --train

# Run sample inference across 4 realistic RF operating scenarios
python main.py --predict

# Run input validation and edge-case boundary checks
python main.py --demo-errors

# Generate a fresh synthetic dataset with custom sample count
python main.py --generate-data --samples 15000
```

### 4. Run Automated Test Suite

```bash
pytest -v
```

---

## 💻 Python Inference Example

```python
from src.predict import predict_signal_level

# Predict signal quality for an Urban LTE cell
result = predict_signal_level(
    frequency_mhz=2100.0,
    distance_m=650.0,
    tx_power_dbm=40.0,
    tx_gain_dbi=16.0,
    rx_gain_dbi=2.0,
    tx_height_m=30.0,
    rx_height_m=1.5,
    environment="Urban",
    obstacle_count=4,
    humidity_pct=65.0,
    weather_condition="Rain",
)

print(f"Predicted Signal: {result['predicted_signal_level']}")
print(f"Confidence:       {result['confidence']:.2%}")
print(f"Probabilities:    {result['probabilities']}")
print(f"Interpretation:   {result['interpretation']}")
```

### Output:
```json
{
  "predicted_signal_level": "Poor",
  "confidence": 0.5812,
  "probabilities": {
    "Poor": 0.5812,
    "Fair": 0.4141,
    "Good": 0.0042,
    "Excellent": 0.0005
  },
  "interpretation": "RSSI < -100 dBm: Extremely degraded or unusable signal. High frame error rate, frequent call drops, and connection time-outs likely. Consider boosting Tx power or relocating Rx antenna."
}
```

---

## 🛡️ Robustness & Input Validation

The system guards against corrupted or physically implausible parameters using `RFValidationError`:
- **Physical Boundaries**: Rejects negative distances ($d \le 0$), out-of-band frequencies ($f \notin [100, 10000]$ MHz), and invalid humidity percentages ($H \notin [0, 100]\%$).
- **Type Checking & NaN Prevention**: Rejects infinite or non-numeric entries with explicit diagnostic descriptions.
- **Categorical Normalization**: Validates environmental classifications and weather conditions with case-insensitivity.
- **Missing Value Imputation**: The pipeline automatically imputes missing inputs using training set medians/modes before computing derived RF physics quantities.

---

## 📄 License
This project is licensed under the MIT License.
