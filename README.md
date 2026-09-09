# Wi-Fi-Indoor-Localization-and-Signal-Coverage-Analysis
# Wi-Fi Indoor Localization and Signal Coverage Analysis

An ML-powered indoor positioning dashboard that predicts which floor a device is on using only Wi-Fi signal strength (RSS) readings — no GPS required. Also maps signal coverage across a building and flags weak-signal zones for optimal router placement.

Built with Streamlit, scikit-learn, and the UJIIndoorLoc dataset.

## What it does

- **Floor prediction**: Trains Random Forest and KNN classifiers on Wi-Fi fingerprint data (520 access points) to predict which floor a device is on.
- **Signal coverage mapping**: Grids and smooths signal strength across the building, flags weak-coverage zones, and suggests router placement using clustering.
- **Interactive dashboard**: Tune model hyperparameters (tree count, depth, k-neighbors) live, view confusion matrices and classification reports.
- **Simulated real-time monitor**: Replays sensor readings with added noise to demonstrate live floor-tracking behavior.

## Dataset

This project uses the **UJIIndoorLoc** dataset from the UCI Machine Learning Repository. The training file is too large to include in this repo (~40 MB), so you'll need to download it separately:

1. Go to: https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc
2. Download `TrainingData.csv` (and optionally `ValidationData.csv`)
3. Upload them via the sidebar when you run the app

## Setup

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### Requirements

```
streamlit
pandas
numpy
scikit-learn
scipy
matplotlib
```

## Usage

```bash
streamlit run main.py
```

Then open the local URL Streamlit gives you, and upload `TrainingData.csv` in the sidebar to get started.

## How it works

**Data cleaning**: The dataset marks "router not detected" as `100`. This is replaced with `-110 dBm` (a realistic very-weak-signal value) so the model doesn't mistake it for a strong signal. Low-variance WAP columns are dropped since they carry no useful signal.

**Models**:
- *Random Forest* — an ensemble of decision trees voting on the majority prediction; robust to noise, gives feature importance.
- *KNN* — finds the k most similar past readings and votes on their floor label.

**Coverage mapping**: Readings are plotted on a lat/lon grid, averaged per cell, and Gaussian-smoothed. Cells below a configurable signal threshold are flagged as weak zones, then clustered (BFS flood-fill) into suggested router placement locations.

**Real-time simulation**: Replays dataset rows at a timed interval with added Gaussian noise to simulate natural signal fluctuation, running each sample through the trained model live. Clearly labeled as simulated — no live hardware involved.

## Tech Stack

Python · scikit-learn · Streamlit · pandas · NumPy · SciPy · Matplotlib

## Note

The training/validation CSVs are **not included** in this repo due to file size. See the Dataset section above to get them.
