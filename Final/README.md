# Hybrid Intrusion Detection System for CAN Bus Networks

This project implements a hybrid IDS for CAN bus networks with real-time simulation and Streamlit dashboard visualization.

## Features

- **Preprocessing**: Handles missing values, converts HEX CAN IDs to integers, normalizes timestamps to time differences, encodes labels.
- **Feature Engineering**: Message frequency, time intervals, rolling window statistics, data entropy, spike detection.
- **Hybrid Detection**:
  - Rule-based: Detects DoS (high frequency), Spoofing (unknown CAN ID), Fuzzy (high entropy).
  - Machine Learning: Random Forest classifier.
  - Decision Fusion: OR logic for final detection.
- **Real-time Simulation**: Streams dataset packets with configurable delay.
- **Dashboard**: Live packet table, real-time alerts, packet frequency graph, attack vs normal ratio.
- **Logging**: Saves alerts to CSV.

## Project Structure

- `preprocessing.py`: Data loading and preprocessing.
- `feature_engineering.py`: Feature creation.
- `rules.py`: Rule-based detection.
- `model.py`: ML model training and prediction.
- `realtime_simulation.py`: Real-time streaming simulation.
- `app.py`: Streamlit dashboard.
- `sample_data.csv`: Sample dataset.
- `alerts.csv`: Logged alerts.

## Installation

1. Configure Python environment (virtualenv created).
2. Install dependencies: `pip install pandas scikit-learn streamlit plotly`

## Usage

1. Place the Kaggle Car Hacking dataset CSV files in the `data/` folder. The loader supports multiple attack CSVs and merges them into one dataset.
2. Valid file names include: `dos`, `fuzzy`, `rpm`, `gear`, `normal`.
3. Run the dashboard: `streamlit run app.py`
4. Use the sidebar to adjust simulation speed and replay the attack scenario.

## Advanced System

- Multi-class detection: Normal, DoS, Fuzzy, RPM, Gear
- Advanced feature engineering with sliding windows, payload statistics, inter-arrival jitter, burst detection, and pattern consistency
- Hybrid fusion: rule engine + Random Forest + XGBoost (if available) + Isolation Forest anomaly detection
- Explainability: feature importance and per-sample reasoning
- Real-time dashboard with attack timeline, confidence gauge, suspicious CAN IDs, severity filtering, and source breakdown

## GitHub Backup and Version Control

1. Initialize Git (if not already initialized):
   - `git init`
   - `git remote add origin https://github.com/asmita-21-12/Hybrid-Intrusion-Detection-System-for-CAN-Bus-Networks-CANShield.git`
   - `git add .`
   - `git commit -m "Initial commit: Hybrid IDS for CAN Bus Networks"`
   - `git push -u origin main`

2. Backup script:
   - `python github_backup.py`
   - `python github_backup.py "Updated IDS model"`
   - `python github_backup.py --branch experiment --experiment "Experimental detection improvements"`

3. Manual push function:
   - Use `from github_backup import push_to_github`
   - Call `push_to_github('Custom backup message')`

4. Automatic backup hooks:
   - Use `backup_after_training()` after model training.
   - Use `backup_after_simulation()` after simulation runs.

## Notes

- The project includes `.gitignore` to exclude large or temporary files:
  - `data/raw/`
  - `backups/`
  - `logs/`
  - `__pycache__/`
  - `*.pkl`
  - `*.zip`
  - `.env`

- Do not store large raw dataset files in GitHub. Keep raw data outside the repository or in a local `data/raw/` folder excluded by `.gitignore`.

## Datasets

- Primary: [Car Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
- Alternative: [Kaggle Mirror](https://www.kaggle.com/datasets/pranavjha24/car-hacking-dataset)
- Secondary: [Survival Analysis CAN Dataset](https://ocslab.hksecurity.net/Datasets/survival-ids)

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly