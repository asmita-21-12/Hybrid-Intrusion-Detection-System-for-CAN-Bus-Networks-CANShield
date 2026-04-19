import os
import pandas as pd
import numpy as np

LABEL_MAP = {
    'normal': 0,
    'dos': 1,
    'fuzzy': 2,
    'rpm': 3,
    'gear': 4,
    'rpm spoofing': 3,
    'gear spoofing': 4,
}

ATTACK_LABELS = {
    0: 'Normal',
    1: 'DoS',
    2: 'Fuzzy',
    3: 'RPM',
    4: 'Gear'
}


def normalize_columns(df):
    df = df.rename(columns={
        col: col.strip().lower().replace(' ', '_')
        for col in df.columns
    })
    return df


def infer_label_from_path(path):
    lower = path.lower()
    for key, value in LABEL_MAP.items():
        if key in lower:
            return value
    return 1


def load_kaggle_dataset(data_dir='data'):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'Data folder not found: {data_dir}')

    frames = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.lower().endswith('.csv'):
                continue
            path = os.path.join(root, file)
            df = pd.read_csv(path)
            df = normalize_columns(df)
            if 'label' not in df.columns:
                label = infer_label_from_path(path)
                df['label'] = label
            else:
                df['label'] = df['label'].astype(str).str.strip().str.lower().map(LABEL_MAP).fillna(1).astype(int)
            frames.append(df)

    if not frames:
        raise FileNotFoundError('No CSV files found in data directory.')

    merged = pd.concat(frames, ignore_index=True)
    return merged


def load_and_preprocess_data(data_dir='data', sample_path='sample_data.csv'):
    if os.path.isdir(data_dir):
        df = load_kaggle_dataset(data_dir)
    else:
        df = pd.read_csv(sample_path)
        df = normalize_columns(df)
        if 'label' in df.columns:
            df['label'] = df['label'].astype(str).str.strip().str.lower().map(LABEL_MAP).fillna(1).astype(int)
        elif 'label' in df.columns:
            df['label'] = df['label']
        else:
            df['label'] = 0

    df.dropna(inplace=True)

    if 'can_id' in df.columns:
        df['can_id'] = df['can_id'].apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else int(x) if pd.notna(x) else x)
    else:
        raise ValueError('CAN_ID column is required.')

    if 'timestamp' not in df.columns:
        raise ValueError('Timestamp column is required.')

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    df['interarrival_jitter'] = df['time_diff'].diff().abs().fillna(0)
    df['attack_name'] = df['label'].map(ATTACK_LABELS)
    df['is_attack'] = (df['label'] != 0).astype(int)
    df['payload'] = df['data'].astype(str).str.strip()
    df['payload_repeat'] = (df['payload'] == df['payload'].shift()).astype(int)
    return df