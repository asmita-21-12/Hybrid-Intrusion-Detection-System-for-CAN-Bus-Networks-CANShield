import pandas as pd
import numpy as np
from collections import Counter
import math


def parse_payload_bytes(payload):
    if not isinstance(payload, str) or len(payload.strip()) == 0:
        return []
    items = [x for x in payload.strip().split() if x]
    values = []
    for token in items:
        try:
            values.append(int(token, 16))
        except ValueError:
            try:
                values.append(int(token))
            except ValueError:
                continue
    return values


def calculate_entropy(data_bytes):
    if not data_bytes:
        return 0.0
    c = Counter(data_bytes)
    ent = 0.0
    length = len(data_bytes)
    for count in c.values():
        p = count / length
        ent -= p * math.log2(p) if p > 0 else 0
    return ent


def create_features(df, window=20):
    df = df.copy()

    df['freq'] = df.groupby('can_id')['can_id'].transform('count')
    df['window_count'] = df['can_id'].rolling(window=window, min_periods=1).count().fillna(0)
    df['can_id_window_freq'] = df.groupby('can_id')['can_id'].transform(lambda x: x.rolling(window=window, min_periods=1).count().fillna(0))
    df['rolling_mean_time'] = df['time_diff'].rolling(window=window, min_periods=1).mean().fillna(0)
    df['rolling_var_time'] = df['time_diff'].rolling(window=window, min_periods=1).var().fillna(0)
    df['interarrival_jitter'] = df['time_diff'].diff().abs().fillna(0)
    df['time_diff_ratio'] = df['time_diff'] / (df['rolling_mean_time'] + 1e-6)
    df['payload_bytes'] = df['payload'].apply(parse_payload_bytes)
    df['byte_mean'] = df['payload_bytes'].apply(lambda b: np.mean(b) if b else 0.0)
    df['byte_std'] = df['payload_bytes'].apply(lambda b: np.std(b) if b else 0.0)
    df['byte_max'] = df['payload_bytes'].apply(lambda b: max(b) if b else 0.0)
    df['byte_min'] = df['payload_bytes'].apply(lambda b: min(b) if b else 0.0)
    df['byte_entropy'] = df['payload_bytes'].apply(calculate_entropy)
    df['payload_repeat'] = (df['payload'] == df['payload'].shift()).astype(int)
    df['rolling_payload_repeat'] = df['payload_repeat'].rolling(window=window, min_periods=1).sum().fillna(0)
    df['burst'] = ((df['time_diff'] < (df['rolling_mean_time'] * 0.5).replace({0: 1e-6})) & (df['window_count'] > 5)).astype(int)
    df['payload_hash'] = df['payload'].apply(lambda x: hash(x) if isinstance(x, str) else 0)
    df['signature_consistency'] = df['payload_hash'].rolling(window=window, min_periods=1).apply(
        lambda x: 1 if len(set(x)) == 1 else 0,
        raw=False
    ).astype(int)
    df['rolling_entropy'] = df['byte_entropy'].rolling(window=window, min_periods=1).mean().fillna(0)
    df['recent_attack_ratio'] = df['is_attack'].rolling(window=window, min_periods=1).mean().fillna(0)

    for col in ['freq', 'window_count', 'can_id_window_freq', 'rolling_mean_time', 'rolling_var_time', 'interarrival_jitter', 'time_diff_ratio', 'byte_mean', 'byte_std', 'byte_max', 'byte_min', 'byte_entropy', 'payload_repeat', 'rolling_payload_repeat', 'burst', 'signature_consistency', 'rolling_entropy', 'recent_attack_ratio']:
        if col not in df.columns:
            df[col] = 0

    return df