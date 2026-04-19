import time
import numpy as np
from rules import rule_based_detection
from preprocessing import ATTACK_LABELS
from model import predict_multi

FEATURE_COLUMNS = [
    'freq',
    'window_count',
    'can_id_window_freq',
    'time_diff',
    'interarrival_jitter',
    'rolling_mean_time',
    'rolling_var_time',
    'time_diff_ratio',
    'byte_mean',
    'byte_std',
    'byte_max',
    'byte_min',
    'byte_entropy',
    'payload_repeat',
    'rolling_payload_repeat',
    'burst',
    'signature_consistency',
    'rolling_entropy',
    'recent_attack_ratio'
]


def simulate_realtime(df, known_ids, rf_model, xgb_model, iso_model, delay=0.1, batch_size=16):
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        X_batch = batch[FEATURE_COLUMNS].fillna(0).values

        rf_preds = rf_model.predict(X_batch)
        rf_probs = rf_model.predict_proba(X_batch)
        xgb_preds = None
        xgb_probs = None
        if xgb_model is not None:
            xgb_preds = xgb_model.predict(X_batch)
            xgb_probs = xgb_model.predict_proba(X_batch)

        anomaly_scores = iso_model.decision_function(X_batch)
        anomaly_flags = (anomaly_scores < np.percentile(anomaly_scores, 20)).astype(int)
        anomaly_confidences = np.clip((-(anomaly_scores) + 0.2), 0.0, 1.0)

        for idx, row in batch.iterrows():
            i = idx - batch.index[0]
            rule_label, rule_label_name, reason, rule_score = rule_based_detection(row, known_ids)
            rf_label = int(rf_preds[i])
            rf_score = 1 - rf_probs[i][0]
            xgb_label = int(xgb_preds[i]) if xgb_model is not None else None
            xgb_score = 1 - xgb_probs[i][0] if xgb_model is not None else 0.0
            anomaly_conf = float(anomaly_confidences[i])
            anomaly_flag = bool(anomaly_flags[i])

            candidate_label = rule_label if rule_score >= 0.7 else rf_label
            if anomaly_flag and candidate_label == 0:
                candidate_label = 1

            confidence = (rule_score + rf_score + anomaly_conf) / 3.0
            attack = candidate_label != 0
            source = 'Rule' if rule_score > 0 else ('ML' if rf_label != 0 else ('Anomaly' if anomaly_flag else 'None'))
            attack_type = ATTACK_LABELS.get(candidate_label, 'Unknown')
            severity = 'HIGH' if confidence > 0.7 else 'MEDIUM' if confidence > 0.4 else 'LOW'

            reasons = [reason]
            if rf_label != rule_label and rule_label == 0 and rf_label != 0:
                reasons.append(f'ML predicted {ATTACK_LABELS.get(rf_label, "Unknown")}.')
            if anomaly_flag:
                reasons.append('Anomaly detector raised suspicion.')

            packet_info = {
                'timestamp': row['timestamp'],
                'can_id': row['can_id'],
                'dlc': row.get('dlc', None),
                'data': row.get('data', ''),
                'attack': attack,
                'attack_type': attack_type,
                'confidence': float(confidence),
                'source': source,
                'reason': ' '.join(reasons),
                'severity': severity,
                'rule_label': rule_label_name,
                'rf_label': ATTACK_LABELS.get(rf_label, 'Normal'),
                'anomaly_flag': anomaly_flag,
            }

            yield packet_info
            time.sleep(delay)