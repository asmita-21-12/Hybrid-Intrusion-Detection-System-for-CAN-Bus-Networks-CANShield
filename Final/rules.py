from preprocessing import ATTACK_LABELS


def rule_based_detection(row, known_ids, dynamic_threshold_factor=1.8, entropy_threshold=3.5):
    attack_label = 0
    reason = 'No rule matched.'
    rule_score = 0.0

    dynamic_freq = max(10, row.get('rolling_mean_time', 0) * dynamic_threshold_factor + 1)
    if row.get('can_id_window_freq', 0) > dynamic_freq and row.get('burst', 0) > 0:
        attack_label = 1
        reason = f'DoS-like burst: window freq {row.get("can_id_window_freq", 0):.0f}'
        rule_score = 1.0
    elif row['can_id'] not in known_ids:
        attack_label = 1
        reason = f'Unknown CAN ID: {row["can_id"]}'
        rule_score = 0.9
    elif row.get('byte_entropy', 0) > entropy_threshold:
        attack_label = 2
        reason = f'High payload entropy: {row.get("byte_entropy", 0):.2f}'
        rule_score = 0.7
    elif row.get('rolling_payload_repeat', 0) > 5 and row.get('signature_consistency', 0) == 1:
        attack_label = 1
        reason = 'Repeated payload pattern indicates possible replay/DoS.'
        rule_score = 0.8
    elif row.get('payload_repeat', 0) == 1 and row.get('recent_attack_ratio', 0) > 0.2:
        attack_label = 1
        reason = 'Payload repeated after attack bursts.'
        rule_score = 0.6
    elif row.get('time_diff_ratio', 0) > 5.0 and row.get('window_count', 0) > 1:
        attack_label = 1
        reason = 'Sudden inter-arrival spike detected.'
        rule_score = 0.5
    else:
        attack_label = 0
        reason = 'No rule-based attack detected.'
        rule_score = 0.0

    return attack_label, ATTACK_LABELS.get(attack_label, 'Unknown'), reason, rule_score