import numpy as np


def get_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        return []

    feature_importance = sorted(
        zip(feature_names, importances), key=lambda item: item[1], reverse=True
    )
    return feature_importance[:top_n]


def explain_prediction(model, feature_vector, feature_names, top_n=5):
    feature_vector = np.array(feature_vector).reshape(1, -1)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(feature_vector)[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
    else:
        predicted_class = int(model.predict(feature_vector)[0])
        confidence = 1.0

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_names))

    contributions = []
    for name, value, importance in zip(feature_names, feature_vector.flatten(), importances):
        contributions.append((name, float(value), float(importance), float(value * importance)))

    contributions.sort(key=lambda x: x[3], reverse=True)
    top_features = contributions[:top_n]
    reasons = [f'{name}={value:.2f} (importance {importance:.4f})' for name, value, importance, _ in top_features]

    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top_features': top_features,
        'reason': '; '.join(reasons)
    }
