import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split


def train_models(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    rf_model = RandomForestClassifier(n_estimators=150, random_state=random_state)
    rf_model.fit(X_train, y_train)

    xgb_model = None
    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)
        xgb_model.fit(X_train, y_train)
    except Exception:
        xgb_model = None

    iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=random_state)
    iso_model.fit(X_train[y_train == 0])

    return {
        'rf': rf_model,
        'xgb': xgb_model,
        'iso': iso_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def predict_multi(rf_model, xgb_model, iso_model, features):
    X = np.array(features).reshape(1, -1)
    rf_pred = int(rf_model.predict(X)[0])
    rf_probs = rf_model.predict_proba(X)[0]
    rf_score = 1 - rf_probs[0]

    xgb_pred = None
    xgb_score = 0.0
    xgb_probs = None
    if xgb_model is not None:
        xgb_pred = int(xgb_model.predict(X)[0])
        xgb_probs = xgb_model.predict_proba(X)[0]
        xgb_score = 1 - xgb_probs[0]

    anomaly_score = float(-iso_model.decision_function(X)[0])
    anomaly_confidence = min(max((anomaly_score + 0.5), 0.0), 1.0)

    return {
        'rf_pred': rf_pred,
        'rf_score': rf_score,
        'xgb_pred': xgb_pred,
        'xgb_score': xgb_score,
        'anomaly_confidence': anomaly_confidence,
        'rf_probs': rf_probs,
        'xgb_probs': xgb_probs
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return y_pred