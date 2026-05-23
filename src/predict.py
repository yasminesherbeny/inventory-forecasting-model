# src/predict.py
# ============================================================
# Prediction pipeline:
# Load trained models → predict occurrence → route by lag1 → predict quantity
# This is what the Flask API will call.
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

# ── Cached models (loaded once) ─────────────────────────────
_CLASSIFIER = None
_REGRESSORS = None
_ENCODERS   = None
_LOW_MAX    = None
_HIGH_MIN   = None


def load_models(model_dir: str = 'models') -> None:
    """Load all trained models into cache on first call."""
    global _CLASSIFIER, _REGRESSORS, _ENCODERS, _LOW_MAX, _HIGH_MIN

    if _CLASSIFIER is not None:
        return  # Already loaded

    # Calculate absolute path relative to where this predict.py file lives
    current_dir = os.path.dirname(os.path.abspath(__file__))  # points to src/
    root_dir = os.path.dirname(current_dir)                    # points to the project root
    absolute_model_dir = os.path.join(root_dir, model_dir)     # points to project root/models/

    classifier_path = os.path.join(absolute_model_dir, 'occurrence_classifier.pkl')
    regressors_path = os.path.join(absolute_model_dir, 'segmented_regressors.pkl')
    encoders_path   = os.path.join(absolute_model_dir, 'encoders.pkl')

    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier not found at absolute path: {classifier_path}")
    if not os.path.exists(regressors_path):
        raise FileNotFoundError(f"Regressors not found at absolute path: {regressors_path}")
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Encoders not found at absolute path: {encoders_path}")

    _CLASSIFIER = joblib.load(classifier_path)
    _REGRESSORS = joblib.load(regressors_path)
    _ENCODERS   = joblib.load(encoders_path)

    # Read thresholds from a config file or hard-code them
    # For now, hard-code (from training output):
    _LOW_MAX  = 15.0   # median of positive demand
    _HIGH_MIN = 60.0   # 85th percentile of positive demand

    print(f"✅ Models loaded from absolute path: {absolute_model_dir}")
    print(f"   Thresholds: LOW_MAX={_LOW_MAX}, HIGH_MIN={_HIGH_MIN}")



def get_segment_by_lag1(lag1_value: float) -> str:
    """Route to segment based on lag1 (yesterday's demand)."""
    if lag1_value <= _LOW_MAX:
        return 'low'
    elif lag1_value >= _HIGH_MIN:
        return 'high'
    else:
        return 'medium'


def predict_single(
    store_id: str,
    product_id: str,
    features_dict: dict,
) -> dict:
    """
    Predict next-day demand for one store-product combination.

    Args:
        store_id: Store identifier (e.g., "STORE_001")
        product_id: Product identifier (e.g., "P_042")
        features_dict: Dictionary of ALL 34 features (see preprocessing.get_feature_list())
            Must include: lag1, lag7, lag14, lag3, lag30, rolling_mean_7, etc.
            Typically built by preprocessing pipeline on incoming request.

    Returns:
        dict with keys:
            - 'demand_occurs': bool (will there be demand?)
            - 'predicted_units': float (if demand_occurs, else 0)
            - 'segment': str ('low'/'medium'/'high')
            - 'confidence': float (classifier confidence)
    """
    load_models()

    # Convert features to DataFrame row
    X_row = pd.DataFrame([features_dict])

    # Ensure Store ID and Product ID are encoded
    X_row['Store ID'] = _ENCODERS['Store ID'].transform([store_id])
    X_row['Product ID'] = _ENCODERS['Product ID'].transform([product_id])

    # Stage 1: Will there be demand?
    demand_proba = _CLASSIFIER.predict_proba(X_row)[0]
    has_demand = bool(_CLASSIFIER.predict(X_row)[0])
    confidence = float(demand_proba[1] if has_demand else demand_proba[0])

    if not has_demand:
        return {
            'demand_occurs': False,
            'predicted_units': 0.0,
            'segment': 'zero',
            'confidence': confidence,
        }

    # Stage 2: How much?
    segment = get_segment_by_lag1(X_row['lag1'].values[0])

    if segment not in _REGRESSORS:
        # Fallback to medium if segment model is missing
        segment = 'medium'

    regressor = _REGRESSORS[segment]
    pred_log = regressor.predict(X_row)[0]
    pred_units = float(np.expm1(pred_log))
    pred_units = max(0, pred_units)  # Clamp to non-negative

    return {
        'demand_occurs': True,
        'predicted_units': round(pred_units, 2),
        'segment': segment,
        'confidence': confidence,
    }


def predict_batch(
    feature_rows: list,
) -> list:
    """
    Predict for multiple rows at once (more efficient).

    Args:
        feature_rows: List of dicts, each with keys:
            - store_id
            - product_id
            - all 34 features from get_feature_list()

    Returns:
        List of prediction dicts (one per row)
    """
    load_models()

    results = []
    for row in feature_rows:
        store_id = row.pop('store_id')
        product_id = row.pop('product_id')
        result = predict_single(store_id, product_id, row)
        result['store_id'] = store_id
        result['product_id'] = product_id
        results.append(result)

    return results


if __name__ == "__main__":
    # Simple test
    print("Testing predict.py...\n")

    # Load a sample from your test data
    from data_loader import build_dataset
    from preprocessing import preprocess

    dataset = build_dataset()
    X, y, encoders = preprocess(dataset, encoders=None, fit_encoders=False)

    # Take first test row
    sample_row = X.iloc[0].to_dict()
    store_id = 'London'  # from your data
    product_id = X.iloc[0]['Product ID']  # Will be encoded int

    print(f"Predicting for Store={store_id}, Product={product_id}")
    result = predict_single(store_id, str(product_id), sample_row)
    print(f"\nResult: {result}")
