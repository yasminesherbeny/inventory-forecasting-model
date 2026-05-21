from flask import Flask, request, jsonify
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, 'src')

from predict import predict_single
from preprocessing import get_feature_list

app = Flask(__name__)

# ════════════════════════════════════════════════════════════════
# ENDPOINT 1: Full prediction (raw data → features → prediction)
# ════════════════════════════════════════════════════════════════
@app.route('/predict-from-raw', methods=['POST'])
def predict_from_raw():
    """
    Full end-to-end prediction from raw data.
    
    POST /predict-from-raw with JSON:
    {
        "store_id": "STORE_001",
        "product_id": "P_042",
        "date": "2024-05-21",
        "price": 25.99,
        "discount": 0.1,
        "holiday_promotion": 0,
        "inventory_level": 150,
        "units_ordered": 80,
        "history": [
            {"date": "2024-05-20", "units_sold": 15},
            {"date": "2024-05-19", "units_sold": 18},
            ...last 30 days...
        ],
        "product_stats": {
            "mean": 16.0,
            "max": 80.0,
            "min": 1.0,
            "std": 8.5,
            "median": 15.0
        },
        "store_avg": 16.5
    }
    
    Returns:
    {
        "demand_occurs": true,
        "predicted_units": 18.45,
        "segment": "low",
        "confidence": 0.92
    }
    """
    try:
        data = request.json
        
        # Compute features from raw data
        features = compute_features_from_raw(data)
        
        # Make prediction
        result = predict_single(
            store_id=data['store_id'],
            product_id=data['product_id'],
            features_dict=features
        )
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ════════════════════════════════════════════════════════════════
# ENDPOINT 2: Feature computation only (if they want to debug)
# ════════════════════════════════════════════════════════════════
@app.route('/compute-features', methods=['POST'])
def compute_features_endpoint():
    """
    Just compute features without predicting.
    Same input as /predict-from-raw.
    
    Returns: dict of all 34 features
    """
    try:
        data = request.json
        features = compute_features_from_raw(data)
        return jsonify(features), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION LOGIC
# ════════════════════════════════════════════════════════════════
def compute_features_from_raw(data: dict) -> dict:
    """
    Compute all 34 features from raw database data.
    
    Args:
        data: dict with keys:
            - store_id
            - product_id
            - date
            - price
            - discount
            - holiday_promotion
            - inventory_level
            - units_ordered
            - history: list of {date, units_sold} for last 30+ days
            - product_stats: {mean, max, min, std, median}
            - store_avg
    
    Returns:
        dict of 34 features ready for model
    """
    features = {}
    
    # ── 1. Raw columns (from today's data) ──
    features['Price'] = float(data['price'])
    features['Discount'] = float(data['discount'])
    features['Holiday/Promotion'] = int(data['holiday_promotion'])
    features['Inventory Level'] = float(data['inventory_level'])
    features['Units Ordered'] = float(data['units_ordered'])
    
    # ── 2. Parse history into array ──
    history = data.get('history', [])
    sales = np.array([h['units_sold'] for h in history])
    
    # Sort by date to ensure order
    history_df = pd.DataFrame(history)
    if 'date' in history_df.columns:
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df = history_df.sort_values('date')
        sales = history_df['units_sold'].values
    
    # ── 3. Lag features ──
    features['lag1'] = float(sales[-1]) if len(sales) >= 1 else 0.0
    features['lag3'] = float(sales[-3]) if len(sales) >= 3 else 0.0
    features['lag7'] = float(sales[-7]) if len(sales) >= 7 else 0.0
    features['lag14'] = float(sales[-14]) if len(sales) >= 14 else 0.0
    features['lag30'] = float(sales[-30]) if len(sales) >= 30 else 0.0
    
    # ── 4. Rolling features ──
    features['rolling_mean_7'] = float(np.mean(sales[-7:])) if len(sales) >= 7 else float(np.mean(sales))
    features['rolling_mean_14'] = float(np.mean(sales[-14:])) if len(sales) >= 14 else float(np.mean(sales))
    features['rolling_sum_7'] = float(np.sum(sales[-7:])) if len(sales) >= 7 else float(np.sum(sales))
    features['rolling_std_7'] = float(np.std(sales[-7:])) if len(sales) >= 7 else 0.0
    
    # ── 5. Derived features ──
    features['trend_7'] = features['rolling_mean_7'] - features['rolling_mean_14']
    features['demand_ratio'] = features['lag1'] / (features['rolling_mean_7'] + 1)
    features['inventory_pressure'] = features['Inventory Level'] / (features['rolling_mean_7'] + 1)
    features['price_discount'] = features['Price'] * features['Discount']
    
    # ── 6. Product-level stats ──
    product_stats = data.get('product_stats', {})
    product_mean = product_stats.get('mean', 0)
    
    features['product_mean'] = float(product_stats.get('mean', 0))
    features['product_max'] = float(product_stats.get('max', 0))
    features['product_min'] = float(product_stats.get('min', 0))
    features['product_std'] = float(product_stats.get('std', 0))
    features['product_median'] = float(product_stats.get('median', 0))
    features['product_range'] = features['product_max'] - features['product_min']
    features['product_volatility'] = features['product_std']
    features['interaction_1'] = features['lag1'] * features['product_mean']
    
    # ── 7. Store-level ──
    features['store_avg'] = float(data.get('store_avg', 0))
    
    # ── 8. Temporal features ──
    from datetime import datetime
    date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
    features['day_of_week'] = date_obj.weekday()
    features['month'] = date_obj.month
    features['week_of_year'] = date_obj.isocalendar()[1]
    features['is_weekend'] = 1 if date_obj.weekday() >= 5 else 0
    
    # ── 9. Demand segment (based on product_mean) ──
    if product_mean < 16:
        features['demand_segment'] = 0
    elif product_mean < 25:
        features['demand_segment'] = 1
    else:
        features['demand_segment'] = 2
    
    # ── 10. Interaction ──
    features['interaction_2'] = features['price_discount']
    
    return features


# ════════════════════════════════════════════════════════════════
# UTILITY ENDPOINTS
# ════════════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/features-list', methods=['GET'])
def features_list():
    """Return the 34 required features for reference."""
    return jsonify({
        'features': get_feature_list(),
        'count': len(get_feature_list())
    }), 200


if __name__ == '__main__':
    app.run(debug=False, port=5000)