from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields  # Import Swagger tools
import sys
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.insert(0, 'src')

from predict import predict_single
from preprocessing import get_feature_list

app = Flask(__name__)

# Initialize Swagger UI
api = Api(app, 
    version='1.0', 
    title='Forecast API',
    description='Machine learning prediction endpoints',
    doc='/'  # Places Swagger UI on the homepage
)

# Create API Namespace
ns = api.namespace('api', description='Prediction operations')

# Define expected JSON structure for Swagger UI documentation
prediction_model = api.model('PredictionInput', {
    'store_id': fields.String(required=True, example='STORE_001'),
    'product_id': fields.String(required=True, example='P_042'),
    'date': fields.String(required=True, example='2024-05-21'),
    'price': fields.Float(required=True, example=25.99),
    'discount': fields.Float(required=True, example=0.1),
    'holiday_promotion': fields.Integer(required=True, example=0),
    'inventory_level': fields.Float(required=True, example=150.0),
    'units_ordered': fields.Float(required=True, example=80.0),
    'history': fields.List(fields.Raw, required=True, example=[{"date": "2024-05-20", "units_sold": 15}, {"date": "2024-05-19", "units_sold": 18}]),
    'product_stats': fields.Raw(required=True, example={"mean": 16.0, "max": 80.0, "min": 1.0, "std": 8.5, "median": 15.0}),
    'store_avg': fields.Float(required=True, example=16.5)
})

# ════════════════════════════════════════════════════════════════
# ENDPOINT 1: Full prediction (POST /api/predict-from-raw)
# ════════════════════════════════════════════════════════════════
@ns.route('/predict-from-raw')
class PredictFromRaw(Resource):
    @ns.expect(prediction_model)
    def post(self):
        """Full end-to-end prediction from raw data."""
        try:
            data = request.json
            features = compute_features_from_raw(data)
            result = predict_single(
                store_id=data['store_id'],
                product_id=data['product_id'],
                features_dict=features
            )
            return result, 200
        except Exception as e:
            return {'error': str(e)}, 400

# ════════════════════════════════════════════════════════════════
# ENDPOINT 2: Feature computation only (POST /api/compute-features)
# ════════════════════════════════════════════════════════════════
@ns.route('/compute-features')
class ComputeFeatures(Resource):
    @ns.expect(prediction_model)
    def post(self):
        """Just compute features without predicting."""
        try:
            data = request.json
            features = compute_features_from_raw(data)
            return features, 200
        except Exception as e:
            return {'error': str(e)}, 400

# ════════════════════════════════════════════════════════════════
# UTILITY ENDPOINTS
# ════════════════════════════════════════════════════════════════
@ns.route('/health')
class Health(Resource):
    def get(self):
        """Check api status."""
        return {'status': 'ok'}, 200

@ns.route('/features-list')
class FeaturesList(Resource):
    def get(self):
        """Return the 34 required features for reference."""
        return {
            'features': get_feature_list(),
            'count': len(get_feature_list())
        }, 200

# ════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION LOGIC (RESTORED)
# ════════════════════════════════════════════════════════════════
def compute_features_from_raw(data: dict) -> dict:
    features = {}
    
    # ── 1. Raw columns ──
    features['Price'] = float(data['price'])
    features['Discount'] = float(data['discount'])
    features['Holiday/Promotion'] = int(data['holiday_promotion'])
    features['Inventory Level'] = float(data['inventory_level'])
    features['Units Ordered'] = float(data['units_ordered'])
    
    # ── 2. Parse history ──
    history = data.get('history', [])
    sales = np.array([h['units_sold'] for h in history])
    
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
    date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
    features['day_of_week'] = date_obj.weekday()
    features['month'] = date_obj.month
    features['week_of_year'] = date_obj.isocalendar()[1]
    features['is_weekend'] = 1 if date_obj.weekday() >= 5 else 0
    
    # ── 9. Demand segment ──
    if product_mean < 16:
        features['demand_segment'] = 0
    elif product_mean < 25:
        features['demand_segment'] = 1
    else:
        features['demand_segment'] = 2
    
    # ── 10. Interaction ──
    features['interaction_2'] = features['price_discount']
    
    return features

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
