from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields  # Import Swagger tools
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, 'src')

from predict import predict_single
from preprocessing import get_feature_list

app = Flask(__name__)

# Initialize Swagger UI (This creates the page your friend wants)
api = Api(app, 
    version='1.0', 
    title='Forecast API',
    description='Machine learning prediction endpoints',
    doc='/'  # This places the Swagger UI on the main homepage!
)

# Create an API Namespace (Like "Alert" or "AlertConfigurations" in your friend's image)
ns = api.namespace('api', description='Prediction operations')

# Define the expected JSON input structure for Swagger documentation
prediction_model = api.model('PredictionInput', {
    'store_id': fields.String(required=True, example='STORE_001'),
    'product_id': fields.String(required=True, example='P_042'),
    'date': fields.String(required=True, example='2024-05-21'),
    'price': fields.Float(required=True, example=25.99),
    'discount': fields.Float(required=True, example=0.1),
    'holiday_promotion': fields.Integer(required=True, example=0),
    'inventory_level': fields.Float(required=True, example=150.0),
    'units_ordered': fields.Float(required=True, example=80.0),
    'history': fields.List(fields.Raw, required=True, example=[{"date": "2024-05-20", "units_sold": 15}]),
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

# (Keep your existing compute_features_from_raw function exactly here)
def compute_features_from_raw(data: dict) -> dict:
    # ... keep your exact same 34 feature code here ...
    return features

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
