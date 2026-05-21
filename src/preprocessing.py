# src/preprocessing.py
# ============================================================
# Full feature engineering pipeline:
# - Sorting, lag features, rolling features, derived features
# - Encoding, log transformation, train/test split
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


RANDOM_SEED = 42


def sort_by_time(dataset: pd.DataFrame) -> pd.DataFrame:
    """Sort by Store ID, Product ID, Date to ensure proper lag alignment."""
    return dataset.sort_values(by=['Store ID', 'Product ID', 'Date']).reset_index(drop=True)


def create_target(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create target: Units Sold from T+1 (shifted by -1)."""
    dataset['Target'] = (
        dataset.groupby(['Store ID', 'Product ID'])['Units Sold']
        .shift(-1)
    )
    return dataset


def validate_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows: Units Sold < 0, Price <= 0."""
    dataset = dataset[dataset['Units Sold'] >= 0]
    dataset = dataset[dataset['Price'] > 0]
    return dataset


def create_lag_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create lag1, lag3, lag7, lag14, lag30 with outlier clipping."""
    for lag in [1, 3, 7, 14, 30]:
        col_name = f'lag{lag}'
        dataset[col_name] = (
            dataset.groupby(['Store ID', 'Product ID'])['Units Sold']
            .shift(lag)
        )
        # Clip to 99th percentile to handle outliers
        dataset[col_name] = dataset[col_name].clip(0, dataset[col_name].quantile(0.99))

    return dataset


def create_rolling_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create rolling stats (mean, sum, std) with proper shift to avoid leakage."""
    dataset['rolling_mean_7'] = (
        dataset.groupby(['Store ID', 'Product ID'])['Units Sold']
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )
    dataset['rolling_mean_14'] = (
        dataset.groupby(['Store ID', 'Product ID'])['Units Sold']
        .transform(lambda x: x.shift(1).rolling(14).mean())
    )
    dataset['rolling_sum_7'] = (
        dataset.groupby(['Store ID', 'Product ID'])['Units Sold']
        .transform(lambda x: x.shift(1).rolling(7).sum())
    )
    dataset['rolling_std_7'] = (
        dataset.groupby(['Store ID', 'Product ID'])['Units Sold']
        .transform(lambda x: x.shift(1).rolling(7).std())
    )
    return dataset


def create_derived_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create trend, demand ratio, inventory pressure, interactions."""
    dataset['trend_7'] = dataset['rolling_mean_7'] - dataset['rolling_mean_14']
    dataset['demand_ratio'] = dataset['lag1'] / (dataset['rolling_mean_7'] + 1)
    dataset['inventory_pressure'] = dataset['Inventory Level'] / (dataset['rolling_mean_7'] + 1)
    dataset['price_discount'] = dataset['Price'] * dataset['Discount']
    dataset['interaction_1'] = dataset['lag1'] * dataset.groupby('Product ID')['Units Sold'].transform('mean')
    dataset['interaction_2'] = dataset['Price'] * dataset['Discount']
    return dataset


def create_product_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create product-level statistics: mean, max, min, std, median, range, volatility."""
    dataset['product_mean'] = dataset.groupby('Product ID')['Units Sold'].transform('mean')
    dataset['product_max'] = dataset.groupby('Product ID')['Units Sold'].transform('max')
    dataset['product_min'] = dataset.groupby('Product ID')['Units Sold'].transform('min')
    dataset['product_std'] = dataset.groupby('Product ID')['Units Sold'].transform('std')
    dataset['product_median'] = dataset.groupby('Product ID')['Units Sold'].transform('median')
    dataset['product_range'] = dataset['product_max'] - dataset['product_min']
    dataset['product_volatility'] = dataset.groupby('Product ID')['Units Sold'].transform('std')
    dataset['store_avg'] = dataset.groupby('Store ID')['Units Sold'].transform('mean')
    return dataset


def create_time_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create day of week, month, week of year, weekend flag."""
    dataset['day_of_week'] = dataset['Date'].dt.dayofweek
    dataset['month'] = dataset['Date'].dt.month
    dataset['week_of_year'] = dataset['Date'].dt.isocalendar().week.astype(int)
    dataset['is_weekend'] = (dataset['day_of_week'] >= 5).astype(int)
    return dataset


def create_demand_segment(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create demand segment via quantiles, convert to int."""
    dataset['demand_segment'] = pd.qcut(
        dataset['product_mean'],
        q=3,
        labels=[0, 1, 2],
        duplicates='drop'
    ).astype(int)
    return dataset


def clean_target(dataset: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where Target is NaN (T+1 is unavailable)."""
    return dataset.dropna(subset=['Target'])


def log_transform_target(dataset: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to target for numerical stability."""
    dataset['Target'] = np.log1p(dataset['Target'])
    return dataset


def final_dropna(dataset: pd.DataFrame) -> pd.DataFrame:
    """Drop any remaining NaN rows."""
    return dataset.dropna()


def drop_raw_units(dataset: pd.DataFrame) -> pd.DataFrame:
    """Drop original Units Sold column after feature creation."""
    return dataset.drop(columns=['Units Sold'])


def encode_categorical(dataset: pd.DataFrame, encoders: dict = None) -> tuple:
    """
    Label-encode Store ID and Product ID.
    If encoders provided, use existing ones (inference mode).
    Otherwise fit new ones (training mode).
    """
    if encoders is None:
        encoders = {}
        for col in ['Store ID', 'Product ID']:
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col])
            encoders[col] = le
    else:
        for col in ['Store ID', 'Product ID']:
            dataset[col] = encoders[col].transform(dataset[col])

    return dataset, encoders


def get_feature_list() -> list:
    """Return the final feature list used by the model."""
    return [
        'Store ID', 'Product ID',
        'Inventory Level', 'Units Ordered', 'Price', 'Discount', 'Holiday/Promotion',
        'lag1', 'lag7', 'lag14', 'lag3', 'lag30',
        'rolling_mean_7', 'rolling_mean_14', 'rolling_sum_7', 'rolling_std_7',
        'trend_7', 'demand_ratio', 'inventory_pressure',
        'day_of_week', 'month', 'week_of_year', 'is_weekend',
        'product_mean', 'product_max', 'product_min', 'product_std',
        'product_median', 'product_range',
        'store_avg', 'product_volatility',
        'price_discount', 'demand_segment', 'interaction_1', 'interaction_2',
    ]


def preprocess(
    dataset: pd.DataFrame,
    encoders: dict = None,
    fit_encoders: bool = True
) -> tuple:
    """
    Full preprocessing pipeline.

    Args:
        dataset: Raw dataframe from data_loader
        encoders: Pre-fitted encoders (for inference); None means fit new ones
        fit_encoders: If True, save encoders to disk after fitting

    Returns:
        X: Feature dataframe (ready for model)
        y: Target series (log-transformed)
        encoders: Dictionary of fitted encoders
    """
    # Step 1: Sort and validate
    dataset = sort_by_time(dataset)
    dataset = validate_rows(dataset)

    # Step 2: Create target (before modifying Units Sold)
    dataset = create_target(dataset)

    # Step 3: Feature engineering
    dataset = create_lag_features(dataset)
    dataset = create_rolling_features(dataset)
    dataset = create_derived_features(dataset)
    dataset = create_product_features(dataset)
    dataset = create_time_features(dataset)
    dataset = create_demand_segment(dataset)

    # Step 4: Clean and transform target
    dataset = clean_target(dataset)
    dataset = log_transform_target(dataset)
    dataset = final_dropna(dataset)

    # Step 5: Drop raw column after all features created
    dataset = drop_raw_units(dataset)

    # Step 6: Encode categorical
    dataset, encoders = encode_categorical(dataset, encoders=encoders if not fit_encoders else None)

    # Step 7: Extract X and y
    features = get_feature_list()
    X = dataset[features].copy()
    y = dataset['Target'].copy()

    # Fix demand_segment dtype (sometimes pandas makes it category)
    X['demand_segment'] = X['demand_segment'].astype(int)

    # Step 8: Save encoders if fitting new ones
    if fit_encoders and encoders:
        os.makedirs('models', exist_ok=True)
        joblib.dump(encoders, 'models/encoders.pkl')
        print("Encoders saved → models/encoders.pkl")

    return X, y, encoders


if __name__ == "__main__":
    from data_loader import build_dataset

    dataset = build_dataset()
    X, y, encoders = preprocess(dataset, fit_encoders=True)
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nFeatures ({len(X.columns)}):")
    print(X.columns.tolist())