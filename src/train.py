# src/train.py
# ============================================================
# Two-stage model training:
# Stage 1: Occurrence classifier (demand or no demand)
# Stage 2: Segmented quantity regressors (low/medium/high)
# ============================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')

RANDOM_SEED = 42


def train_two_stage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_dir: str = 'models',
) -> dict:
    """
    Train two-stage model:
      Stage 1: Classify if demand > 0
      Stage 2: Regress quantity for positive demand rows (3 segments by lag1)

    Returns:
        results: Dictionary with classifier, regressors, thresholds, metrics
    """
    os.makedirs(model_dir, exist_ok=True)

    # Convert targets to real scale
    y_train_real = np.expm1(y_train.values)
    y_test_real  = np.expm1(y_test.values)

    y_train_binary = (y_train_real > 0).astype(int)
    y_test_binary  = (y_test_real > 0).astype(int)

    print("\n" + "="*60)
    print("  STAGE 1: OCCURRENCE CLASSIFIER")
    print("="*60)
    print(f"Train — Zero demand: {(y_train_binary==0).sum():,}  "
          f"Positive: {(y_train_binary==1).sum():,}")
    print(f"Test  — Zero demand: {(y_test_binary==0).sum():,}   "
          f"Positive: {(y_test_binary==1).sum():,}")

    # Train occurrence classifier
    classifier = lgb.LGBMClassifier(
        boosting_type='gbdt',
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        is_unbalance=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    classifier.fit(X_train, y_train_binary)
    joblib.dump(classifier, os.path.join(model_dir, 'occurrence_classifier.pkl'))
    print("\n✅ Classifier trained and saved")

    # Evaluate classifier
    y_pred_binary = classifier.predict(X_test)
    print("\nClassifier Performance:")
    print(classification_report(
        y_test_binary, y_pred_binary,
        target_names=['No demand', 'Has demand']
    ))

    # ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STAGE 2: QUANTITY REGRESSORS")
    print("="*60)

    # Filter to positive demand rows
    pos_mask_tr = y_train_real > 0
    pos_mask_te = y_test_real > 0

    print(f"Train rows (positive demand): {pos_mask_tr.sum():,}")
    print(f"Test rows  (positive demand): {pos_mask_te.sum():,}")

    # Thresholds from POSITIVE demand distribution
    y_pos = pd.Series(y_train_real[pos_mask_tr])
    LOW_MAX  = float(y_pos.quantile(0.50))   # median of positive demand
    HIGH_MIN = float(y_pos.quantile(0.85))   # 85th percentile

    print(f"\nPositive demand thresholds:")
    print(f"  LOW_MAX  (50th pct): {LOW_MAX:.1f}")
    print(f"  HIGH_MIN (85th pct): {HIGH_MIN:.1f}")

    # Helper: route by lag1 value
    def get_segment_by_lag1(X, low_max, high_min):
        lag1 = X['lag1'].values
        segments = np.where(
            lag1 <= low_max, 'low',
            np.where(lag1 >= high_min, 'high', 'medium')
        )
        return segments

    # Build training masks
    train_segments_all = get_segment_by_lag1(X_train, LOW_MAX, HIGH_MIN)
    low_mask_tr = (
        pos_mask_tr &
        (pd.Series(train_segments_all, index=X_train.index) == 'low')
    )
    med_mask_tr = (
        pos_mask_tr &
        (pd.Series(train_segments_all, index=X_train.index) == 'medium')
    )
    high_mask_tr = (
        pos_mask_tr &
        (pd.Series(train_segments_all, index=X_train.index) == 'high')
    )

    print(f"\nTrain segment sizes (positive demand only):")
    print(f"  Low  (lag1≤{LOW_MAX:.0f}):           {low_mask_tr.sum():,}")
    print(f"  Med  ({LOW_MAX:.0f}<lag1<{HIGH_MIN:.0f}):  {med_mask_tr.sum():,}")
    print(f"  High (lag1≥{HIGH_MIN:.0f}):          {high_mask_tr.sum():,}")

    # Regressor hyperparameters
    SEGMENT_PARAMS = {
        "low": dict(
            boosting_type='gbdt', objective='regression', metric='mae',
            n_estimators=800, learning_rate=0.02, num_leaves=31, max_depth=6,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.3, reg_lambda=2.0, random_state=RANDOM_SEED,
            n_jobs=-1, verbose=-1,
        ),
        "medium": dict(
            boosting_type='gbdt', objective='regression', metric='mae',
            n_estimators=800, learning_rate=0.02, num_leaves=63, max_depth=-1,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED,
            n_jobs=-1, verbose=-1,
        ),
        "high": dict(
            boosting_type='gbdt', objective='regression', metric='mae',
            n_estimators=600, learning_rate=0.02, num_leaves=63, max_depth=5,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.2, reg_lambda=1.5, random_state=RANDOM_SEED,
            n_jobs=-1, verbose=-1,
        ),
    }

    # Train regressors
    regressors = {}
    for label, mask in [("low", low_mask_tr), ("medium", med_mask_tr), ("high", high_mask_tr)]:
        if mask.sum() < 20:
            print(f"⚠️  Segment '{label}' has only {mask.sum()} rows — skipping")
            continue

        model = lgb.LGBMRegressor(**SEGMENT_PARAMS[label])
        model.fit(X_train[mask], y_train[mask])
        regressors[label] = model
        print(f"✅ '{label}' regressor trained on {mask.sum():,} rows")

    joblib.dump(regressors, os.path.join(model_dir, 'segmented_regressors.pkl'))
    print("\n✅ Regressors saved")

    # Return config for predict.py
    results = {
        'classifier'  : classifier,
        'regressors'  : regressors,
        'LOW_MAX'     : LOW_MAX,
        'HIGH_MIN'    : HIGH_MIN,
        'y_test_real' : y_test_real,
        'y_test_binary': y_test_binary,
    }

    return results


if __name__ == "__main__":
    from data_loader import build_dataset
    from preprocessing import preprocess

    print("Loading dataset...")
    dataset = build_dataset()

    print("\nPreprocessing...")
    X, y, encoders = preprocess(dataset, fit_encoders=True)

    print("\nSplitting...")
    split_date = dataset['Date'].max() - pd.Timedelta(days=30)
    train_mask = dataset['Date'] < split_date
    test_mask  = dataset['Date'] >= split_date

    # Need to filter X, y by mask (they're derived from dataset)
    dataset_with_target = dataset.copy()
    X_full, y_full, _ = preprocess(dataset_with_target, encoders=encoders, fit_encoders=False)

    X_train = X_full[train_mask]
    y_train = y_full[train_mask]
    X_test = X_full[test_mask]
    y_test = y_full[test_mask]

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    print("\nTraining...")
    results = train_two_stage(X_train, y_train, X_test, y_test)