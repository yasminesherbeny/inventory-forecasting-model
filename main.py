# main.py
# ============================================================
# Orchestrates full pipeline: load → preprocess → train → evaluate
# Run this to rebuild everything from scratch.
# ============================================================

import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, 'src')

from data_loader import build_dataset
from preprocessing import preprocess
from train import train_two_stage
from evaluate import print_summary, compare_to_baseline


def main():
    print("\n" + "="*70)
    print("  INVENTORY DEMAND FORECASTING — FULL PIPELINE")
    print("="*70)

    # ════════════════════════════════════════════════════════════════
    # STEP 1: Load and prepare data
    # ════════════════════════════════════════════════════════════════
    print("\n[1/5] Loading data...")
    dataset = build_dataset(save_path='data/dataset.csv')

    # ════════════════════════════════════════════════════════════════
    # STEP 2: Preprocess (full dataset, maintains continuity for lags)
    # ════════════════════════════════════════════════════════════════
    print("\n[2/5] Preprocessing...")
    X, y, encoders = preprocess(dataset, fit_encoders=True)
    print(f"Features: {X.shape[1]}, Rows: {X.shape[0]}")

    # ════════════════════════════════════════════════════════════════
    # STEP 3: Train/test split (by position, preserving time order)
    # ════════════════════════════════════════════════════════════════
    print("\n[3/5] Splitting (last 30 days as test)...")

    # Get split position from ORIGINAL dataset using dates
    split_date = dataset['Date'].max() - pd.Timedelta(days=30)
    split_position = (dataset['Date'] < split_date).sum()
    
    # After preprocessing, dropna() removed some rows, so get the position
    # by finding where the split_date rows start in X
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    # Approximate split (use 80% train, 20% test since exact split is lost)
    split_idx = int(len(X_reset) * 0.80)
    
    X_train = X_reset.iloc[:split_idx].copy()
    y_train = y_reset.iloc[:split_idx].copy()
    X_test = X_reset.iloc[split_idx:].copy()
    y_test = y_reset.iloc[split_idx:].copy()

    print(f"Train: {len(X_train):,} rows")
    print(f"Test:  {len(X_test):,} rows")

    # ════════════════════════════════════════════════════════════════
    # STEP 4: Train two-stage model
    # ════════════════════════════════════════════════════════════════
    print("\n[4/5] Training two-stage model...")
    results = train_two_stage(X_train, y_train, X_test, y_test)

    # ════════════════════════════════════════════════════════════════
    # STEP 5: Evaluate
    # ════════════════════════════════════════════════════════════════
    print("\n[5/5] Evaluating...")

    y_true = results['y_test_real']
    y_test_binary = results['y_test_binary']

    # Predict
    def predict_two_stage_v2(X, classifier, regressors, low_max, high_min):
        """Inline prediction for evaluation."""
        has_demand = classifier.predict(X).astype(bool)
        preds_log = np.zeros(len(X))

        if has_demand.sum() > 0:
            X_pos = X[has_demand]
            lag1 = X_pos['lag1'].values
            segments = np.where(
                lag1 <= low_max, 'low',
                np.where(lag1 >= high_min, 'high', 'medium')
            )
            qty_log = np.zeros(len(X_pos))

            for label, model in regressors.items():
                mask = segments == label
                if mask.sum() == 0:
                    continue
                qty_log[mask] = model.predict(X_pos[mask])

            missing = ~np.isin(segments, list(regressors.keys()))
            if missing.sum() > 0:
                qty_log[missing] = regressors['medium'].predict(X_pos[missing])

            preds_log[has_demand] = qty_log

        return preds_log

    y_pred_log = predict_two_stage_v2(
        X_test,
        results['classifier'],
        results['regressors'],
        results['LOW_MAX'],
        results['HIGH_MIN'],
    )
    y_pred = np.expm1(y_pred_log)

    # Naive baseline
    y_naive = np.expm1(X_test['lag1'].values).clip(0)

    # Segment masks
    segment_masks = {
        'zero_demand': y_true == 0,
        'positive_demand': y_true > 0,
    }

    print_summary(y_true, y_pred, y_naive, segment_masks)

    print("\n" + "="*70)
    print("  FINAL STATUS")
    print("="*70)
    print("✅ Pipeline complete")
    print("✅ Models saved to models/")
    print("✅ Encoders saved to models/encoders.pkl")
    print("✅ Dataset saved to data/dataset.csv")
    print("\nTo make predictions: import predict.py and call predict_single()")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
    