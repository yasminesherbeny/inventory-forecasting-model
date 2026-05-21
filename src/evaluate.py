# src/evaluate.py
# ============================================================
# Evaluation metrics and performance reporting
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_full_pipeline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_real: np.ndarray = None,
) -> dict:
    """
    Compute comprehensive metrics and comparison to naive baseline.

    Args:
        y_true: Actual values (real units, not log)
        y_pred: Predicted values (real units, not log)
        y_train_real: Optional; training set actuals for baseline context

    Returns:
        metrics dict with MAE, RMSE, MAPE, directional accuracy, etc.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE on non-zero days only (since MAPE is meaningless for very small values)
    nonzero = y_true > 0
    mape = np.mean(np.abs(
        (y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]
    )) * 100 if nonzero.sum() > 0 else np.nan

    # Directional accuracy (did we predict the right trend?)
    actual_change = np.diff(y_true)
    pred_change = np.diff(y_pred)
    directional = (np.sign(actual_change) == np.sign(pred_change)).mean() * 100

    # Zero-demand detection accuracy
    zero_acc = ((y_pred == 0) == (y_true == 0)).mean() * 100

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': directional,
        'zero_demand_accuracy': zero_acc,
    }

    return metrics


def evaluate_by_segment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segment_masks: dict,
) -> pd.DataFrame:
    """
    Evaluate performance per demand segment.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        segment_masks: Dict of {segment_name: boolean mask}

    Returns:
        DataFrame with MAE/RMSE per segment
    """
    results = []
    for seg_name, mask in segment_masks.items():
        if mask.sum() == 0:
            continue

        mae  = mean_absolute_error(y_true[mask], y_pred[mask])
        rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))

        results.append({
            'segment': seg_name,
            'n_samples': mask.sum(),
            'mae': mae,
            'rmse': rmse,
        })

    return pd.DataFrame(results)


def compare_to_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_naive: np.ndarray,
) -> dict:
    """
    Compare model performance against naive baseline (yesterday's value).

    Args:
        y_true: Actual values
        y_pred: Model predictions
        y_naive: Naive baseline (e.g., lag1)

    Returns:
        dict with comparison metrics
    """
    mae_model  = mean_absolute_error(y_true, y_pred)
    rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))

    mae_naive  = mean_absolute_error(y_true, y_naive)
    rmse_naive = np.sqrt(mean_squared_error(y_true, y_naive))

    improvement_mae = (mae_naive - mae_model) / mae_naive * 100
    improvement_rmse = (rmse_naive - rmse_model) / rmse_naive * 100

    return {
        'mae_model': mae_model,
        'mae_naive': mae_naive,
        'mae_improvement_pct': improvement_mae,
        'rmse_model': rmse_model,
        'rmse_naive': rmse_naive,
        'rmse_improvement_pct': improvement_rmse,
    }


def print_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_naive: np.ndarray = None,
    segment_masks: dict = None,
) -> None:
    """Pretty-print comprehensive evaluation summary."""
    metrics = evaluate_full_pipeline(y_true, y_pred)

    print("\n" + "="*60)
    print("  OVERALL PERFORMANCE")
    print("="*60)
    print(f"  MAE           : {metrics['mae']:.2f}")
    print(f"  RMSE          : {metrics['rmse']:.2f}")
    print(f"  MAPE          : {metrics['mape']:.1f}%")
    print(f"  Directional   : {metrics['directional_accuracy']:.1f}%")
    print(f"  Zero-demand   : {metrics['zero_demand_accuracy']:.1f}%")

    if y_naive is not None:
        comparison = compare_to_baseline(y_true, y_pred, y_naive)
        print("\n" + "="*60)
        print("  vs NAIVE BASELINE (yesterday's value)")
        print("="*60)
        print(f"  Model MAE     : {comparison['mae_model']:.2f}")
        print(f"  Baseline MAE  : {comparison['mae_naive']:.2f}")
        print(f"  Improvement   : {comparison['mae_improvement_pct']:.1f}%")

    if segment_masks is not None:
        seg_eval = evaluate_by_segment(y_true, y_pred, segment_masks)
        print("\n" + "="*60)
        print("  PER-SEGMENT PERFORMANCE")
        print("="*60)
        print(seg_eval.to_string(index=False))


if __name__ == "__main__":
    # Example usage
    y_true = np.array([10, 20, 30, 0, 5, 100])
    y_pred = np.array([12, 19, 35, 0, 8, 95])
    y_naive = np.array([8, 10, 20, 30, 0, 5])

    segment_masks = {
        'low': y_true < 15,
        'high': y_true >= 15,
    }

    print_summary(y_true, y_pred, y_naive, segment_masks)