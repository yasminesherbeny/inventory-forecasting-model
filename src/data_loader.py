# src/data_loader.py
# ============================================================
# Loads UCI Online Retail dataset, transforms to pipeline format,
# filters to top 20 high-signal products, simulates missing columns.
# Run once to produce dataset.csv — after that load from CSV.
# ============================================================

import os
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


# ── Constants ───────────────────────────────────────────────
TOP_N_PRODUCTS  = 20
MIN_COVERAGE    = 0.40      # product must sell on ≥40% of days
STORES          = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow']
UK_BANK_HOLIDAYS = [
    '2010-12-27', '2011-01-03', '2011-04-22', '2011-04-25',
    '2011-05-02', '2011-05-30', '2011-08-29', '2011-12-26',
    '2011-12-27',
]
RANDOM_SEED = 42


def load_raw() -> pd.DataFrame:
    """Fetch UCI Online Retail (id=352) and return combined raw dataframe."""
    print("Fetching UCI Online Retail dataset...")
    online_retail = fetch_ucirepo(id=352)
    df_raw = pd.concat(
        [online_retail.data.ids, online_retail.data.features], axis=1
    )
    print(f"Raw shape: {df_raw.shape}")
    return df_raw


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Remove cancellations, negatives, nulls; parse dates; filter to UK."""
    if 'InvoiceNo' in df.columns:
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df = df.dropna(subset=['StockCode', 'Quantity', 'UnitPrice', 'InvoiceDate'])

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Date'] = df['InvoiceDate'].dt.normalize()

    df = df[df['Country'] == 'United Kingdom'].copy()
    print(f"After cleaning (UK only): {len(df):,} rows")
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to one row per product-date."""
    daily = (
        df.groupby(['StockCode', 'Date'])
        .agg(Units_Sold=('Quantity', 'sum'), Price=('UnitPrice', 'mean'))
        .reset_index()
    )
    print(f"Daily aggregated: {daily.shape}")
    return daily


def filter_dense_products(daily: pd.DataFrame) -> pd.DataFrame:
    """Keep only products sold on ≥MIN_COVERAGE of all days."""
    total_days = daily['Date'].nunique()

    coverage = (
        daily[daily['Units_Sold'] > 0]
        .groupby('StockCode')['Date']
        .nunique()
        .reset_index()
        .rename(columns={'Date': 'active_days'})
    )
    coverage['coverage'] = coverage['active_days'] / total_days

    dense = coverage[coverage['coverage'] >= MIN_COVERAGE]['StockCode'].tolist()
    print(f"Products with ≥{MIN_COVERAGE*100:.0f}% coverage: {len(dense)}")
    return daily[daily['StockCode'].isin(dense)].copy()


def fill_date_grid(daily: pd.DataFrame) -> pd.DataFrame:
    """Expand to full date range — one row per product-date."""
    products   = daily['StockCode'].unique().tolist()
    date_range = pd.date_range(daily['Date'].min(), daily['Date'].max(), freq='D')

    full_index = pd.MultiIndex.from_product(
        [products, date_range], names=['StockCode', 'Date']
    )
    daily = (
        daily.set_index(['StockCode', 'Date'])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Fill zero-price days with product median
    daily['Price'] = daily.groupby('StockCode')['Price'].transform(
        lambda x: x.replace(0, np.nan).fillna(x.replace(0, np.nan).median())
    )
    return daily


def select_top_products(daily: pd.DataFrame) -> pd.DataFrame:
    """Keep TOP_N_PRODUCTS by combined lag1+lag7 autocorrelation signal."""
    results = []
    for pid in daily['StockCode'].unique():
        s = daily[daily['StockCode'] == pid]['Units_Sold']
        pct_nonzero = (s > 0).mean()
        if pct_nonzero < MIN_COVERAGE:
            continue
        results.append({
            'product'      : pid,
            'autocorr_lag1': s.autocorr(1),
            'autocorr_lag7': s.autocorr(7),
            'pct_nonzero'  : pct_nonzero,
        })

    ac_df = pd.DataFrame(results)
    ac_df['signal_score'] = (ac_df['autocorr_lag1'] + ac_df['autocorr_lag7']) / 2
    top = ac_df.nlargest(TOP_N_PRODUCTS, 'signal_score')

    print(f"\nTop {TOP_N_PRODUCTS} products selected.")
    print(f"  Mean lag1 autocorr : {top['autocorr_lag1'].mean():.3f}")
    print(f"  Mean lag7 autocorr : {top['autocorr_lag7'].mean():.3f}")

    return daily[daily['StockCode'].isin(top['product'].tolist())].copy(), top


def simulate_missing_columns(daily: pd.DataFrame) -> pd.DataFrame:
    """Simulate Discount, Holiday/Promotion, Inventory Level, Units Ordered."""
    np.random.seed(RANDOM_SEED)
    n = len(daily)

    daily['Discount'] = np.where(
        np.random.random(n) < 0.10,
        np.random.uniform(0.05, 0.30, n).round(2),
        0.0
    )

    bank_holidays = pd.to_datetime(UK_BANK_HOLIDAYS)
    daily['Holiday/Promotion'] = (
        daily['Date'].isin(bank_holidays) |
        (daily['Date'].dt.month == 12)
    ).astype(int)

    daily = daily.sort_values(['StockCode', 'Date'])
    rolling = (
        daily.groupby('StockCode')['Units_Sold']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        .fillna(0)
    )

    daily['Inventory Level'] = (
        rolling * np.random.uniform(1.5, 3.0, n) +
        np.random.uniform(0, 20, n)
    ).round(0).clip(0)

    daily['Units Ordered'] = (
        rolling * np.random.uniform(0.8, 1.5, n) +
        np.random.uniform(0, 10, n)
    ).round(0).clip(0)

    return daily


def assign_stores(daily: pd.DataFrame, top_products: list) -> pd.DataFrame:
    """Assign each product to one of 5 simulated UK store regions."""
    store_map = {pid: STORES[i % len(STORES)] for i, pid in enumerate(top_products)}
    daily['Store ID'] = daily['StockCode'].map(store_map)
    return daily


def build_dataset(save_path: str = "data/dataset.csv") -> pd.DataFrame:
    """
    Full pipeline: load → clean → aggregate → filter → simulate → save.
    If save_path already exists, loads from CSV instead.
    """
    if os.path.exists(save_path):
        print(f"Loading existing dataset from {save_path}")
        dataset = pd.read_csv(save_path, parse_dates=['Date'])
        print(f"Loaded: {dataset.shape}")
        return dataset

    df_raw  = load_raw()
    df_uk   = clean_raw(df_raw)
    daily   = aggregate_daily(df_uk)
    daily   = filter_dense_products(daily)
    daily   = fill_date_grid(daily)
    daily, top_products_df = select_top_products(daily)

    top_products = top_products_df['product'].tolist()
    daily = simulate_missing_columns(daily)
    daily = assign_stores(daily, top_products)

    # Rename to match pipeline
    dataset = daily.rename(columns={
        'StockCode' : 'Product ID',
        'Units_Sold': 'Units Sold',
    })
    dataset = dataset[[
        'Product ID', 'Store ID', 'Date',
        'Units Sold', 'Price', 'Discount',
        'Holiday/Promotion', 'Inventory Level', 'Units Ordered',
    ]]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset.to_csv(save_path, index=False)
    print(f"\nDataset saved → {save_path}")
    print(f"Final shape: {dataset.shape}")
    return dataset


if __name__ == "__main__":
    dataset = build_dataset()
    print(dataset.head())