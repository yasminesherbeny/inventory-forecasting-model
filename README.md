# Inventory Demand Forecasting

A two-stage LightGBM pipeline for next-day product demand forecasting across multiple stores, deployed as a REST API on Hugging Face Spaces.

---

## Overview

This project predicts next-day inventory demand at the store-product level. The model handles zero-inflated demand data (48% of days have zero demand) through a two-stage architecture: a binary classifier that gates zero vs. positive demand, followed by segmented regressors that predict the actual demand quantity.

**Final Results:**

| Metric | Naive Baseline | Final Model |
|--------|---------------|-------------|
| MAE | 48.20 | **30.55** |
| RMSE | 104.18 | **71.15** |
| Improvement | — | **36.6%** |
| Zero-demand accuracy | — | **96.1%** |
| Directional accuracy | — | **64.5%** |

---

## Model Architecture

### Stage 1 — LGBMClassifier
- **Task:** Binary classification — will there be any demand tomorrow?
- **Output:** 0 (no demand) or 1 (positive demand)
- **Accuracy:** 96.1%
- Handles zero-inflated days directly rather than forcing a regressor to learn them

### Stage 2 — Segmented LGBMRegressors
- **Task:** Predict the actual demand quantity (only called when Stage 1 predicts positive demand)
- **Routing:** Deterministic hard threshold on `lag1` — no learned router, no misclassification risk
  - `lag1 < 80` → Low-demand regressor
  - `lag1 ≥ 80` → High-demand regressor
- Each regressor is trained exclusively on its own demand segment

```
Features
   │
   ▼
[Stage 1 — Classifier]
   │
   ├── No demand → Forecast = 0
   │
   └── Positive demand
            │
            ▼
      [lag1 threshold router]
            │
     ┌──────┴──────┐
     ▼             ▼
 Low model    High model
     │             │
     └──────┬──────┘
            ▼
     Final demand forecast
```

---

## Dataset

**Source:** [UCI Online Retail Dataset (ID=352)](https://archive.ics.uci.edu/dataset/352/online+retail) — loaded directly via `ucimlrepo`, no login required.

**Transformation pipeline:**
- Loaded 485,000 UK retail transactions
- Aggregated to daily demand per product
- Filtered to top 20 products by **autocorrelation signal** (not volume) to ensure learnable patterns
- Simulated realistic missing columns (`Discount`, `Inventory Level`, `Units Ordered`) tied to rolling demand

**Final dataset:** 7,480 rows · 20 products · 5 stores · 374 days

**Key diagnostic:** Mean lag-1 autocorrelation jumped from `0.0001` (synthetic data) → `0.324` (real data), confirming learnable signal.

---

## Features

| Feature | Description |
|---------|-------------|
| `lag1` | Units sold yesterday |
| `lag7`, `lag14` | Units sold 7 and 14 days prior |
| `rolling_mean_7/14/30` | Rolling average demand |
| `rolling_std_7/14/30` | Rolling standard deviation |
| `inventory_level` | Current stock on hand |
| `price`, `discount` | Pricing signals |
| `day_of_week`, `month` | Calendar features |
| `holiday_promotion` | Holiday / promotion flag |

---

## Why LightGBM?

| | XGBoost | Random Forest | LSTM | LightGBM ✓ |
|---|---|---|---|---|
| Sparse/skewed data | ❌ | ⚠️ | ⚠️ | ✅ |
| Small tabular datasets | ✅ | ✅ | ❌ | ✅ |
| Captures demand spikes | ⚠️ | ❌ | ✅ | ✅ |
| Training speed | ⚠️ | ⚠️ | ❌ | ✅ |
| Interpretability | ✅ | ✅ | ❌ | ✅ |
| One framework for both tasks | ❌ | ❌ | ❌ | ✅ |

LightGBM's leaf-wise tree growth targets the highest-loss leaves first, making it naturally effective at capturing rare high-demand spikes — exactly what inventory forecasting requires.

---

## Installation

```bash
git clone https://github.com/yasminesherbeny/forecasting-model.git
cd YOUR_REPO
pip install -r requirements.txt
```

**Requirements:**
```
lightgbm
pandas
numpy
scikit-learn
flask
ucimlrepo
flask-restx
```

---

## Usage

### Run the API locally

```bash
python app.py
```

The API will be available at `http://localhost:5000`. Navigate to `/swagger` for the interactive Swagger UI.

### API Endpoint

**POST** `/predict`

```json
{
  "store_id": 1,
  "product_id": "85123A",
  "last_30_days": [
    { "date": "2024-01-01", "units_sold": 45, "inventory_level": 200, "price": 2.55, "discount": 0.0, "units_ordered": 50 },
    ...
  ]
}
```

**Response:**

```json
{
  "store_id": 1,
  "product_id": "85123A",
  "forecast_date": "2024-02-01",
  "predicted_demand": 38
}
```

---

## Database Requirements

For integration with a live database, each record must follow this schema:

| Column | Type | Description |
|--------|------|-------------|
| `store_id` | INT | Store identifier |
| `product_id` | VARCHAR | Product identifier |
| `date` | DATE | Record date (one row per store-product-date, no duplicates) |
| `units_sold` | INT | Units sold that day |
| `units_ordered` | INT | Units ordered that day |
| `inventory_level` | INT | Stock on hand |
| `price` | FLOAT | Unit price |
| `discount` | FLOAT | Discount applied (0–1) |
| `holiday_promotion` | BOOL | Holiday or promotion flag |

> Minimum 30 days of history must be retained per store-product pair for lag features to be computed correctly.

---

## Deployment

The model is deployed on **Hugging Face Spaces** as a Flask REST API with a **Swagger UI** for interactive testing.

🔗 **Live API:** (https://huggingface.co/spaces/yasminesherbeny/forecast-api)

---

## Project Structure

```
├── app.py                  # Flask API entry point
├── model/
│   ├── classifier.pkl      # Stage 1 — LGBMClassifier
│   ├── regressor_low.pkl   # Stage 2 — Low-demand regressor
│   └── regressor_high.pkl  # Stage 2 — High-demand regressor
├── pipeline/
│   ├── preprocess.py       # Feature engineering
│   ├── predict.py          # Two-stage prediction logic
│   └── train.py            # Model training scripts
├── data/
│   └── prepare_dataset.py  # UCI data loading & transformation
├── requirements.txt
└── README.md
```

---

## Key Lessons

- **Autocorrelation diagnostics first** — no model can learn from random data regardless of architecture
- **Two-stage over single regressor** — zero-inflated data needs explicit handling, not regression averaging
- **Hard routing over learned routing** — when a deterministic signal is always available at prediction time, use it
- **Data quality over model complexity** — switching datasets improved performance more than any architecture change

---

## License

MIT License — see [LICENSE](LICENSE) for details.
