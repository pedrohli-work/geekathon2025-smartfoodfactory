🥛 Smart Milk Factory

A lightweight end-to-end demo of a dairy operations pipeline and dashboard.
It generates synthetic but realistic data (production, inventory, sales, factory, logistics), runs an ETL + forecasting pipeline, and presents actionable KPIs in a Streamlit app.

Built for Geekathon: shows demand forecasting, stock outlook, shelf-life risk, and sustainability proxies—simple to run locally or on Streamlit Cloud.

✨ What you get

Dashboard (Streamlit app.py)

Overview KPIs

Forecasted Units (next N days) — sum of product×store forecasts within the selected horizon

Predicted Waste (units) — heuristic waste proxy from pipeline

Near Expiration (≤ 3 days) — count of at-risk items

Current Available Stock (L) — latest available stock

Demand vs. Projected Stock (line)
Projected stock = last known stock − cumulative forecast (clipped at 0).

Batch Drill-down (wide table)
Lot-level details: batch, date, stock, shelf-life, temperatures, energy.

Expiration Risk by Product & Store (heatmap)
Worst risk per pair using thresholds (≤1 day = High, ≤3 = Medium, else Low).

Suggested Production Plan (table)
Daily liters/day suggested to meet demand within capacity.

Sustainability & Energy
Factory kWh/L (proxy), average transport temperature, average logistics kWh/day.

Pipeline (main.py)

Runs ETL (modules/etl) + Forecast (modules/forecasting, default ETS)

Saves a flat dashboard JSON to curated/dashboard_output.json

Fallback: if no new forecasts were created, reads recent horizon from DB.

Synthetic Data

synthetic_data/datasets_sqlite.py — builds a fresh SQLite schema + initial data

synthetic_data/datsets_sqlite_dynamic.py — appends new data (incremental)

🧱 Architecture (high-level)
SQLite (database/milk_factory.db)
   ├─ product_catalog (stable product_id → product_type)
   ├─ milk_production_base, milk_quality
   ├─ milk_inventory, milk_sales
   ├─ milk_factory, milk_logistics
   └─ milk_forecast (product_id, store_id, forecast_date, metrics...)

ETL (modules/etl.py)
   ├─ Writes curated parquet:
   │    curated/catalog_curated.parquet
   │    curated/sales_curated.parquet
   └─ Provides DataFrames to forecasting

Forecast (modules/forecasting.py, default ETS)
   ├─ Aggregates to product_id × store_id × date
   ├─ Fits ETS (or baseline if history too short)
   ├─ Heuristics for waste, capacity, logistics
   └─ Upserts into SQLite::milk_forecast (+ returns list of dicts)

Export (main.py)
   └─ Writes curated/dashboard_output.json

📁 Repo layout (key files)
app.py                         # Streamlit dashboard
main.py                        # Orchestration: ETL + Forecast + JSON export
config.py                      # Paths & knobs (FORECAST_PERIODS, LOG_FILE, etc.)
modules/
  ├─ etl.py                    # ETL into curated/*.parquet
  └─ forecasting.py            # ETS model / baseline + SQLite writes
synthetic_data/
  ├─ datasets_sqlite.py        # create SQLite schema + seed data (DROP/CREATE)
  └─ datsets_sqlite_dynamic.py # append new data (typo 'datsets' is intentional here)
curated/                       # artifacts written at runtime (JSON, parquet)
database/
  └─ milk_factory.db           # SQLite DB (created at runtime)
utils/
  └─ logger.py                 # file+console logging
requirements.txt               # Python deps (no Prophet)
README.md

⚙️ Requirements

Python 3.10+ (3.11 recommended)

pip for dependencies

requirements.txt:

pandas
numpy
plotly
streamlit
pyarrow
fastparquet
statsmodels>=0.14


Prophet is not required—forecasting uses ETS (statsmodels).

Create database (rebuilds fresh schema/data)

Generate new data (append rows)

Run pipeline (ETL + Forecast + export)

Reset dashboard (clears curated/, DB, log, cache)

🌐 Deploy to Streamlit Cloud

Push the repo to GitHub.

In Streamlit Cloud, set Main file path to app.py.

(Optional) Set environment variables (override defaults), e.g.:

FORECAST_PERIODS=30

MIN_HISTORY_DAYS=7

FORECAST_MODEL=ets (if you expose this in config.py)

Deploy. The app will create database/milk_factory.db and curated/* at runtime.

You do not need to commit the DB or curated files. The app creates them.

🔧 Config knobs (edit config.py)

DATABASE_FILE — path to SQLite DB (default database/milk_factory.db)

OUTPUT_JSON — dashboard JSON (default curated/dashboard_output.json)

CATALOG_CURATED, SALES_CURATED — parquet outputs from ETL

Forecasting

FORECAST_PERIODS — horizon (days), e.g. 14 or 30

MIN_HISTORY_DAYS — required history per (product,store) for ETS; below this, fallback baseline is used

(Optional) FORECAST_MODEL — keep "ets" (default)

LOG_FILE — path to pipeline log, shown in the app (“Run Log (tail)”)

Override via environment variables if desired.

🧪 Using the synthetic data scripts

Create from scratch
python synthetic_data/datasets_sqlite.py
Drops & recreates tables, seeds data, and writes initial inventory/sales/factory/logistics.

Append new data
python synthetic_data/datsets_sqlite_dynamic.py
Appends rows—simulates continuous operations.
Tune inside the script (e.g., days=14, n_batches=60) to “thicken” history and make curves more realistic.

The product catalog (product_catalog) stabilizes product_id → product_type.
If you want exactly one SKU per type, set in datasets_sqlite.py:

PRODUCT_TYPES = ["whole milk", "skimmed milk", "yogurt", "cheese", "cream"]
N_PRODUCTS = len(PRODUCT_TYPES)
def build_product_catalog(n=N_PRODUCTS):
    return {i + 1: PRODUCT_TYPES[i] for i in range(n)}


Re-run datasets_sqlite.py after changing this (it rebuilds the DB).

🔮 Forecasting details

Granularity: product_id × store_id × daily

Model: ETS (statsmodels Exponential Smoothing)

If a group has < MIN_HISTORY_DAYS, uses a simple baseline (last-7-day mean)

Safety: predictions clipped to ≥ 0

Extras: heuristic predicted_waste, suggested_production (bounded by daily capacity), expiration_risk, logistics cost proxy

Results are written to SQLite table milk_forecast and exported to curated/dashboard_output.json.

🧹 Resetting / starting clean

In-app: click “Reset dashboard (clear data & cache)”
→ deletes curated/ dir, database/milk_factory.db, LOG_FILE, clears app cache, and reruns.

Manual: delete the same paths and restart the app.

🆘 Troubleshooting

Empty dashboard / zeros everywhere

Click Create database, then Run pipeline.

Ensure curated/dashboard_output.json is created by main.py.

“File not found: datsets_sqlite_dynamic.py”

The filename is intentionally spelled datsets_sqlite_dynamic.py in this repo. Keep path consistent in app.py.

Duplicate product names in filter

If you generate many SKUs with the same type, you’ll see multiple “• Cheese”.
Use a stable catalog (see “Synthetic data” note) to make IDs deterministic—or one SKU per type.

Streamlit duplicate element key

Ensure each widget has a unique key. The provided app.py already does this.

Windows console encoding errors (emojis)

The dynamic script avoids emojis. If you add any, use UTF-8 console or remove them.

📈 Tips for “more realistic” numbers

In datsets_sqlite_dynamic.py bump:

days=14, n_batches=60 → more history → better ETS fit

In config.py:

FORECAST_PERIODS=30 → more forecasted days (more rows)

MIN_HISTORY_DAYS=7 → more groups qualify for ETS (instead of baseline)

🔌 Extending the demo

Add evaluation (MAPE/SMAPE) comparing baseline vs. ETS

Add cost of stockouts, revenue at risk, or cold-chain breaches

Ingest real datasets by replacing synthetic scripts and adjusting ETL

Swap ETS for your favorite model (ARIMA, ML, etc.) inside modules/forecasting.py

📜 License

MIT (or your preferred license). Add a LICENSE file if needed.

🙌 Acknowledgements

Built with Streamlit, pandas, statsmodels, plotly.

Thanks to Geekathon organizers & reviewers!

🧭 One-liner workflow
# Fresh run (local dev)
python synthetic_data/datasets_sqlite.py && python main.py && streamlit run app.py
