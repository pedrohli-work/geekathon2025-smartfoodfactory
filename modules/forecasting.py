"""
Incremental product-level forecasting for Smart Milk Factory.

- Trains a demand model per (product_id, store_id) on daily sales.
- Uses ETS (Holt-Winters, statsmodels) when available; otherwise falls back to a baseline.
- Writes/updates forecasts into SQLite table `milk_forecast` with product-level schema.
- Returns a list[dict] with per-day forecasts for the app.

This version intentionally removes Prophet and is safe to run in environments
without heavy dependencies (e.g., Streamlit Cloud). If statsmodels is missing,
it still works using the baseline forecaster.
"""

from __future__ import annotations

import sqlite3
from typing import List, Dict

import numpy as np
import pandas as pd

from utils.logger import logger
from config import (
    DATABASE_FILE,
    FORECAST_PERIODS,
    MIN_HISTORY_DAYS,
    FORECAST_MODEL,  # "ets" (default) or "baseline"
)

# Try ETS (statsmodels); if not available, we will fall back to baseline.
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    _HAVE_ETS = True
except Exception:
    ExponentialSmoothing = None  # type: ignore
    _HAVE_ETS = False


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _prepare_group_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse curated rows to one row per (product_id, store_id, date),
    summing demand/stock/energy and keeping a conservative min for expiry.

    Ensures the following columns exist (filling with zeros when missing):
      - units_sold, available_stock, max_capacity_per_day_liters,
        energy_consumption_kwh, transport_time_hours, transport_temperature_c,
        days_to_expiration
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # Ensure expected columns (fill with zeros/defaults if missing)
    for c in [
        "units_sold",
        "available_stock",
        "max_capacity_per_day_liters",
        "energy_consumption_kwh",
        "transport_time_hours",
        "transport_temperature_c",
        "days_to_expiration",
    ]:
        if c not in d.columns:
            d[c] = 0

    # Coerce core columns
    for c in ["product_id"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("Int64")
    if "store_id" in d.columns:
        d["store_id"] = d["store_id"].astype(str)
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.floor("D")

    # Aggregate to daily per product/store
    g = (
        d.groupby(["product_id", "store_id", "date"], as_index=False)
         .agg(
             units_sold=("units_sold", "sum"),
             available_stock=("available_stock", "sum"),
             max_capacity_per_day_liters=("max_capacity_per_day_liters", "sum"),
             energy_consumption_kwh=("energy_consumption_kwh", "sum"),
             transport_time_hours=("transport_time_hours", "sum"),
             transport_temperature_c=("transport_temperature_c", "mean"),
             days_to_expiration=("days_to_expiration", "min"),  # worst case (closest expiry)
         )
    )
    # Clean types
    g["product_id"] = g["product_id"].astype("Int64")
    g["store_id"] = g["store_id"].astype(str)
    g["date"] = pd.to_datetime(g["date"]).dt.floor("D")

    return g.sort_values(["product_id", "store_id", "date"])


def _fit_demand_model(history: pd.DataFrame, future_dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Train the selected model ("ets" or "baseline") on daily series (ds, y).
    Falls back to baseline when model isn't available or history is too short.

    Baseline = mean of the last up-to-7 historical days (non-negative).
    """
    history = history.copy()
    # Aggregate by day and sort
    history = history.groupby("ds", as_index=False).agg(y=("y", "sum")).sort_values("ds")

    def _baseline(n: int) -> np.ndarray:
        tail = history.tail(7)
        mean_val = float(tail["y"].mean()) if not tail.empty else 0.0
        return np.full(n, max(0.0, mean_val), dtype=float)

    # Not enough history -> baseline
    n_days = history["ds"].nunique()
    if n_days < max(3, MIN_HISTORY_DAYS):
        return _baseline(len(future_dates))

    model_choice = (FORECAST_MODEL or "ets").lower()

    # ETS (Holt-Winters): trend + optional weekly seasonality
    if model_choice == "ets":
        if not _HAVE_ETS:
            logger.warning("FORECAST_MODEL=ets but statsmodels not available; using baseline.")
            return _baseline(len(future_dates))
        try:
            y = history["y"].astype(float).to_numpy()
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            # Use weekly seasonality if we have at least ~2 weeks
            seasonal_periods = 7 if len(y) >= 14 else None
            if seasonal_periods:
                model = ExponentialSmoothing(
                    y,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated",
                ).fit(optimized=True)
            else:
                model = ExponentialSmoothing(
                    y,
                    trend="add",
                    seasonal=None,
                    initialization_method="estimated",
                ).fit(optimized=True)

            yhat = model.forecast(len(future_dates))
            return np.clip(np.asarray(yhat, dtype=float), 0.0, None)
        except Exception as e:
            logger.exception("ETS failed; using baseline. Reason: %s", e)
            return _baseline(len(future_dates))

    # Default: baseline
    return _baseline(len(future_dates))


def _ensure_product_forecast_table(conn: sqlite3.Connection) -> None:
    """
    Ensure a product-level forecast table exists.
    If an old batch-level schema existed, we drop it and recreate.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(milk_forecast)")
    cols = [r[1].lower() for r in cur.fetchall()]
    needs_recreate = False
    if cols:
        # Legacy schema contained 'batch_id'; we need a product-level PK instead
        if "batch_id" in cols:
            needs_recreate = True

    if needs_recreate:
        logger.info("Recreating milk_forecast table for product-level schema...")
        cur.execute("DROP TABLE IF EXISTS milk_forecast")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS milk_forecast (
            product_id INTEGER,
            store_id   TEXT,
            forecast_date DATE,
            forecasted_sales REAL,
            predicted_waste REAL,
            suggested_production REAL,
            expiration_risk INTEGER,
            predicted_logistics_cost REAL,
            PRIMARY KEY (product_id, store_id, forecast_date)
        )
        """
    )
    conn.commit()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def run_incremental_forecast(sales_curated_df: pd.DataFrame, forecast_days: int | None = None) -> List[Dict]:
    """
    Product/Store incremental forecasting.

    - Aggregates curated sales to product_id × store_id × date
    - Trains ETS when available (or baseline)
    - Writes results to SQLite table milk_forecast (product-level schema)
    - Returns a list[dict] for the app
    """
    if sales_curated_df is None or sales_curated_df.empty:
        logger.warning("No curated sales provided for forecasting.")
        return []

    forecast_days = int(forecast_days or FORECAST_PERIODS)
    logger.info("Running product-level forecasting for %d days...", forecast_days)

    # Prepare daily product/store series
    daily = _prepare_group_daily(sales_curated_df)
    if daily.empty:
        logger.warning("Daily aggregated data is empty; nothing to forecast.")
        return []

    results: List[Dict] = []

    with sqlite3.connect(DATABASE_FILE) as conn:
        _ensure_product_forecast_table(conn)
        cursor = conn.cursor()

        # Load already-forecasted dates to keep it incremental
        try:
            existing = pd.read_sql_query(
                "SELECT product_id, store_id, forecast_date FROM milk_forecast",
                conn
            )
        except Exception:
            existing = pd.DataFrame(columns=["product_id", "store_id", "forecast_date"])

        if not existing.empty:
            # Normalize types for consistent comparison
            if "product_id" in existing.columns:
                existing["product_id"] = pd.to_numeric(existing["product_id"], errors="coerce").astype("Int64")
            if "store_id" in existing.columns:
                existing["store_id"] = existing["store_id"].astype(str)
            if "forecast_date" in existing.columns:
                existing["forecast_date"] = pd.to_datetime(existing["forecast_date"], errors="coerce").dt.floor("D")

        # Forecast per group
        for (pid, sid), grp in daily.groupby(["product_id", "store_id"], as_index=False):
            grp = grp.sort_values("date")
            if grp.empty:
                continue

            last_hist_date = pd.to_datetime(grp["date"]).max()
            if pd.isna(last_hist_date):
                continue

            # Future horizon
            future_dates = pd.date_range(last_hist_date + pd.Timedelta(days=1),
                                         periods=forecast_days, freq="D")

            # Remove dates already predicted (incremental)
            if not existing.empty:
                done_mask = (existing["product_id"] == pid) & (existing["store_id"] == str(sid))
                done_dates = set(existing.loc[done_mask, "forecast_date"].tolist())
                future_dates = [d for d in future_dates if d not in done_dates]

            if len(future_dates) == 0:
                continue

            # Fit model on daily demand
            hist = grp[["date", "units_sold"]].rename(columns={"date": "ds", "units_sold": "y"})
            yhat = _fit_demand_model(hist, pd.DatetimeIndex(future_dates))

            # Heuristics for waste, capacity and logistics
            last_stock = float(grp.get("available_stock", pd.Series([0.0])).iloc[-1])
            total_forecast = float(np.sum(yhat))
            cap_sum = float(grp.get("max_capacity_per_day_liters", pd.Series([0.0])).sum())

            # Suggested production (bounded by capacity if available)
            suggested_total = max(0.0, total_forecast - last_stock)
            if cap_sum > 0:
                suggested_total = min(suggested_total, cap_sum * len(future_dates))

            # Expiration risk: if any near-expiry in last history AND stock>0
            exp_risk = int(((grp["days_to_expiration"] <= 3) & (grp["available_stock"] > 0)).any())

            # Logistics cost proxy: average historical energy (factory+transport if present)
            avg_energy = float(grp.get("energy_consumption_kwh", pd.Series([0.0])).mean())

            # Distribute totals uniformly across the horizon (simple)
            per_day_waste = max(0.0, last_stock - total_forecast) / max(1, len(future_dates))
            per_day_sugg = suggested_total / max(1, len(future_dates))
            per_day_cost = avg_energy  # keep same scale as historical day

            for i, fdate in enumerate(future_dates):
                row = {
                    "product_id": int(pid) if pd.notna(pid) else None,
                    "store_id": str(sid),
                    "forecast_date": pd.to_datetime(fdate),
                    "forecasted_sales": float(yhat[i]),
                    "predicted_waste": float(per_day_waste),
                    "suggested_production": float(per_day_sugg),
                    "expiration_risk": int(exp_risk),
                    "predicted_logistics_cost": float(per_day_cost),
                }
                results.append(row)

                # Upsert into SQLite
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO milk_forecast
                    (product_id, store_id, forecast_date,
                     forecasted_sales, predicted_waste, suggested_production,
                     expiration_risk, predicted_logistics_cost)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        row["product_id"],
                        row["store_id"],
                        row["forecast_date"].strftime("%Y-%m-%d"),
                        row["forecasted_sales"],
                        row["predicted_waste"],
                        row["suggested_production"],
                        row["expiration_risk"],
                        row["predicted_logistics_cost"],
                    ),
                )

        conn.commit()

    logger.info("Forecasting completed. Inserted %d new forecast rows.", len(results))
    return results
