import sqlite3
from typing import List, Dict

import numpy as np
import pandas as pd
from prophet import Prophet

from utils.logger import logger
from config import (
    DATABASE_FILE,
    FORECAST_PERIODS,
    MIN_HISTORY_DAYS,
    PROPHET_DAILY_SEASONALITY,
)

# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------
def _prepare_group_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse curated rows to one row per (product_id, store_id, date),
    summing demand/stock/energy and keeping a conservative min for expiry.
    """
    for c in [
        "units_sold", "available_stock", "max_capacity_per_day_liters",
        "energy_consumption_kwh", "transport_time_hours",
        "transport_temperature_c", "days_to_expiration"
    ]:
        if c not in df.columns:
            df[c] = 0

    g = (
        df.groupby(["product_id", "store_id", "date"], as_index=False)
          .agg(
              units_sold=("units_sold", "sum"),
              available_stock=("available_stock", "sum"),
              max_capacity_per_day_liters=("max_capacity_per_day_liters", "sum"),
              energy_consumption_kwh=("energy_consumption_kwh", "sum"),
              transport_time_hours=("transport_time_hours", "sum"),
              transport_temperature_c=("transport_temperature_c", "mean"),
              days_to_expiration=("days_to_expiration", "min"),
          )
    )
    g["date"] = pd.to_datetime(g["date"]).dt.floor("D")
    return g.sort_values(["product_id", "store_id", "date"])


def _fit_prophet_or_baseline(history: pd.DataFrame, future_dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Train Prophet on daily series (ds, y). If insufficient history,
    return a simple baseline (mean of last up-to-7 days).
    """
    history = history.groupby("ds", as_index=False).agg(y=("y", "sum")).sort_values("ds")
    unique_days = history["ds"].nunique()

    if unique_days < max(3, MIN_HISTORY_DAYS):
        tail = history.tail(7)
        mean_val = float(tail["y"].mean()) if not tail.empty else 0.0
        return np.full(len(future_dates), max(0.0, mean_val), dtype=float)

    try:
        model = Prophet(
            daily_seasonality=bool(PROPHET_DAILY_SEASONALITY),
            weekly_seasonality=True,
        )
        model.fit(history)
        future = pd.DataFrame({"ds": future_dates})
        forecast = model.predict(future)
        yhat = forecast["yhat"].to_numpy(dtype=float)
        return np.clip(yhat, 0.0, None)
    except Exception as e:
        logger.exception("Prophet failed; using baseline. Reason: %s", e)
        tail = history.tail(7)
        mean_val = float(tail["y"].mean()) if not tail.empty else 0.0
        return np.full(len(future_dates), max(0.0, mean_val), dtype=float)


def _ensure_product_forecast_table(conn: sqlite3.Connection) -> None:
    """
    Ensure product-level forecast schema. If a legacy table with batch_id exists, recreate it.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(milk_forecast)")
    cols = [r[1].lower() for r in cur.fetchall()]
    needs_recreate = False
    if cols and "batch_id" in cols:
        needs_recreate = True

    if needs_recreate:
        logger.info("Recreating milk_forecast table for product-level schema...")
        cur.execute("DROP TABLE IF EXISTS milk_forecast")

    cur.execute("""
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
    """)
    conn.commit()

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def run_incremental_forecast(sales_curated_df: pd.DataFrame, forecast_days: int = None) -> List[Dict]:
    """
    Product/Store incremental forecasting.

    - Aggregates curated sales to product_id × store_id × date
    - Trains Prophet (daily + weekly) when history >= MIN_HISTORY_DAYS; baseline otherwise
    - Writes results to SQLite table milk_forecast (product-level schema)
    - Returns a list[dict] for the app
    """
    if sales_curated_df is None or sales_curated_df.empty:
        logger.warning("No curated sales provided for forecasting.")
        return []

    forecast_days = int(forecast_days or FORECAST_PERIODS)
    logger.info(" Running product-level forecasting for %d days...", forecast_days)

    # Prepare daily product/store series
    daily = _prepare_group_daily(sales_curated_df)

    results: List[Dict] = []
    with sqlite3.connect(DATABASE_FILE) as conn:
        _ensure_product_forecast_table(conn)
        cursor = conn.cursor()

        # Load existing forecasts to keep it incremental
        existing = pd.read_sql_query(
            "SELECT product_id, store_id, forecast_date FROM milk_forecast",
            conn
        )
        if not existing.empty:
            existing["forecast_date"] = pd.to_datetime(existing["forecast_date"]).dt.floor("D")

        for (pid, sid), grp in daily.groupby(["product_id", "store_id"], as_index=False):
            grp = grp.sort_values("date")
            last_hist_date = grp["date"].max()

            # Build future index
            future_index = pd.date_range(last_hist_date + pd.Timedelta(days=1),
                                         periods=forecast_days, freq="D")

            # Remove dates already predicted for this (product, store)
            if not existing.empty:
                done_mask = (existing["product_id"] == pid) & (existing["store_id"] == str(sid))
                done_dates = pd.to_datetime(existing.loc[done_mask, "forecast_date"]).dt.floor("D")
                future_index = future_index.difference(done_dates)

            if future_index.empty:
                continue

            # Prophet/baseline on daily units
            hist = grp[["date", "units_sold"]].rename(columns={"date": "ds", "units_sold": "y"})
            yhat = _fit_prophet_or_baseline(hist, future_index)

            # Heuristics
            last_stock = float(grp["available_stock"].iloc[-1]) if "available_stock" in grp.columns else 0.0
            total_forecast = float(np.sum(yhat))
            cap_sum = float(grp.get("max_capacity_per_day_liters", pd.Series([0])).sum())

            suggested_total = max(0.0, total_forecast - last_stock)
            if cap_sum > 0:
                suggested_total = min(suggested_total, cap_sum * len(future_index))

            exp_risk = int(((grp["days_to_expiration"] <= 3) & (grp["available_stock"] > 0)).any())
            avg_energy = float(grp.get("energy_consumption_kwh", pd.Series([0.0])).mean())

            per_day_waste = max(0.0, last_stock - total_forecast) / max(1, len(future_index))
            per_day_sugg = suggested_total / max(1, len(future_index))
            per_day_cost = avg_energy

            for i, fdate in enumerate(future_index):
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

                # Upsert
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

    logger.info(" Forecasting completed. Inserted %d new forecast rows.", len(results))
    return results
