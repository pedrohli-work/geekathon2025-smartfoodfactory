"""
Main orchestration pipeline for Smart Milk Factory:
1) Run ETL to build curated datasets
2) Run incremental forecasting
3) Export a dashboard-friendly JSON (with fallback to existing forecasts if no new rows)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from utils.logger import get_logger
from config import (
    DATABASE_FILE,
    OUTPUT_JSON,
    FORECAST_PERIODS,
    FORECAST_MODEL,  # <-- só para logar o modelo escolhido
)
from modules.etl import build_integrated_pipeline
from modules.forecasting import run_incremental_forecast

logger = get_logger(__name__)


def _serialize_forecast_for_json(forecast_results: List[Dict]) -> list[dict]:
    """
    Convert the list of forecast dicts into a lightweight JSON-serializable list.
    We keep only the fields used by the dashboard.
    """
    payload = []
    for r in forecast_results:
        payload.append({
            "product_id": r.get("product_id"),
            "store_id": r.get("store_id"),
            "forecast_date": pd.to_datetime(r.get("forecast_date")).strftime("%Y-%m-%d") if r.get("forecast_date") else None,
            "forecasted_sales": float(r.get("forecasted_sales", 0.0) or 0.0),
            "predicted_waste": float(r.get("predicted_waste", 0.0) or 0.0),
            "suggested_production": float(r.get("suggested_production", 0.0) or 0.0),
            "expiration_risk": int(r.get("expiration_risk", 0) or 0),
            "predicted_logistics_cost": float(r.get("predicted_logistics_cost", 0.0) or 0.0),
        })
    return payload


def _load_recent_forecasts_from_db(horizon_days: int) -> List[Dict]:
    """
    Fallback: if the incremental step didn't create new rows, read the most recent
    horizon from the existing milk_forecast table so the dashboard isn't empty.

    Strategy:
    - Try next-dates (>= today). If empty,
    - Use the last available date in the table, and take the last <horizon_days> days window.
    """
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            df = pd.read_sql_query(
                """
                SELECT product_id, store_id, forecast_date,
                       forecasted_sales, predicted_waste,
                       suggested_production, expiration_risk,
                       predicted_logistics_cost
                FROM milk_forecast
                """,
                conn,
            )
    except Exception as e:
        logger.exception("Failed to read milk_forecast for fallback: %s", e)
        return []

    if df.empty:
        return []

    df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
    df = df.dropna(subset=["forecast_date"])

    # 1) Prefer future window (>= today)
    today = pd.Timestamp.today().normalize()
    future = df[df["forecast_date"] >= today].copy()
    if not future.empty:
        max_date = future["forecast_date"].min() + pd.Timedelta(days=horizon_days - 1)
        future = future[future["forecast_date"] <= max_date]
        logger.info("Using fallback: %d future forecast rows from DB.", len(future))
        return future.to_dict(orient="records")

    # 2) Otherwise, take the last available window
    last_date = df["forecast_date"].max()
    start = last_date - pd.Timedelta(days=horizon_days - 1)
    window = df[(df["forecast_date"] >= start) & (df["forecast_date"] <= last_date)].copy()
    logger.info("Using fallback: %d recent forecast rows from DB.", len(window))
    return window.to_dict(orient="records")


def main() -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    Execute ETL + Forecast + JSON export.
    Returns (catalog_df, sales_curated_df, forecast_results) for optional downstream usage.
    """
    logger.info("🚀 Starting Smart Milk Factory pipeline (ETL + Forecast + JSON)")

    # Ensure output dir
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    # 1) ETL
    with sqlite3.connect(DATABASE_FILE) as conn:
        logger.info("🔧 Running ETL pipeline...")
        catalog_df, sales_curated_df = build_integrated_pipeline(conn)

        if catalog_df is None or catalog_df.empty:
            logger.warning("Catalog is empty after ETL.")
        else:
            logger.info("Catalog ready: %d rows (unique batches/products).", len(catalog_df))

        if sales_curated_df is None or sales_curated_df.empty:
            logger.warning("Sales curated is empty after ETL — forecasting may be skipped.")
        else:
            logger.info("Sales curated ready: %d rows.", len(sales_curated_df))

    # 2) Forecast (incremental insert)
    logger.info("📈 Running incremental forecasting for %d days (model=%s)...", FORECAST_PERIODS, FORECAST_MODEL)
    forecast_results = run_incremental_forecast(sales_curated_df, forecast_days=FORECAST_PERIODS)

    # 2b) Fallback: if nothing new was inserted/predicted, load existing forecasts for the next horizon
    if not forecast_results:
        logger.info("No new forecast rows were created; loading existing forecasts from DB as fallback.")
        forecast_results = _load_recent_forecasts_from_db(FORECAST_PERIODS)

    # 3) Export dashboard JSON (flat rows)
    output_data = _serialize_forecast_for_json(forecast_results)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    logger.info("💾 Dashboard JSON saved to %s (%d rows).", OUTPUT_JSON, len(output_data))

    logger.info("✅ Pipeline finished successfully.")
    return catalog_df, sales_curated_df, forecast_results


if __name__ == "__main__":
    main()
