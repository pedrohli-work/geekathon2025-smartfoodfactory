"""
Global configuration for the Smart Milk Factory project.

This module centralizes:
- Base paths and filenames
- SQLite connection helper with recommended PRAGMAs
- Canonical table names (keep in sync with the DB schema)
- Forecast/ML settings (overridable via environment variables)
- Curated/output file locations
- Utility helpers for date formatting
"""

from pathlib import Path
import os
import sqlite3
import pandas as pd

# -----------------------------------------------------------------------------
# Base paths
# -----------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATABASE_FILE: Path = BASE_DIR / "database" / "milk_factory.db"
DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)

# Logs (used by utils/logger.py)
LOG_DIR: Path = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE: Path = LOG_DIR / "milk_factory.log"

# -----------------------------------------------------------------------------
# Canonical table names (keep in sync with your schema)
# -----------------------------------------------------------------------------
TABLE_PRODUCTION = "milk_production_base"
TABLE_QUALITY    = "milk_quality"
TABLE_INVENTORY  = "milk_inventory"
TABLE_SALES      = "milk_sales"
TABLE_FACTORY    = "milk_factory"
TABLE_LOGISTICS  = "milk_logistics"
TABLE_FORECAST   = "milk_forecast"  # created by forecasting step

# -----------------------------------------------------------------------------
# Forecast / ML settings (can be overridden via environment variables)
# -----------------------------------------------------------------------------
FORECAST_PERIODS   = int(os.getenv("FORECAST_PERIODS", "30"))  # number of days to forecast
MIN_HISTORY_DAYS   = int(os.getenv("MIN_HISTORY_DAYS", "30"))  # minimal historical span per group (days)
PARALLEL_JOBS      = int(os.getenv("PARALLEL_JOBS", str(max(1, (os.cpu_count() or 2) // 2))))
# NEW: pick the model to use: "prophet", "ets", or "baseline"
FORECAST_MODEL = os.getenv("FORECAST_MODEL", "ets").lower()
# -----------------------------------------------------------------------------
# Curated / Output datasets
# -----------------------------------------------------------------------------
CURATED_DIR: Path = BASE_DIR / "curated"
CURATED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON: Path      = CURATED_DIR / "dashboard_output.json"
SALES_CURATED: Path    = CURATED_DIR / "sales_curated.parquet"
CATALOG_CURATED: Path  = CURATED_DIR / "catalog_curated.parquet"

# -----------------------------------------------------------------------------
# Common formats / dtype hints (optional but handy)
# -----------------------------------------------------------------------------
DATE_FMT = "%Y-%m-%d"  # ISO day format used across the project

# Columns that should be integer-like across tables
ID_INT_COLUMNS = [
    "Batch_ID", "Product_ID", "Production_Hour", "Sale_Hour", "Transport_Hour"
]

# Columns that should be ISO dates (TEXT in SQLite, parsed to datetime in pandas)
DATE_COLUMNS = [
    "Production_Date", "Sale_Date", "Transport_Date", "Forecast_Date", "date"
]

# -----------------------------------------------------------------------------
# SQLite connection helper
# -----------------------------------------------------------------------------
def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """
    Open a SQLite connection with recommended PRAGMAs for reliability/performance.

    - foreign_keys=ON      : enforce referential integrity
    - journal_mode=WAL     : better concurrency and durability
    - synchronous=NORMAL   : balanced durability/performance

    Always close with a context manager:
        with get_connection() as conn:
            ...
    """
    path = str(db_path or DATABASE_FILE)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    return conn

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def ensure_iso_date(value) -> str:
    """
    Cast a date-like value to an ISO day string (YYYY-MM-DD).
    """
    return pd.to_datetime(value, errors="coerce").strftime(DATE_FMT)

# Friendly default error messages (useful in ETL/forecast logs)
ERR_MIN_HISTORY = f"Insufficient history (< {MIN_HISTORY_DAYS} days) to run forecasting for this group."
ERR_EMPTY_JOIN  = "Join produced no rows — check ID types (int vs bytes) and date granularities."
