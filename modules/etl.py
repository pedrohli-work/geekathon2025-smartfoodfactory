import sqlite3
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from utils.logger import logger
from config import (
    TABLE_PRODUCTION,
    TABLE_INVENTORY,
    TABLE_SALES,
    TABLE_FACTORY,
    TABLE_LOGISTICS,
    CATALOG_CURATED,
    SALES_CURATED,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + underscore columns."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def coerce_id_int(series: pd.Series) -> pd.Series:
    """
    Ensure ID columns are integer-like. Handles bytes -> int if needed.
    """
    def _coerce(x):
        if isinstance(x, (bytes, bytearray)):
            # interpret as little-endian integer (most common in SQLite oddities)
            try:
                return int.from_bytes(x, "little", signed=False)
            except Exception:
                try:
                    return int(x.decode("utf-8"))
                except Exception:
                    return pd.NA
        try:
            return int(x)
        except Exception:
            return pd.NA

    out = series.apply(_coerce)
    return out.astype("Int64")


def parse_iso_date(series: pd.Series) -> pd.Series:
    """Parse dates and keep only date component."""
    s = pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(s.dt.date)


def load_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    """Read table safely (empty df on error), normalized cols."""
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception as e:
        logger.exception("Failed to read table %s: %s", table, e)
        return pd.DataFrame()
    return normalize_column_names(df)


def map_sales_columns(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the sales table robust to legacy/variant column names.
    Ensures presence of: batch_id, product_id, store_id, date, units_sold, sale_hour, promotion_flag, seasonality_factor
    """
    df = sales_df.copy()

    # Map possible legacy names to our canonical names
    rename_map = {}
    # date
    if "sale_date" in df.columns:
        rename_map["sale_date"] = "date"
    elif "ds" in df.columns:
        rename_map["ds"] = "date"
    # units
    if "units" in df.columns:
        rename_map["units"] = "units_sold"
    elif "y" in df.columns:
        rename_map["y"] = "units_sold"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = {"batch_id", "product_id", "store_id", "date", "units_sold"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("Sales table missing key columns: %s. Found: %s", missing, df.columns.tolist())
        return pd.DataFrame()

    # Types & parsing
    df["batch_id"] = coerce_id_int(df["batch_id"])
    df["product_id"] = coerce_id_int(df["product_id"])
    df["store_id"] = df["store_id"].astype(str)
    df["date"] = parse_iso_date(df["date"])
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0).astype(int)

    # Optional cols
    for c, default in [
        ("sale_hour", 12),
        ("promotion_flag", 0),
        ("seasonality_factor", 1.0),
    ]:
        if c not in df.columns:
            df[c] = default

    return df.dropna(subset=["batch_id", "product_id", "date"])


def aggregate_sales_daily(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group hourly sales to daily per (batch_id, product_id, store_id, date).
    """
    if sales_df.empty:
        return pd.DataFrame()
    grp = (
        sales_df.groupby(["batch_id", "product_id", "store_id", "date"], as_index=False)
        .agg(
            units_sold=("units_sold", "sum"),
            promotion_flag=("promotion_flag", "max"),
            seasonality_factor=("seasonality_factor", "mean"),
        )
    )
    return grp


def map_inventory_columns(inv_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce inventory dtypes & dates."""
    if inv_df.empty:
        return inv_df
    df = inv_df.copy()
    for c in ["batch_id", "product_id", "production_hour"]:
        if c in df.columns:
            df[c] = coerce_id_int(df[c])
    if "production_date" in df.columns:
        df["production_date"] = parse_iso_date(df["production_date"])
    for c in [
        "initial_stock_liters",
        "current_stock_liters",
        "reorder_level_liters",
        "reorder_quantity_liters",
        "storage_temperature_c",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "warehouse_location" in df.columns:
        df["warehouse_location"] = df["warehouse_location"].astype(str)
    return df


def map_production_columns(prod_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce production dtypes & dates."""
    if prod_df.empty:
        return prod_df
    df = prod_df.copy()
    for c in ["batch_id", "product_id", "production_hour", "shelf_life_days"]:
        if c in df.columns:
            if c in ["batch_id", "product_id", "production_hour"]:
                df[c] = coerce_id_int(df[c])
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    if "production_date" in df.columns:
        df["production_date"] = parse_iso_date(df["production_date"])
    if "product_type" in df.columns:
        df["product_type"] = df["product_type"].astype(str)
    if "grade" in df.columns:
        df["grade"] = df["grade"].astype(str)
    return df


def map_factory_columns(f_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce factory dtypes."""
    if f_df.empty:
        return f_df
    df = f_df.copy()
    for c in ["batch_id", "product_id", "production_hour", "lead_time_days", "max_capacity_per_day_liters"]:
        if c in df.columns:
            if c in ["batch_id", "product_id", "production_hour"]:
                df[c] = coerce_id_int(df[c])
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["quantity_produced_liters", "energy_consumption_kwh"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "production_date" in df.columns:
        df["production_date"] = parse_iso_date(df["production_date"])
    if "line_id" in df.columns:
        df["line_id"] = df["line_id"].astype(str)
    return df


def map_logistics_columns(l_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce logistics dtypes; prepare daily aggregation."""
    if l_df.empty:
        return l_df
    df = l_df.copy()
    for c in ["batch_id", "product_id", "transport_hour"]:
        if c in df.columns:
            df[c] = coerce_id_int(df[c])
    if "transport_date" in df.columns:
        df["date"] = parse_iso_date(df["transport_date"])
    else:
        # legacy fallback
        if "date" in df.columns:
            df["date"] = parse_iso_date(df["date"])
    for c in ["transport_time_hours", "transport_temperature_c", "energy_consumption_kwh"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # daily aggregate per batch/product/date (summing energy/time, avg temperature)
    if {"batch_id", "product_id", "date"}.issubset(df.columns):
        agg = (
            df.groupby(["batch_id", "product_id", "date"], as_index=False)
            .agg(
                transport_time_hours=("transport_time_hours", "sum"),
                transport_temperature_c=("transport_temperature_c", "mean"),
                energy_consumption_kwh=("energy_consumption_kwh", "sum"),
            )
        )
        return agg
    return pd.DataFrame()


# ---------------------------------------------------------------------
# Main ETL
# ---------------------------------------------------------------------
def build_integrated_pipeline(conn: sqlite3.Connection) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build curated catalog and sales datasets:
      - Load & normalize all tables
      - Robust mapping of variants/legacy column names
      - Daily sales aggregation
      - Merge with inventory, factory, logistics
      - Derive: available_stock, days_to_expiration, suggested_production, expiration_risk
      - Save to Parquet
    Returns:
      catalog_df, sales_curated_df
    """
    # Load
    production_df = load_table(conn, TABLE_PRODUCTION)
    inventory_df = load_table(conn, TABLE_INVENTORY)
    sales_df = load_table(conn, TABLE_SALES)
    factory_df = load_table(conn, TABLE_FACTORY)
    logistics_df = load_table(conn, TABLE_LOGISTICS)

    logger.info(
        "Loaded tables counts: production=%d, inventory=%d, sales=%d, factory=%d, logistics=%d",
        len(production_df), len(inventory_df), len(sales_df), len(factory_df), len(logistics_df)
    )

    # Map/Coerce
    production_df = map_production_columns(production_df)
    inventory_df = map_inventory_columns(inventory_df)
    sales_df = map_sales_columns(sales_df)
    logistics_daily = map_logistics_columns(logistics_df)
    factory_df = map_factory_columns(factory_df)

    # Catalog (distinct per batch/product)
    catalog_cols = ["batch_id", "product_id", "product_type", "shelf_life_days", "production_date"]
    catalog_df = (
        production_df[catalog_cols].dropna(how="all").drop_duplicates()
        if set(catalog_cols).issubset(production_df.columns)
        else pd.DataFrame(columns=catalog_cols)
    )
    CATALOG_CURATED.parent.mkdir(parents=True, exist_ok=True)
    catalog_df.to_parquet(CATALOG_CURATED, index=False)
    logger.info("Saved catalog curated to %s (%d rows)", CATALOG_CURATED, len(catalog_df))

    # Sales daily
    if sales_df.empty:
        logger.warning("Sales table missing key columns; sales_curation will be empty.")
        return catalog_df, pd.DataFrame()

    sales_daily = aggregate_sales_daily(sales_df)
    if sales_daily.empty:
        logger.warning("No sales_daily data to build curated sales. Returning empty curated sales.")
        return catalog_df, pd.DataFrame()

    # Merge chain
    cur = sales_daily.merge(
        catalog_df, on=["batch_id", "product_id"], how="left"
    )

    if not inventory_df.empty:
        cur = cur.merge(
            inventory_df[
                ["batch_id", "product_id", "initial_stock_liters", "current_stock_liters",
                 "reorder_level_liters", "reorder_quantity_liters", "warehouse_location",
                 "storage_temperature_c", "production_date", "production_hour"]
            ],
            on=["batch_id", "product_id"],
            how="left",
            suffixes=("", "_inv")
        )

    if not factory_df.empty:
        cur = cur.merge(
            factory_df[
                ["batch_id", "product_id", "line_id", "quantity_produced_liters",
                 "lead_time_days", "max_capacity_per_day_liters", "energy_consumption_kwh"]
            ],
            on=["batch_id", "product_id"],
            how="left",
            suffixes=("", "_fac")
        )

    if not logistics_daily.empty:
        cur = cur.merge(
            logistics_daily,
            on=["batch_id", "product_id", "date"],
            how="left",
            suffixes=("", "_log")
        )

    # Derivations
    for c in [
        "current_stock_liters", "units_sold", "max_capacity_per_day_liters",
        "shelf_life_days", "energy_consumption_kwh"
    ]:
        if c in cur.columns:
            cur[c] = pd.to_numeric(cur[c], errors="coerce")

    # Ensure dates
    if "production_date" in cur.columns:
        cur["production_date"] = parse_iso_date(cur["production_date"])
    if "date" in cur.columns:
        cur["date"] = parse_iso_date(cur["date"])

    # available stock (simple daily view)
    cur["available_stock"] = (cur.get("current_stock_liters", 0) - cur.get("units_sold", 0)).clip(lower=0)

    # days to expiration
    cur["days_to_expiration"] = (
        (cur["production_date"] + pd.to_timedelta(cur["shelf_life_days"].fillna(0), unit="D")) - cur["date"]
    ).dt.days.clip(lower=0)

    # suggested production per day
    demand_gap = (cur["units_sold"] - cur["available_stock"]).clip(lower=0)
    cur["suggested_production"] = np.where(
        cur["max_capacity_per_day_liters"].notna(),
        np.minimum(demand_gap, cur["max_capacity_per_day_liters"]),
        demand_gap
    )

    # expiration risk
    cur["expiration_risk"] = ((cur["days_to_expiration"] <= 3) & (cur["available_stock"] > 0)).astype(int)

    # tidy columns order
    preferred = [
        "batch_id", "product_id", "store_id", "date",
        "product_type", "production_date", "shelf_life_days",
        "units_sold", "available_stock", "days_to_expiration",
        "suggested_production", "expiration_risk",
        "max_capacity_per_day_liters", "energy_consumption_kwh",
        "transport_time_hours", "transport_temperature_c",
        "warehouse_location", "storage_temperature_c",
        "line_id", "lead_time_days", "quantity_produced_liters",
        "reorder_level_liters", "reorder_quantity_liters"
    ]
    cols = [c for c in preferred if c in cur.columns] + [c for c in cur.columns if c not in preferred]
    cur = cur[cols]

    # Save curated sales
    SALES_CURATED.parent.mkdir(parents=True, exist_ok=True)
    cur.to_parquet(SALES_CURATED, index=False)
    logger.info("Saved sales curated to %s (%d rows)", SALES_CURATED, len(cur))

    return catalog_df, cur
