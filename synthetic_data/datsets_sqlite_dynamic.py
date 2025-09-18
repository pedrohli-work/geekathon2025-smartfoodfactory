"""
Incremental dataset generator for Smart Milk Factory.

This script:
- Appends new rows (does not drop schema)
- Uses the stable product_catalog table for Product_ID -> Product_Type (1–to–1)
- Generates new production/quality, inventory, sales, factory, logistics
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import timedelta

# ----------------------------------
# Settings
# ----------------------------------
np.random.seed(123)  # reproducibility for incremental runs

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "milk_factory.db"

# Reuse the same CSV as quality source to map new batches 1–to–1
CSV_PATH = Path(__file__).resolve().parent / "milknew.csv"

STORE_IDS = [f"S{i}" for i in range(1, 11)]
WAREHOUSES = ["A", "B", "C", "D"]
LINES = [f"Line_{i}" for i in range(1, 6)]

# Optional seed for an empty DB (should match initial script)
PRODUCT_TYPES = ["whole milk", "skimmed milk", "yogurt", "cheese", "cream"]
N_PRODUCTS = len(PRODUCT_TYPES)

def build_product_catalog(n: int = N_PRODUCTS) -> dict[int, str]:
    if n != len(PRODUCT_TYPES):
        raise ValueError("N_PRODUCTS must equal len(PRODUCT_TYPES) for 1–to–1 mapping.")
    return {i + 1: PRODUCT_TYPES[i] for i in range(n)}

SHIFT_DEFS = [
    ("MORNING", 6, 14, 0.45),
    ("AFTERNOON", 14, 22, 0.40),
    ("NIGHT", 22, 6, 0.15),  # crosses midnight
]

GRADE_SHELF_BASE = {"high": 15, "medium": 10, "low": 5}


def choose_production_hour() -> int:
    """Pick a production hour based on shift probabilities (returns 0..23)."""
    _, starts, ends, probs = zip(*SHIFT_DEFS)
    shift_idx = np.random.choice(range(len(SHIFT_DEFS)), p=probs)
    start, end = starts[shift_idx], ends[shift_idx]
    if start < end:
        return int(np.random.randint(start, end))
    # night shift (22–06)
    return int(np.random.choice(list(range(22, 24)) + list(range(0, 6))))


def ensure_iso_date(s) -> str:
    """Cast any date-like input to ISO string YYYY-MM-DD."""
    return pd.to_datetime(s).strftime("%Y-%m-%d")


def calc_shelf_life_days(row) -> int:
    """Compute shelf life (days) from Grade, adjusted by Temperature and pH. Clipped to [2, 20]."""
    base = GRADE_SHELF_BASE.get(str(row["Grade"]).lower(), 7)
    adj = 0
    if float(row["Temperature"]) > 40:
        adj -= 0.2 * (float(row["Temperature"]) - 40)
    if float(row["pH"]) < 6.8:
        adj -= 0.5 * (6.8 - float(row["pH"]))
    return int(np.clip(round(base + adj), 2, 20))


def load_quality_source() -> pd.DataFrame:
    """Load and sanitize milk quality CSV to drive new batches."""
    q = pd.read_csv(CSV_PATH)
    q = q.rename(columns={"Temprature": "Temperature"})
    expected = ["pH", "Temperature", "Taste", "Odor", "Fat", "Turbidity", "Colour", "Grade"]
    missing = [c for c in expected if c not in q.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Light coercions
    q["Taste"] = q["Taste"].clip(0, 1).astype(int)
    q["Odor"] = q["Odor"].clip(0, 1).astype(int)
    q["Fat"] = q["Fat"].clip(0, 1).astype(int)
    q["Turbidity"] = q["Turbidity"].clip(0, 1).astype(int)
    q["Colour"] = q["Colour"].clip(0, 255).astype(int)
    q["Grade"] = q["Grade"].astype(str).str.lower().str.strip()
    q["Temperature"] = pd.to_numeric(q["Temperature"], errors="coerce")
    q["pH"] = pd.to_numeric(q["pH"], errors="coerce")

    q = q.dropna(subset=["pH", "Temperature", "Grade"]).reset_index(drop=True)
    q = q.sample(frac=1.0, random_state=123).reset_index(drop=True)  # deterministic shuffle
    return q


def get_last_production_date(conn: sqlite3.Connection) -> pd.Timestamp:
    """Get last production date from DB. Defaults to 2025-01-01 if empty."""
    row = conn.execute("""
        SELECT MAX(Production_Date)
        FROM (SELECT strftime('%Y-%m-%d', Production_Date) AS Production_Date FROM milk_production_base)
    """).fetchone()
    return pd.to_datetime(row[0]) if row and row[0] else pd.Timestamp("2025-01-01")


def ensure_catalog(conn: sqlite3.Connection) -> dict[int, str]:
    """
    Ensure product_catalog exists and is populated.
    Returns a dict Product_ID -> Product_Type (1–to–1).
    """
    try:
        existing = pd.read_sql_query("SELECT Product_ID, Product_Type FROM product_catalog", conn)
        if not existing.empty:
            return dict(zip(existing["Product_ID"].astype(int), existing["Product_Type"].astype(str)))
    except Exception:
        pass

    # Create and seed if missing (align with initial script)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS product_catalog (
            Product_ID   INTEGER PRIMARY KEY,
            Product_Type TEXT NOT NULL
        )
    """)
    m = build_product_catalog(N_PRODUCTS)
    pd.DataFrame([{"Product_ID": k, "Product_Type": v} for k, v in m.items()]).to_sql(
        "product_catalog", conn, if_exists="append", index=False
    )
    return m


def generate_new_data(days: int = 7, n_batches: int = 30) -> None:
    """
    Incrementally generate new synthetic data for a dairy factory:
      1) Production batches
      2) Milk quality (1:1)
      3) Inventory
      4) Sales (respect stock & shelf-life)
      5) Factory (energy ~ quantity)
      6) Logistics (energy ~ transport time)

    Appends rows; does not drop schema. Uses stable product_catalog (1–to–1).
    """
    quality_src = load_quality_source()

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")

        # Ensure catalog
        cat_map = ensure_catalog(conn)
        cat_ids = list(cat_map.keys())

        last_date = get_last_production_date(conn)
        print(f"Last recorded production date: {last_date.date()}")

        # 1) Production
        print("Inserting new production batches...")
        prod_records = []
        for i in range(n_batches):
            prod_date = last_date + pd.to_timedelta(np.random.randint(1, days + 1), unit="D")
            prod_hour = choose_production_hour()
            qrow = quality_src.iloc[i % len(quality_src)]
            grade = qrow["Grade"]
            shelf = calc_shelf_life_days(qrow)

            # Choose PID from stable catalog (no new IDs)
            pid = int(np.random.choice(cat_ids))
            ptype = cat_map[pid]

            prod_records.append({
                "Product_ID": pid,
                "Product_Type": ptype,
                "Production_Date": ensure_iso_date(prod_date),
                "Production_Hour": int(prod_hour),
                "Grade": grade,
                "Shelf_Life_Days": int(shelf),
            })

        df_prod = pd.DataFrame(prod_records)
        before = cur.execute("SELECT COALESCE(MAX(Batch_ID), 0) FROM milk_production_base").fetchone()[0] or 0
        df_prod.to_sql("milk_production_base", conn, if_exists="append", index=False)
        after = cur.execute("SELECT MAX(Batch_ID) FROM milk_production_base").fetchone()[0]

        df_batches = pd.read_sql_query("""
            SELECT Batch_ID, Product_ID, Product_Type, Production_Date, Production_Hour, Grade, Shelf_Life_Days
            FROM milk_production_base
            WHERE Batch_ID BETWEEN ? AND ?
            ORDER BY Batch_ID
        """, conn, params=(before + 1, after))
        print(f"Inserted {len(df_batches)} rows into milk_production_base")

        # 2) Milk quality
        print("Inserting milk quality rows...")
        qual_rows = []
        for i, (_, brow) in enumerate(df_batches.iterrows()):
            qrow = quality_src.iloc[i % len(quality_src)]
            qual_rows.append({
                "Batch_ID": int(brow["Batch_ID"]),
                "pH": float(qrow["pH"]),
                "Temperature": float(qrow["Temperature"]),
                "Taste": int(qrow["Taste"]),
                "Odor": int(qrow["Odor"]),
                "Fat": int(qrow["Fat"]),
                "Turbidity": int(qrow["Turbidity"]),
                "Colour": int(qrow["Colour"]),
                "Grade": str(qrow["Grade"]),
            })
        df_quality = pd.DataFrame(qual_rows)
        if not df_quality.empty:
            df_quality.to_sql("milk_quality", conn, if_exists="append", index=False)
        print(f"Inserted {len(df_quality)} rows into milk_quality")

        # 3) Inventory
        print("Inserting inventory rows...")
        inv = df_batches[["Batch_ID", "Product_ID", "Production_Date", "Production_Hour", "Product_Type"]].copy()

        def init_stock(ptype: str) -> int:
            if ptype in ("whole milk", "skimmed milk"):
                return int(np.random.normal(3000, 700))
            if ptype in ("yogurt", "cream"):
                return int(np.random.normal(1500, 400))
            if ptype == "cheese":
                return int(np.random.normal(800, 200))
            return int(np.random.randint(500, 5000))

        inv["Initial_Stock_Liters"] = [max(100, init_stock(pt)) for pt in inv["Product_Type"]]
        inv["Current_Stock_Liters"] = (inv["Initial_Stock_Liters"] * np.random.uniform(0.6, 0.95, size=len(inv))).astype(int)
        inv["Reorder_Level_Liters"] = (inv["Initial_Stock_Liters"] * np.random.uniform(0.1, 0.25, size=len(inv))).astype(int)
        inv["Reorder_Quantity_Liters"] = (inv["Initial_Stock_Liters"] * np.random.uniform(0.2, 0.5, size=len(inv))).astype(int)
        inv["Warehouse_Location"] = np.random.choice(WAREHOUSES, size=len(inv))
        inv["Storage_Temperature_C"] = np.random.uniform(2.0, 6.0, size=len(inv))

        inv = inv.drop(columns=["Product_Type"])
        inv.to_sql("milk_inventory", conn, if_exists="append", index=False)
        print(f"Inserted {len(inv)} rows into milk_inventory")

        # 4) Sales
        print("Inserting sales rows...")
        sales = []
        for _, row in df_batches.iterrows():
            batch_id = int(row["Batch_ID"])
            product_id = int(row["Product_ID"])
            prod_date = pd.to_datetime(row["Production_Date"])
            shelf_days = int(row["Shelf_Life_Days"])
            expiry_date = prod_date + timedelta(days=shelf_days)

            rinv = pd.read_sql_query(
                "SELECT Current_Stock_Liters FROM milk_inventory WHERE Batch_ID=? AND Product_ID=?",
                conn, params=(batch_id, product_id)
            )
            remaining = int(rinv.iloc[0, 0]) if not rinv.empty else 0
            if remaining <= 0:
                continue

            days_to_sell = min(np.random.randint(5, 12), shelf_days)
            end_date = min(prod_date + timedelta(days=days_to_sell), expiry_date)

            for day in pd.date_range(prod_date, end_date, freq="D"):
                if remaining <= 0:
                    break
                weekday = day.weekday()
                daily_factor = 1.1 if weekday in (4, 5) else 1.0
                promo = np.random.choice([0, 1], p=[0.85, 0.15])
                promo_boost = 1.25 if promo else 1.0

                alloc = np.random.dirichlet(np.ones(len(STORE_IDS))) * remaining
                for sid, chunk in zip(STORE_IDS, alloc):
                    if remaining <= 0:
                        break
                    base = int(max(0, chunk * daily_factor * promo_boost * np.random.uniform(0.8, 1.2)))
                    sold = min(base, remaining)
                    if sold <= 0:
                        continue
                    hour_pool = ([11, 12, 13] * 3 + [17, 18, 19] * 3 + list(range(8, 21)))
                    sale_hour = int(np.random.choice(hour_pool))
                    sales.append({
                        "Batch_ID": batch_id,
                        "Product_ID": product_id,
                        "Store_ID": sid,
                        "Sale_Date": ensure_iso_date(day),
                        "Sale_Hour": sale_hour,
                        "Units_Sold": sold,
                        "Promotion_Flag": promo,
                        "Seasonality_Factor": float(daily_factor),
                    })
                    remaining -= sold

        sales_df = pd.DataFrame(sales)
        if not sales_df.empty:
            sales_df.to_sql("milk_sales", conn, if_exists="append", index=False)
        print(f"Inserted {len(sales_df)} rows into milk_sales")

        # 5) Factory
        print("Inserting factory rows...")
        prod = df_batches[["Batch_ID", "Product_ID", "Production_Date", "Production_Hour"]].copy()
        prod["Line_ID"] = np.random.choice(LINES, size=len(prod))

        inv_qty = pd.read_sql_query("""
            SELECT Batch_ID, Product_ID, Initial_Stock_Liters
            FROM milk_inventory
        """, conn)
        prod = prod.merge(inv_qty, on=["Batch_ID", "Product_ID"], how="left")
        prod["Quantity_Produced_Liters"] = prod["Initial_Stock_Liters"].fillna(0).astype(int)
        prod["Lead_Time_Days"] = np.random.randint(1, 4, size=len(prod))
        prod["Max_Capacity_Per_Day_Liters"] = (prod["Quantity_Produced_Liters"] * np.random.uniform(1.5, 3.5, size=len(prod))).astype(int)

        qlit = prod["Quantity_Produced_Liters"].astype(float)
        prod["Energy_Consumption_kWh"] = (80 + 0.12 * qlit + np.random.normal(0, 15, size=len(prod)))
        prod["Energy_Consumption_kWh"] = prod["Energy_Consumption_kWh"].clip(lower=50)

        prod = prod.drop(columns=["Initial_Stock_Liters"])
        prod.to_sql("milk_factory", conn, if_exists="append", index=False)
        print(f"Inserted {len(prod)} rows into milk_factory")

        # 6) Logistics
        print("Inserting logistics rows...")
        logistics = []
        for _, row in df_batches.iterrows():
            batch_id = int(row["Batch_ID"])
            product_id = int(row["Product_ID"])
            prod_date = pd.to_datetime(row["Production_Date"])
            days = int(np.random.randint(3, 7))
            for d in range(days):
                t_hour = int(np.random.choice(list(range(8, 21))))
                t_time = float(np.random.uniform(1, 8))
                temp = float(np.random.uniform(2.0, 6.0))
                energy = float(5 + 2.5 * t_time + np.random.normal(0, 3))
                logistics.append({
                    "Batch_ID": batch_id,
                    "Product_ID": product_id,
                    "Transport_Date": ensure_iso_date(prod_date + timedelta(days=d)),
                    "Transport_Hour": t_hour,
                    "Transport_Time_Hours": t_time,
                    "Transport_Temperature_C": temp,
                    "Energy_Consumption_kWh": max(1.0, energy),
                    "Warehouse_Location": np.random.choice(WAREHOUSES)
                })

        log_df = pd.DataFrame(logistics)
        if not log_df.empty:
            log_df.to_sql("milk_logistics", conn, if_exists="append", index=False)
        print(f"Inserted {len(log_df)} rows into milk_logistics")

    print("\nIncremental synthetic data generated successfully.")


if __name__ == "__main__":
    # Default incremental window: 14 days ahead, 60 new batches
    generate_new_data(days=14, n_batches=60)
