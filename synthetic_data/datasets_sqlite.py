"""
Initial dataset generator for the Smart Milk Factory project.

This script:
- Creates a fresh SQLite schema (drops existing tables)
- Builds a stable product catalog (Product_ID -> Product_Type) with 1–to–1 mapping
- Generates realistic production/quality, inventory, sales, factory, and logistics
- Ensures plausible ranges for temperatures, energy, volumes, and times

All messages and comments are in English (per request).
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import timedelta

# -----------------------------
# Settings
# -----------------------------
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "milk_factory.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# CSV is inside the synthetic_data folder
CSV_PATH = Path(__file__).resolve().parent / "milknew.csv"

START_DATE = pd.Timestamp("2025-01-01")
HORIZON_DAYS = 180

STORE_IDS = [f"S{i}" for i in range(1, 11)]
WAREHOUSES = ["A", "B", "C", "D"]
LINES = [f"Line_{i}" for i in range(1, 6)]

# Stable product catalog (1 SKU per type)
PRODUCT_TYPES = ["whole milk", "skimmed milk", "yogurt", "cheese", "cream"]
N_PRODUCTS = len(PRODUCT_TYPES)  # 5

def build_product_catalog(n: int = N_PRODUCTS) -> dict[int, str]:
    """
    Build a stable mapping Product_ID -> Product_Type (1–to–1).
    Product_ID 1..n maps deterministically to the PRODUCT_TYPES list.
    """
    if n != len(PRODUCT_TYPES):
        raise ValueError("N_PRODUCTS must equal len(PRODUCT_TYPES) for 1–to–1 mapping.")
    return {i + 1: PRODUCT_TYPES[i] for i in range(n)}

PRODUCT_CATALOG = build_product_catalog()

# Production shifts (realistic probabilities)
SHIFT_DEFS = [
    ("MORNING", 6, 14, 0.45),
    ("AFTERNOON", 14, 22, 0.40),
    ("NIGHT", 22, 6, 0.15),  # crosses midnight
]

GRADE_SHELF_BASE = {"high": 15, "medium": 10, "low": 5}


def choose_production_hour() -> int:
    """
    Pick a production hour based on shift probabilities.
    Returns an integer hour in 0..23.
    """
    _, starts, ends, probs = zip(*SHIFT_DEFS)
    shift_idx = np.random.choice(range(len(SHIFT_DEFS)), p=probs)
    start, end = starts[shift_idx], ends[shift_idx]
    if start < end:
        return int(np.random.randint(start, end))
    # night shift (22–06)
    return int(np.random.choice(list(range(22, 24)) + list(range(0, 6))))


def ensure_iso_date(s) -> str:
    """
    Cast any date-like input to ISO string YYYY-MM-DD.
    """
    return pd.to_datetime(s).strftime("%Y-%m-%d")


# -----------------------------
# Load & sanitize milk quality CSV
# -----------------------------
print("Loading milk quality CSV...")
q = pd.read_csv(CSV_PATH)

# Fix typo and standardize columns
q = q.rename(columns={"Temprature": "Temperature"})
expected = ["pH", "Temperature", "Taste", "Odor", "Fat", "Turbidity", "Colour", "Grade"]
missing = [c for c in expected if c not in q.columns]
if missing:
    raise ValueError(f"CSV missing columns: {missing}")

# Light coercions and clamps for plausible ranges
q["Taste"] = q["Taste"].clip(0, 1).astype(int)
q["Odor"] = q["Odor"].clip(0, 1).astype(int)
q["Fat"] = q["Fat"].clip(0, 1).astype(int)
q["Turbidity"] = q["Turbidity"].clip(0, 1).astype(int)
q["Colour"] = q["Colour"].clip(0, 255).astype(int)
q["Grade"] = q["Grade"].astype(str).str.lower().str.strip()
q["Temperature"] = pd.to_numeric(q["Temperature"], errors="coerce")
q["pH"] = pd.to_numeric(q["pH"], errors="coerce")

q = q.dropna(subset=["pH", "Temperature", "Grade"]).reset_index(drop=True)

# Sample up to 100 rows (or less if the CSV is smaller)
N = min(100, len(q))
q = q.sample(n=N, random_state=42).reset_index(drop=True)


def calc_shelf_life_days(row) -> int:
    """
    Compute shelf life (in days) from Grade, adjusted by Temperature (high temp reduces)
    and pH (low pH reduces). Clipped to [2, 20] days.
    """
    base = GRADE_SHELF_BASE.get(row["Grade"], 7)
    adj = 0
    if row["Temperature"] > 40:
        # each °C above 40 reduces 0.2 day
        adj -= 0.2 * (row["Temperature"] - 40)
    if row["pH"] < 6.8:
        adj -= 0.5 * (6.8 - row["pH"])
    return int(np.clip(round(base + adj), 2, 20))


# -----------------------------
# SQLite connection & schema
# -----------------------------
with sqlite3.connect(DB_PATH) as conn:
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON;")
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    # Fresh base: drop and recreate schema
    cur.executescript("""
    DROP TABLE IF EXISTS milk_logistics;
    DROP TABLE IF EXISTS milk_factory;
    DROP TABLE IF EXISTS milk_sales;
    DROP TABLE IF EXISTS milk_inventory;
    DROP TABLE IF EXISTS milk_quality;
    DROP TABLE IF EXISTS milk_production_base;
    DROP TABLE IF EXISTS product_catalog;
    """)

    cur.executescript("""
    CREATE TABLE product_catalog (
        Product_ID   INTEGER PRIMARY KEY,
        Product_Type TEXT NOT NULL
    );

    CREATE TABLE milk_production_base (
        Batch_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Product_ID INTEGER NOT NULL,
        Product_Type TEXT,
        Production_Date TEXT,         -- ISO YYYY-MM-DD
        Production_Hour INTEGER,      -- 0..23
        Grade TEXT,
        Shelf_Life_Days INTEGER
    );

    CREATE TABLE milk_quality (      -- 1-to-1 with batch
        Batch_ID INTEGER PRIMARY KEY,
        pH REAL NOT NULL,
        Temperature REAL NOT NULL,
        Taste INTEGER NOT NULL CHECK (Taste IN (0,1)),
        Odor INTEGER NOT NULL CHECK (Odor IN (0,1)),
        Fat INTEGER NOT NULL CHECK (Fat IN (0,1)),
        Turbidity INTEGER NOT NULL CHECK (Turbidity IN (0,1)),
        Colour INTEGER NOT NULL CHECK (Colour BETWEEN 0 AND 255),
        Grade TEXT NOT NULL CHECK (Grade IN ('high','medium','low')),
        FOREIGN KEY (Batch_ID) REFERENCES milk_production_base(Batch_ID)
    );

    CREATE TABLE milk_inventory (
        Batch_ID INTEGER,
        Product_ID INTEGER,
        Production_Date TEXT,
        Production_Hour INTEGER,
        Initial_Stock_Liters INTEGER,
        Current_Stock_Liters INTEGER,
        Reorder_Level_Liters INTEGER,
        Reorder_Quantity_Liters INTEGER,
        Warehouse_Location TEXT,
        Storage_Temperature_C REAL,
        PRIMARY KEY (Batch_ID, Product_ID),
        FOREIGN KEY (Batch_ID) REFERENCES milk_production_base(Batch_ID)
    );

    CREATE TABLE milk_sales (
        Batch_ID INTEGER,
        Product_ID INTEGER,
        Store_ID TEXT,
        Sale_Date TEXT,               -- ISO day
        Sale_Hour INTEGER,
        Units_Sold INTEGER,
        Promotion_Flag INTEGER,
        Seasonality_Factor REAL,
        PRIMARY KEY (Batch_ID, Product_ID, Store_ID, Sale_Date, Sale_Hour),
        FOREIGN KEY (Batch_ID) REFERENCES milk_production_base(Batch_ID)
    );

    CREATE TABLE milk_factory (
        Batch_ID INTEGER,
        Product_ID INTEGER,
        Production_Date TEXT,
        Production_Hour INTEGER,
        Line_ID TEXT,
        Quantity_Produced_Liters INTEGER,
        Lead_Time_Days INTEGER,
        Max_Capacity_Per_Day_Liters INTEGER,
        Energy_Consumption_kWh REAL,
        PRIMARY KEY (Batch_ID, Product_ID),
        FOREIGN KEY (Batch_ID) REFERENCES milk_production_base(Batch_ID)
    );

    CREATE TABLE milk_logistics (
        Batch_ID INTEGER,
        Product_ID INTEGER,
        Transport_Date TEXT,
        Transport_Hour INTEGER,
        Transport_Time_Hours REAL,
        Transport_Temperature_C REAL,
        Energy_Consumption_kWh REAL,
        Warehouse_Location TEXT,
        PRIMARY KEY (Batch_ID, Product_ID, Transport_Date, Transport_Hour),
        FOREIGN KEY (Batch_ID) REFERENCES milk_production_base(Batch_ID)
    );

    CREATE INDEX IF NOT EXISTS idx_sales_date ON milk_sales(Sale_Date);
    CREATE INDEX IF NOT EXISTS idx_log_date ON milk_logistics(Transport_Date);
    """)

    # Populate product catalog once (deterministic 1–to–1)
    print("Creating product catalog (stable SKU map 1–to–1)...")
    cat_rows = [{"Product_ID": pid, "Product_Type": ptype} for pid, ptype in PRODUCT_CATALOG.items()]
    pd.DataFrame(cat_rows).to_sql("product_catalog", conn, if_exists="append", index=False)

    # -----------------------------
    # 1) Production + Quality
    # -----------------------------
    print("Generating Production + Quality...")

    prod_records = []
    for i in range(N):
        prod_date = START_DATE + pd.to_timedelta(np.random.randint(0, HORIZON_DAYS), unit="D")
        prod_hour = choose_production_hour()
        qrow = q.iloc[i]
        grade = qrow["Grade"]
        shelf = calc_shelf_life_days(qrow)

        # Use stable product catalog (1–to–1)
        pid = int(np.random.randint(1, N_PRODUCTS + 1))
        ptype = PRODUCT_CATALOG[pid]

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

    # one-to-one quality
    qual_rows = []
    for i, (_, brow) in enumerate(df_batches.iterrows()):
        qrow = q.iloc[i % len(q)]
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
    df_quality.to_sql("milk_quality", conn, if_exists="append", index=False)

    # -----------------------------
    # 2) Inventory
    # -----------------------------
    print("Generating Inventory...")
    inv = df_batches[["Batch_ID", "Product_ID", "Production_Date", "Production_Hour", "Product_Type"]].copy()

    def init_stock(ptype: str) -> int:
        if ptype in ("whole milk", "skimmed milk"):
            return int(np.random.normal(3000, 700))  # liters per batch/day
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

    # -----------------------------
    # 3) Sales / Demand
    # -----------------------------
    print("Generating Sales...")
    sales = []
    for _, row in df_batches.iterrows():
        batch_id = int(row["Batch_ID"])
        product_id = int(row["Product_ID"])
        prod_date = pd.to_datetime(row["Production_Date"])
        shelf_days = int(row["Shelf_Life_Days"])
        expiry_date = prod_date + timedelta(days=shelf_days)

        # current stock from inventory
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
            weekday = day.weekday()  # 0=Mon
            daily_factor = 1.1 if weekday in (4, 5) else 1.0  # Fri/Sat sell more
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

    # -----------------------------
    # 4) Factory
    # -----------------------------
    print("Generating Factory...")
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

    # -----------------------------
    # 5) Logistics
    # -----------------------------
    print("Generating Logistics...")
    logistics = []
    for _, row in df_batches.iterrows():
        batch_id = int(row["Batch_ID"])
        product_id = int(row["Product_ID"])
        prod_date = pd.to_datetime(row["Production_Date"])
        days = int(np.random.randint(3, 7))
        for d in range(days):
            # Commercial transport hours 8–20
            t_hour = int(np.random.choice(list(range(8, 21))))
            t_time = float(np.random.uniform(1, 8))  # transit hours
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

print("\nOK: Initial SQLite datasets generated with 1–to–1 product catalog, realistic hours, shelf-life, stock & energy correlations.")
