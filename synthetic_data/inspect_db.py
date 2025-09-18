import sqlite3
import pandas as pd
from pathlib import Path
import argparse
from textwrap import indent

# ----------------------------------
# Defaults / paths
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "milk_factory.db"

# Tables (keep in sync with your schema)
TABLES = [
    "milk_production_base",
    "milk_quality",
    "milk_inventory",
    "milk_sales",
    "milk_factory",
    "milk_logistics",
    "milk_forecast",  # may not exist yet; ignore errors
]

# Columns to type-check with typeof()
TYPEOF_CHECKS = {
    "milk_production_base": ["Batch_ID", "Product_ID", "Production_Date", "Production_Hour"],
    "milk_quality": ["Batch_ID", "Grade"],
    "milk_inventory": ["Batch_ID", "Product_ID", "Production_Date", "Production_Hour"],
    "milk_sales": ["Batch_ID", "Product_ID", "Store_ID", "Sale_Date", "Sale_Hour"],
    "milk_factory": ["Batch_ID", "Product_ID", "Production_Date", "Production_Hour"],
    "milk_logistics": ["Batch_ID", "Product_ID", "Transport_Date", "Transport_Hour"],
    "milk_forecast": ["Batch_ID", "Product_ID", "Store_ID", "Forecast_Date"],
}

# Primary keys (composite) to check duplicates
PRIMARY_KEYS = {
    "milk_inventory": ["Batch_ID", "Product_ID"],
    "milk_sales": ["Batch_ID", "Product_ID", "Store_ID", "Sale_Date", "Sale_Hour"],
    "milk_factory": ["Batch_ID", "Product_ID"],
    "milk_logistics": ["Batch_ID", "Product_ID", "Transport_Date", "Transport_Hour"],
    "milk_forecast": ["Batch_ID", "Product_ID", "Store_ID", "Forecast_Date"],
}


# ----------------------------------
# Connection helper
# ----------------------------------
def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    return conn


# ----------------------------------
# Utilities
# ----------------------------------
def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_query(conn: sqlite3.Connection, sql: str, params: tuple | None = None) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception as e:
        print(f" Query failed: {e}\n  SQL: {sql[:180]}...")
        return pd.DataFrame()


def list_tables(conn: sqlite3.Connection) -> list[str]:
    df = safe_query(conn, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return df["name"].tolist() if not df.empty else []


def show_schema(conn: sqlite3.Connection, table: str):
    info = safe_query(conn, f"PRAGMA table_info({table})")
    if info.empty:
        print(f"  ⚠️  No schema info for table {table} (may not exist).")
        return
    print(f"\n Schema for {table}:")
    print(indent(info[["cid", "name", "type", "notnull", "dflt_value", "pk"]].to_string(index=False), "  "))

    idx = safe_query(conn, f"PRAGMA index_list({table})")
    if not idx.empty:
        print(f"\n Indexes on {table}:")
        print(indent(idx[["name", "unique", "origin", "partial"]].to_string(index=False), "  "))
        for name in idx["name"]:
            cols = safe_query(conn, f"PRAGMA index_info({name})")
            if not cols.empty:
                cols_s = ", ".join(cols["name"].astype(str).tolist())
                print(f"  - {name}: {cols_s}")


def row_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return int(cur.fetchone()[0])
    except Exception:
        return 0


def typeof_hist(conn: sqlite3.Connection, table: str, columns: list[str]):
    print(f"\n typeof() by column in {table}:")
    for c in columns:
        try:
            df = safe_query(conn, f"SELECT typeof({c}) AS t, COUNT(*) AS n FROM {table} GROUP BY typeof({c})")
            if df.empty:
                print(f"  - {c}: (no rows)")
            else:
                counts = ", ".join([f"{t}:{n}" for t, n in df.itertuples(index=False)])
                print(f"  - {c}: {counts}")
        except Exception as e:
            print(f"  - {c}: error ({e})")


def date_range_check(conn: sqlite3.Connection, table: str, date_col: str):
    print(f"\n Date range check for {table}.{date_col}:")
    df = safe_query(conn, f"SELECT MIN({date_col}) AS min_d, MAX({date_col}) AS max_d FROM {table}")
    if df.empty:
        print("  (no rows)")
        return
    dmin, dmax = df.iloc[0]["min_d"], df.iloc[0]["max_d"]
    print(f"  Range: {dmin} → {dmax}")

    # quick ISO sample validation
    sample = safe_query(conn, f"SELECT {date_col} FROM {table} WHERE {date_col} IS NOT NULL LIMIT 50")
    if not sample.empty:
        non_iso = sample[~sample[date_col].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
        if len(non_iso):
            print(f"    Non-ISO examples:")
            print(indent(non_iso.head(5).to_string(index=False), "    "))


def pk_duplicates(conn: sqlite3.Connection, table: str, cols: list[str], limit: int = 10):
    print(f"\n Duplicate PK check for {table} on ({', '.join(cols)}):")
    cols_s = ", ".join(cols)
    sql = f"""
      SELECT {cols_s}, COUNT(*) AS n
      FROM {table}
      GROUP BY {cols_s}
      HAVING COUNT(*) > 1
      LIMIT {limit}
    """
    df = safe_query(conn, sql)
    if df.empty:
        print("  No duplicates found.")
    else:
        print(indent(df.to_string(index=False), "  "))


def join_health(conn: sqlite3.Connection):
    print_header(" Join health checks (key relationships)")

    # inventory ⨝ sales
    q1 = """
      SELECT COUNT(*) AS n
      FROM milk_inventory i
      JOIN milk_sales s
        ON i.Batch_ID = s.Batch_ID AND i.Product_ID = s.Product_ID
    """
    # inventory ⨝ factory
    q2 = """
      SELECT COUNT(*) AS n
      FROM milk_inventory i
      JOIN milk_factory f
        ON i.Batch_ID = f.Batch_ID AND i.Product_ID = f.Product_ID
    """
    # inventory ⨝ logistics
    q3 = """
      SELECT COUNT(*) AS n
      FROM milk_inventory i
      JOIN milk_logistics l
        ON i.Batch_ID = l.Batch_ID AND i.Product_ID = l.Product_ID
    """
    for label, q in [
        ("inventory ⨝ sales", q1),
        ("inventory ⨝ factory", q2),
        ("inventory ⨝ logistics", q3),
    ]:
        df = safe_query(conn, q)
        n = int(df.iloc[0]["n"]) if not df.empty else 0
        print(f"  {label}: {n:,} rows")

    # orphans
    print("\n Orphan checks:")
    orphans = [
        ("sales without inventory",
         """SELECT COUNT(*) AS n
            FROM milk_sales s
            LEFT JOIN milk_inventory i
              ON i.Batch_ID = s.Batch_ID AND i.Product_ID = s.Product_ID
            WHERE i.Batch_ID IS NULL"""),
        ("factory without inventory",
         """SELECT COUNT(*) AS n
            FROM milk_factory f
            LEFT JOIN milk_inventory i
              ON i.Batch_ID = f.Batch_ID AND i.Product_ID = f.Product_ID
            WHERE i.Batch_ID IS NULL"""),
        ("logistics without inventory",
         """SELECT COUNT(*) AS n
            FROM milk_logistics l
            LEFT JOIN milk_inventory i
              ON i.Batch_ID = l.Batch_ID AND i.Product_ID = l.Product_ID
            WHERE i.Batch_ID IS NULL"""),
    ]
    for label, q in orphans:
        df = safe_query(conn, q)
        n = int(df.iloc[0]["n"]) if not df.empty else 0
        print(f"  {label}: {n:,}")


def head_with_types(conn: sqlite3.Connection, table: str, n: int = 5):
    print(f"\n Sample rows from {table} (n={n}):")
    df = safe_query(conn, f"SELECT * FROM {table} LIMIT {n}")
    if df.empty:
        print("  (no rows)")
        return
    print(indent(df.to_string(index=False), "  "))

    # show typeof() for common ID/date columns
    probe_cols = TYPEOF_CHECKS.get(table, [])
    for c in probe_cols:
        try:
            tdf = safe_query(conn, f"SELECT typeof({c}) AS t FROM {table} LIMIT {n}")
            if not tdf.empty:
                print(f"  typeof({c}): {tdf['t'].tolist()}")
        except Exception:
            pass


# ----------------------------------
# Main inspection routine
# ----------------------------------
def inspect_db(db_path: Path, show_samples: bool = True, sample_size: int = 5):
    print_header(" Database overview")
    print(f"DB path: {db_path}")

    with get_connection(db_path) as conn:
        tables = list_tables(conn)
        if not tables:
            print("No tables found.")
            return

        print("\n Tables in database:")
        for t in tables:
            print(f"  - {t}")

        print_header(" Row counts per table")
        for t in TABLES:
            if t in tables:
                cnt = row_count(conn, t)
                print(f"  {t}: {cnt:,} rows")
            else:
                print(f"  {t}: (missing)")

        print_header(" Schemas & indexes")
        for t in TABLES:
            if t in tables:
                show_schema(conn, t)

        print_header(" typeof() histograms (key columns)")
        for t in TABLES:
            if t in tables:
                typeof_hist(conn, t, TYPEOF_CHECKS.get(t, []))

        print_header(" Date range checks")
        date_checks = [
            ("milk_production_base", "Production_Date"),
            ("milk_inventory", "Production_Date"),
            ("milk_sales", "Sale_Date"),
            ("milk_factory", "Production_Date"),
            ("milk_logistics", "Transport_Date"),
            ("milk_forecast", "Forecast_Date"),
        ]
        for t, c in date_checks:
            if t in tables:
                date_range_check(conn, t, c)

        print_header("❗ Duplicate PK checks")
        for t, cols in PRIMARY_KEYS.items():
            if t in tables:
                pk_duplicates(conn, t, cols)

        print_header(" Relationship health")
        # Only run if the core tables exist
        core = {"milk_inventory", "milk_sales", "milk_factory", "milk_logistics"}
        if core.issubset(set(tables)):
            join_health(conn)
        else:
            print("Core tables missing; skipping join checks.")

        if show_samples:
            print_header(" Data samples")
            for t in TABLES:
                if t in tables:
                    head_with_types(conn, t, n=sample_size)

        print("\n Inspection finished.")


# ----------------------------------
# CLI
# ----------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inspect SQLite DB for schema, types, dates, joins, and duplicates.")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Path to SQLite database file.")
    parser.add_argument("--no-samples", action="store_true", help="Do not print sample rows.")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of sample rows per table.")
    args = parser.parse_args()

    inspect_db(
        db_path=Path(args.db),
        show_samples=not args.no_samples,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
