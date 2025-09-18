# app.py — Clean keys, reset button, horizon filter, English UI

import json
import re
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    DATABASE_FILE,
    OUTPUT_JSON,
    SALES_CURATED,
    CATALOG_CURATED,
    FORECAST_PERIODS,
    LOG_FILE,  # for log tail
)

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Smart Milk Factory", layout="wide")

st.markdown("# 🥛 Smart Milk Factory")
st.caption("Dashboard for demand, stock, shelf-life risk and sustainability")

st.markdown(
    """
### How to use
1. **Create database** — initialize a fresh, realistic dataset (production, inventory, sales, factory, logistics).  
2. **Run pipeline** — execute ETL + Forecast to produce KPIs and future demand.  
3. **Generate new data** — simulate continuous operations by appending new batches/sales/logistics.  
4. **Run pipeline again** — refresh curated data, KPIs and forecasts with the newly generated data.
"""
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SYNTH_DIR = PROJECT_ROOT / "synthetic_data"
PY = sys.executable  # ensure we call the venv Python

def _tail(path: Path, n: int = 120) -> str:
    """Return the last n lines of a log file (or placeholder)."""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
    except Exception:
        pass
    return "<log not available>"

@st.cache_data(show_spinner=False)
def load_artifacts() -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """Load curated parquet and dashboard JSON if available."""
    catalog_df = pd.DataFrame()
    sales_curated_df = pd.DataFrame()
    forecast_list: List[Dict] = []

    if CATALOG_CURATED.exists():
        catalog_df = pd.read_parquet(CATALOG_CURATED)
    if SALES_CURATED.exists():
        sales_curated_df = pd.read_parquet(SALES_CURATED)
    if OUTPUT_JSON.exists():
        forecast_list = json.loads(Path(OUTPUT_JSON).read_text(encoding="utf-8"))

    return catalog_df, sales_curated_df, forecast_list


def _thousands(x: float | int) -> str:
    """Format number with thousands separator (no decimals)."""
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "0"


def _fmt_float(x: float, decimals: int = 1) -> str:
    """Format float with fixed decimals and thousands separator."""
    try:
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return f"{0:.{decimals}f}"


def _ensure_df_forecast(forecast_list: List[Dict]) -> pd.DataFrame:
    """Ensure forecast JSON → DataFrame, with daily dates."""
    if not forecast_list:
        return pd.DataFrame()
    df = pd.DataFrame(forecast_list)
    if "forecast_date" in df.columns:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce").dt.floor("D")
        df = df.dropna(subset=["forecast_date"])
    return df


def _build_product_map(catalog_df: pd.DataFrame) -> Dict[int, str]:
    """
    Build product_id -> product_type (deduplicated). If a product has multiple types in
    historical data, pick the mode; fall back to the first observed.
    """
    if catalog_df.empty or "product_id" not in catalog_df.columns or "product_type" not in catalog_df.columns:
        return {}

    tmp = (
        catalog_df[["product_id", "product_type"]]
        .dropna(subset=["product_id", "product_type"])
        .copy()
    )
    tmp["product_id"] = tmp["product_id"].astype(int)
    tmp["product_type"] = tmp["product_type"].astype(str)

    def _mode_or_first(s: pd.Series) -> str:
        m = s.mode()
        return str(m.iloc[0]) if not m.empty else str(s.iloc[0])

    by_pid = tmp.groupby("product_id", as_index=True)["product_type"].agg(_mode_or_first)
    return {int(pid): str(pt) for pid, pt in by_pid.to_dict().items()}


def _product_label(pid: Optional[int], product_map: Dict[int, str]) -> str:
    """Human-friendly label for product selectbox and charts."""
    if pid is None:
        return "All"
    name = product_map.get(int(pid), f"Product {int(pid)}")
    return f"{int(pid)} • {name.title()}"


def _store_natural_key(s: str) -> tuple:
    """Natural sort key for store IDs like 'S1'...'S10' (not lexicographic)."""
    s = str(s)
    m = re.match(r"^([A-Za-z]*)(\d+)$", s.strip())
    if m:
        prefix, num = m.groups()
        return (prefix.upper(), int(num))
    return (s.upper(), 0)


def _run_status(label: str, cmd: list[str], success_label: str = "Completed") -> bool:
    """Run a subprocess with a visible status and show stdout + log tail."""
    ok = False
    with st.status(f"{label}...", expanded=True) as status:
        try:
            status.write("Starting…")
            res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
            if res.returncode == 0:
                ok = True
                status.update(label=f"{success_label} ✅", state="complete")
                with st.expander("Details (stdout)"):
                    st.code(res.stdout or "<no stdout>")
                with st.expander("Run Log (tail)"):
                    st.code(_tail(Path(LOG_FILE), 160))
            else:
                status.update(label=f"Failed ❌ (exit {res.returncode})", state="error")
                st.error("See diagnostic output below.")
                with st.expander("Details (stdout/stderr)"):
                    st.code(res.stdout or "<no stdout>")
                    st.code(res.stderr or "<no stderr>")
                with st.expander("Run Log (tail)"):
                    st.code(_tail(Path(LOG_FILE), 160))
        except Exception as e:
            status.update(label="Failed ❌", state="error")
            st.exception(e)
    return ok


def _run_pipeline_with_status() -> bool:
    """Call main.py with the same Python executable used by the app."""
    return _run_status("Running ETL + Forecast", [PY, "main.py"], success_label="Data ready")


def _restrict_horizon(df: pd.DataFrame, date_col: str, days: int) -> pd.DataFrame:
    """Return rows within the first `days` from the minimum date in `date_col`."""
    if df.empty or date_col not in df.columns or days is None:
        return df
    dmin = pd.to_datetime(df[date_col]).min()
    dend = dmin + pd.Timedelta(days=int(days) - 1)
    return df[(pd.to_datetime(df[date_col]) >= dmin) & (pd.to_datetime(df[date_col]) <= dend)].copy()

# -----------------------------------------------------------------------------
# Controls (no sidebar) — unique keys + single reset block
# -----------------------------------------------------------------------------
KEY_CREATE = "ctrl_create_db"
KEY_RUN    = "ctrl_run_pipeline"
KEY_NEW    = "ctrl_new_data"
KEY_RESET  = "ctrl_reset_dashboard_v2"  # unique key

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    create_clicked = st.button(
        "🧱 Create database",
        help="Builds a fresh realistic SQLite with all base tables.",
        key=KEY_CREATE,
        width="stretch",
    )
with c2:
    run_clicked = st.button(
        "📈 Run pipeline",
        help="Runs ETL + Forecast. Tip: after generating new data, run the pipeline again.",
        key=KEY_RUN,
        width="stretch",
    )
with c3:
    new_clicked = st.button(
        "➕ Generate new data",
        help="Appends new batches/sales/logistics to simulate continuous operations.",
        key=KEY_NEW,
        width="stretch",
    )

reset_col, _ = st.columns([1, 3])
if reset_col.button(
    "🔄 Reset dashboard (clear data & cache)",
    help="Deletes curated files and the local database, then clears cache.",
    key=KEY_RESET,
    type="secondary",
    width="content",
):
    # (1) remove entire curated/ directory
    try:
        CURATED_DIR = OUTPUT_JSON.parent  # e.g., Path('curated')
        if CURATED_DIR.exists():
            shutil.rmtree(CURATED_DIR)
    except Exception:
        pass

    # (2) delete DB and (optional) log file
    try:
        DATABASE_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        Path(LOG_FILE).unlink(missing_ok=True)
    except Exception:
        pass

    # (3) clear cached loaders + stage flags and rerun
    try:
        load_artifacts.clear()
    except Exception:
        pass
    for k in ("stage_db", "stage_new", "stage_pipe"):
        st.session_state.pop(k, None)

    st.success("Dashboard reset. Click **Create database** and then **Run pipeline** to rebuild.")
    st.rerun()

# Initialize stage flags (once)
for k in ("stage_db", "stage_new", "stage_pipe"):
    if k not in st.session_state:
        st.session_state[k] = False

# Actions
if create_clicked:
    ok = _run_status("Creating database", [PY, str(SYNTH_DIR / "datasets_sqlite.py")], success_label="Database created")
    if ok:
        st.session_state["stage_db"] = True
        load_artifacts.clear()

if new_clicked:
    dyn_path = SYNTH_DIR / "datsets_sqlite_dynamic.py"  # your file named with 'datsets'
    ok = _run_status("Generating new data", [PY, str(dyn_path)], success_label="New data generated")
    if ok:
        st.session_state["stage_new"] = True
        load_artifacts.clear()

if run_clicked:
    ok = _run_pipeline_with_status()
    if ok:
        st.session_state["stage_pipe"] = True
        load_artifacts.clear()

# Load artifacts
catalog_df, sales_curated_df, forecast_list = load_artifacts()
forecast_df = _ensure_df_forecast(forecast_list)

# Stage messages
if st.session_state.get("stage_db"):
    st.success("Database created — base tables are ready.")
if st.session_state.get("stage_new"):
    st.success("New data generated — incremental rows appended.")
if st.session_state.get("stage_pipe"):
    st.success("Data ready — dashboards reflect the latest pipeline run.")
if OUTPUT_JSON.exists() and forecast_df.empty:
    st.info("A dashboard file exists but looks empty. Try **Run pipeline** again after creating data.")

# Guardrails
if catalog_df.empty and sales_curated_df.empty:
    st.info("No data yet. Click **Create database** to initialize everything.")
    st.stop()

# -----------------------------------------------------------------------------
# Filters + Horizon
# -----------------------------------------------------------------------------
st.markdown("### Filters")
st.caption("Pick a product and store to focus the views. Choose a horizon to summarize the forecast window.")

product_map = _build_product_map(catalog_df)
product_ids = [None] + sorted(product_map.keys())

store_ids = [None]
if not sales_curated_df.empty and "store_id" in sales_curated_df.columns:
    unique_stores = sorted({str(s) for s in sales_curated_df["store_id"].dropna().tolist()}, key=_store_natural_key)
    store_ids += unique_stores

HORIZON_OPTIONS = {1: "1 day", 7: "7 days", 14: "14 days", 30: "30 days"}
default_h = 14 if 14 in HORIZON_OPTIONS else list(HORIZON_OPTIONS.keys())[0]

colp, cols, colh = st.columns([2, 2, 1])
selected_pid = colp.selectbox(
    "Product",
    options=product_ids,
    index=0,
    format_func=lambda pid: _product_label(pid, product_map),
)
selected_store = cols.selectbox(
    "Store",
    options=store_ids,
    index=0,
    format_func=lambda s: "All" if s is None else str(s),
)
horizon_days = colh.selectbox(
    "Horizon",
    options=list(HORIZON_OPTIONS.keys()),
    index=list(HORIZON_OPTIONS.keys()).index(default_h),
    format_func=lambda k: HORIZON_OPTIONS[k],
    help="Aggregate KPIs and charts over the next N forecast days.",
    key="h_sel",
)

# Apply filters
fc = forecast_df.copy()
sc = sales_curated_df.copy()

if selected_pid is not None:
    if "product_id" in fc.columns:
        fc = fc[fc["product_id"] == int(selected_pid)]
    if "product_id" in sc.columns:
        sc = sc[sc["product_id"] == int(selected_pid)]

if selected_store is not None:
    if "store_id" in fc.columns:
        fc = fc[fc["store_id"].astype(str) == str(selected_store)]
    if "store_id" in sc.columns:
        sc = sc[sc["store_id"].astype(str) == str(selected_store)]

# Restrict forecast to the chosen horizon (from its earliest forecast date)
fcw = _restrict_horizon(fc, "forecast_date", horizon_days)

# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------
st.markdown("### Overview")
st.caption("Forecasted demand, expected waste, items near expiration, and current stock.")

total_forecast_units = float(fcw["forecasted_sales"].sum()) if not fcw.empty else 0.0
total_predicted_waste = float(fcw["predicted_waste"].sum()) if not fcw.empty else 0.0
near_expiration = int((sc["days_to_expiration"] <= 3).sum()) if not sc.empty and "days_to_expiration" in sc.columns else 0
current_stock_l = float(sc["available_stock"].sum()) if not sc.empty and "available_stock" in sc.columns else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Forecasted Units (next {horizon_days} days)", _thousands(total_forecast_units))
k2.metric("Predicted Waste (units)", _thousands(total_predicted_waste))
k3.metric("Near Expiration (≤ 3 days)", _thousands(near_expiration))
k4.metric("Current Available Stock (L)", _thousands(current_stock_l))

# -----------------------------------------------------------------------------
# Demand vs. Projected Stock (respects horizon)
# -----------------------------------------------------------------------------
st.markdown("### Demand vs. Projected Stock")
st.caption("Projected Stock (L) = last known stock − cumulative forecast (clipped at 0).")
if not fcw.empty:
    # Forecast (future) within horizon
    fc_daily = fcw.groupby("forecast_date", as_index=False)["forecasted_sales"].sum()
    fc_daily.rename(columns={"forecast_date": "Date", "forecasted_sales": "Forecasted Units"}, inplace=True)
    fc_daily["Date"] = pd.to_datetime(fc_daily["Date"]).dt.date

    # Latest historical stock (as-of last date in curated sales)
    as_of_stock = 0.0
    if not sc.empty and "date" in sc.columns and "available_stock" in sc.columns:
        sc_dates = pd.to_datetime(sc["date"], errors="coerce")
        if sc_dates.notna().any():
            last_hist = sc_dates.max()
            as_of_stock = float(sc.loc[sc_dates == last_hist, "available_stock"].sum())

    # Projected stock over the horizon
    proj = fc_daily.copy()
    proj["CumForecast"] = proj["Forecasted Units"].cumsum()
    proj["Projected Stock (L)"] = np.maximum(as_of_stock - proj["CumForecast"], 0.0)
    proj_plot = proj[["Date", "Forecasted Units", "Projected Stock (L)"]]

    fig = px.line(proj_plot, x="Date", y=["Forecasted Units", "Projected Stock (L)"])
    fig.update_layout(legend_title_text="", margin=dict(l=20, r=20, t=10, b=10), height=360)
    st.plotly_chart(fig, theme="streamlit", width="stretch")
else:
    st.info("Run the pipeline to generate forecasts and stock context for the selected horizon.")

# -----------------------------------------------------------------------------
# Batch Drill-down (bigger table)
# -----------------------------------------------------------------------------
st.markdown("### Batch Drill-down")
st.caption("Lot-level operational details for the current selection (validity, stock, temperature, energy).")

if not sc.empty:
    cols = [
        "batch_id", "product_id", "store_id", "date",
        "production_date", "shelf_life_days", "days_to_expiration",
        "available_stock", "warehouse_location", "storage_temperature_c",
        "transport_temperature_c", "energy_consumption_kwh"
    ]
    show_cols = [c for c in cols if c in sc.columns]
    drill = (
        sc[show_cols]
        .drop_duplicates(subset=[c for c in show_cols if c in ["batch_id", "date", "store_id"]])
        .sort_values(["date", "batch_id"])
        .copy()
    )
    drill = drill.rename(columns={
        "batch_id": "Batch ID",
        "product_id": "Product",
        "store_id": "Store",
        "date": "Date",
        "production_date": "Production Date",
        "shelf_life_days": "Shelf-life (days)",
        "days_to_expiration": "Days to Expiration",
        "available_stock": "Available Stock (L)",
        "warehouse_location": "Warehouse",
        "storage_temperature_c": "Storage Temp (ºC)",
        "transport_temperature_c": "Transport Temp (ºC)",
        "energy_consumption_kwh": "Energy (kWh)",
    })
    for col in ["Date", "Production Date"]:
        if col in drill.columns:
            drill[col] = pd.to_datetime(drill[col]).dt.strftime("%Y-%m-%d")
    st.dataframe(drill, width="stretch", height=520)
else:
    st.info("No batch-level data for the current selection.")

# -----------------------------------------------------------------------------
# Expiration Risk — heatmap (Product × Store, worst risk per pair)
# -----------------------------------------------------------------------------
st.markdown("### Expiration Risk by Product and Store")
st.caption(
    "Risk per product–store (High/Medium/Low) based on Days to Expiration. "
    "Use it to prioritize actions: promotion, donation or reallocation."
)

if not sc.empty and {"product_id", "store_id", "days_to_expiration"}.issubset(sc.columns):
    # 1) Score per line (2=High, 1=Medium, 0=Low)
    risk = sc[["product_id", "store_id", "days_to_expiration"]].dropna().copy()
    risk["RiskScore"] = np.where(
        risk["days_to_expiration"] <= 1, 2,
        np.where(risk["days_to_expiration"] <= 3, 1, 0)
    )

    # 2) Worst risk per product × store
    agg = (
        risk.groupby(["product_id", "store_id"], as_index=False)["RiskScore"]
            .max()
            .rename(columns={"store_id": "Store"})
    )

    # 3) Labels and ordering
    def _prod_name(pid: int) -> str:
        return product_map.get(int(pid), f"Product {int(pid)}").title()

    agg["Product"] = agg["product_id"].astype(int).map(_prod_name)
    agg["Store"] = agg["Store"].astype(str)

    store_order = sorted(agg["Store"].unique(), key=_store_natural_key)
    prod_order_df = agg[["product_id", "Product"]].drop_duplicates().sort_values("product_id")
    product_order = prod_order_df["Product"].tolist()

    # 4) Pivot heatmap (rows=product, columns=store)
    heat = (
        agg.pivot(index="Product", columns="Store", values="RiskScore")
           .reindex(index=product_order, columns=store_order)
           .fillna(0)
    )

    # 5) Heatmap Low/Medium/High
    fig = px.imshow(
        heat,
        color_continuous_scale=["#b3ffb3", "#ffeb99", "#ff4d4d"],  # Low / Medium / High
        zmin=0, zmax=2,
        aspect="auto",
        labels=dict(color="Risk Level"),
    )
    fig.update_xaxes(title="Store")
    fig.update_yaxes(title="Product")
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        height=420,
        coloraxis_colorbar=dict(
            tickvals=[0, 1, 2],
            ticktext=["Low", "Medium", "High"],
        ),
    )
    st.plotly_chart(fig, theme="streamlit", width="stretch")
else:
    st.info("Not enough data to build the heatmap yet.")

# -----------------------------------------------------------------------------
# Suggested Production Planner (respects horizon)
# -----------------------------------------------------------------------------
st.markdown("### Suggested Production Plan")
st.caption("Daily production needed to meet demand while respecting line capacity.")
if not fcw.empty:
    planner = fcw[["forecast_date", "product_id", "store_id", "suggested_production"]].copy()
    planner.rename(columns={
        "forecast_date": "Date",
        "product_id": "Product",
        "store_id": "Store",
        "suggested_production": "Suggested Production (L/day)"
    }, inplace=True)
    planner["Date"] = pd.to_datetime(planner["Date"]).dt.date

    def _prod_lbl(pid: int) -> str:
        return _product_label(int(pid), product_map)

    planner["Product"] = planner["Product"].astype(int).map(_prod_lbl)
    planner["Store"] = planner["Store"].astype(str)
    st.dataframe(planner.sort_values(["Date", "Product", "Store"]), width="stretch", height=420)
else:
    st.info("No suggested production for the selected filters/horizon — run the pipeline or adjust filters.")

# -----------------------------------------------------------------------------
# Sustainability & Energy
# -----------------------------------------------------------------------------
st.markdown("### Sustainability & Energy")
st.caption("Factory energy per liter, average transport temperature, and average daily logistics energy.")

if not sc.empty:
    enr = sc.copy()

    # Factory energy per liter (avg)
    fac_ok = {"energy_consumption_kwh", "quantity_produced_liters"}.issubset(enr.columns)
    factory_kwh_per_l = None
    if fac_ok:
        denom = enr["quantity_produced_liters"].replace(0, np.nan)
        factory_kwh_per_l = (enr["energy_consumption_kwh"] / denom).dropna().mean()

    # Transport temperature (avg)
    temp_c = None
    if "transport_temperature_c" in enr.columns:
        temp_c = enr["transport_temperature_c"].dropna().mean()

    # Logistics energy per day
    log_kwh_per_day = None
    if {"date", "energy_consumption_kwh"}.issubset(enr.columns):
        log_kwh_per_day = enr.groupby("date")["energy_consumption_kwh"].sum().mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Factory Energy Intensity", f"{_fmt_float(factory_kwh_per_l or 0.0, 2)} kWh/L")
    m2.metric("Avg Transport Temperature", f"{_fmt_float(temp_c or 0.0, 1)} ºC")
    m3.metric("Avg Logistics Energy per Day", f"{_fmt_float(log_kwh_per_day or 0.0, 1)} kWh")
else:
    st.info("No curated data available to compute sustainability metrics yet.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.caption(
    "Tip: Use **Create database** to start, **Generate new data** to simulate continuous operations, "
    "and **Run pipeline** to refresh forecasts and KPIs."
)
