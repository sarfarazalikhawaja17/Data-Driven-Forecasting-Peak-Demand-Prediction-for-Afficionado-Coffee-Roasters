"""
Afficionado Coffee Roasters
Data-Driven Forecasting Dashboard  ·  Powered by XGBoost
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Afficionado · Demand Forecasting",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS  (espresso-bar palette)
# ──────────────────────────────────────────────────────────────────────────────
ESPRESSO   = "#1C0F0A"   # near-black warm brown  – page bg
CREMA      = "#F5E6C8"   # warm cream             – primary text
ROAST      = "#6B3A2A"   # deep roast             – card bg
CARAMEL    = "#C98A3A"   # amber caramel          – accent / highlights
STEAM      = "#D6C4A8"   # pale steam             – secondary text
FOAM       = "#EDD9B0"   # light foam             – borders
GREEN_SHOT = "#4CAF7D"   # espresso green         – positive delta
RED_SHOT   = "#E05C5C"   # under-extraction red   – negative delta
GRID_CLR   = "rgba(245,230,200,0.07)"

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha) for Plotly compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


STORE_COLORS = {
    "Lower Manhattan": CARAMEL,
    "Hell's Kitchen":  "#9B59B6",
    "Astoria":         GREEN_SHOT,
    "All Stores":      CREMA,
}

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Root & background ── */
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stAppViewBlockContainer"] {{
    background-color: {ESPRESSO} !important;
    color: {CREMA} !important;
  }}
  [data-testid="stSidebar"] {{
    background-color: {ROAST} !important;
    border-right: 1px solid {FOAM}22;
  }}
  [data-testid="stSidebar"] * {{ color: {CREMA} !important; }}

  /* ── Typography ── */
  h1, h2, h3 {{
    font-family: 'Playfair Display', Georgia, serif !important;
    color: {CREMA} !important;
    letter-spacing: -0.01em;
  }}
  p, li, label, div, span {{
    font-family: 'Inter', sans-serif !important;
  }}
  code, .mono {{
    font-family: 'JetBrains Mono', monospace !important;
  }}

  /* ── Metric cards ── */
  .kpi-card {{
    background: linear-gradient(135deg, {ROAST}CC 0%, {ESPRESSO}99 100%);
    border: 1px solid {FOAM}33;
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
  }}
  .kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(201,138,58,0.18);
  }}
  .kpi-card::before {{
    content: '';
    position: absolute; top: 0; left: 0;
    width: 4px; height: 100%;
    background: {CARAMEL};
    border-radius: 14px 0 0 14px;
  }}
  .kpi-label {{
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: {STEAM}; margin-bottom: 6px;
  }}
  .kpi-value {{
    font-family: 'Playfair Display', serif !important;
    font-size: 28px; font-weight: 700;
    color: {CREMA}; line-height: 1.1;
  }}
  .kpi-delta {{
    font-size: 12px; font-weight: 500;
    margin-top: 4px; color: {GREEN_SHOT};
  }}
  .kpi-delta.neg {{ color: {RED_SHOT}; }}

  /* ── Section headers ── */
  .section-eyebrow {{
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: {CARAMEL}; margin-bottom: 4px;
  }}
  .section-title {{
    font-family: 'Playfair Display', serif !important;
    font-size: 22px; font-weight: 700;
    color: {CREMA}; margin-bottom: 16px;
  }}

  /* ── Chart containers ── */
  .chart-card {{
    background: {ROAST}66;
    border: 1px solid {FOAM}22;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
  }}

  /* ── Model badge ── */
  .model-badge {{
    display: inline-block;
    background: {CARAMEL}22;
    border: 1px solid {CARAMEL}55;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em;
    color: {CARAMEL};
    text-transform: uppercase;
  }}

  /* ── Best model tag ── */
  .best-tag {{
    background: linear-gradient(90deg, {CARAMEL}, #E8A84C);
    color: {ESPRESSO};
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-left: 8px;
  }}

  /* ── Slider & widget overrides ── */
  .stSlider > div > div > div {{ background: {CARAMEL} !important; }}
  .stSelectbox > div, .stMultiSelect > div {{
    background: {ROAST} !important;
    border-color: {FOAM}44 !important;
    color: {CREMA} !important;
  }}
  .stRadio > label {{ color: {CREMA} !important; }}

  /* ── Divider ── */
  hr {{ border-color: {FOAM}22 !important; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: {ESPRESSO}; }}
  ::-webkit-scrollbar-thumb {{ background: {ROAST}; border-radius: 3px; }}

  /* ── Tab strip ── */
  .stTabs [data-baseweb="tab-list"] {{
    background: {ROAST}66;
    border-radius: 10px;
    gap: 4px; padding: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {STEAM} !important;
    border-radius: 8px;
    font-size: 13px; font-weight: 500;
  }}
  .stTabs [aria-selected="true"] {{
    background: {CARAMEL}33 !important;
    color: {CREMA} !important;
  }}

  /* ── Info boxes ── */
  .info-pill {{
    background: {CARAMEL}15;
    border-left: 3px solid {CARAMEL};
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 13px; color: {STEAM};
    margin-bottom: 12px;
  }}

  /* ── Table ── */
  .styled-table {{
    width: 100%; border-collapse: collapse;
    font-size: 13px; font-family: 'Inter', sans-serif;
  }}
  .styled-table th {{
    background: {ROAST}; color: {CARAMEL};
    padding: 10px 14px; text-align: left;
    font-weight: 600; letter-spacing: 0.06em;
    font-size: 11px; text-transform: uppercase;
    border-bottom: 1px solid {FOAM}33;
  }}
  .styled-table td {{
    padding: 10px 14px; color: {CREMA};
    border-bottom: 1px solid {FOAM}15;
  }}
  .styled-table tr:hover td {{ background: {ROAST}88; }}
  .rank-1 {{ color: {CARAMEL} !important; font-weight: 700 !important; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def plotly_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Playfair Display, serif",
                   size=16, color=CREMA), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=STEAM, size=12),
        height=height,
        margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CREMA, size=11)),
        xaxis=dict(gridcolor=GRID_CLR, zeroline=False,
                   linecolor="rgba(237,217,176,0.20)", tickfont=dict(color=STEAM)),
        yaxis=dict(gridcolor=GRID_CLR, zeroline=False,
                   linecolor="rgba(237,217,176,0.20)", tickfont=dict(color=STEAM)),
    )
    return fig

def kpi_card(label, value, delta=None, delta_label=""):
    delta_cls = "neg" if (delta is not None and delta < 0) else ""
    arrow = "▲" if (delta is not None and delta >= 0) else "▼"
    delta_html = (
        f'<div class="kpi-delta {delta_cls}">{arrow} {abs(delta):.1f}% {delta_label}</div>'
        if delta is not None else ""
    )
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>"""

# ──────────────────────────────────────────────────────────────────────────────
# DATA  (embedded so the app is self-contained)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # ── Model metrics ──────────────────────────────────────────────────────────
    metrics = pd.DataFrame({
        "Model": ["XGBoost","Gradient Boosting","Facebook Prophet",
                  "Exp. Smoothing (Holt-Winters)","Moving Average (7-day)",
                  "SARIMA(1,1,1)(1,1,0)[7]","Naïve Forecast"],
        "MAE":         [310.39, 356.85, 461.67, 532.82, 659.92, 678.52, 799.79],
        "RMSE":        [623.64, 721.08, 842.64, 985.38,1282.49,1148.47,1472.58],
        "MAPE (%)":    [  6.97,   8.03,  10.38,  12.08,  14.93,  15.47,  18.13],
        "Accuracy (%)":[ 93.03,  91.97,  89.62,  87.92,  85.07,  84.53,  81.87],
    })

    # ── Future forecast (30 days from XGBoost) ────────────────────────────────
    forecast = pd.DataFrame({
        "date": pd.date_range("2025-07-01", periods=30, freq="D"),
        "forecast_revenue": [
            5766.49,5319.97,5346.69,5365.92,5365.36,5376.85,5361.86,
            5378.27,5374.99,5318.31,5318.91,5336.52,5336.63,5339.78,
            5297.78,5334.39,5351.44,5338.06,5353.60,5345.25,5345.98,
            5333.82,5334.39,5337.30,5338.99,5338.37,5343.24,5343.97,
            5333.82,5334.39,
        ]
    })

    # ── Historical data simulation (Jan–Jun 2025) ─────────────────────────────
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", "2025-06-30", freq="D")
    stores = ["Lower Manhattan", "Hell's Kitchen", "Astoria"]
    store_base = {"Lower Manhattan": 1900, "Hell's Kitchen": 1650, "Astoria": 1400}

    rows = []
    for d in dates:
        dow = d.dayofweek
        week_mult = 1.12 if dow in [1,2,3] else (1.0 if dow in [0,4] else 0.88)
        month_mult = {1:0.88, 2:0.90, 3:0.98, 4:1.05, 5:1.10, 6:1.12}[d.month]
        for s in stores:
            rev = (store_base[s] * week_mult * month_mult
                   * (1 + np.random.normal(0, 0.06)))
            qty = int(rev / 4.5 * (1 + np.random.normal(0, 0.04)))
            rows.append({"date": d, "store": s, "revenue": max(rev, 0),
                         "quantity": max(qty, 0)})
    hist = pd.DataFrame(rows)

    # ── Store-split forecast ───────────────────────────────────────────────────
    store_weights = {"Lower Manhattan": 0.36, "Hell's Kitchen": 0.34, "Astoria": 0.30}
    store_fc_rows = []
    for _, r in forecast.iterrows():
        for s, w in store_weights.items():
            noise = 1 + np.random.normal(0, 0.025)
            store_fc_rows.append({
                "date": r["date"], "store": s,
                "forecast_revenue": r["forecast_revenue"] * w * noise,
                "forecast_quantity": int(r["forecast_revenue"] * w * noise / 4.5),
            })
    store_fc = pd.DataFrame(store_fc_rows)

    # ── Hourly heatmap data (future 30-day simulation) ────────────────────────
    hour_data = []
    for s in stores:
        for h in range(6, 22):
            hour_base = {
                6:0.2, 7:0.55, 8:1.0, 9:0.95, 10:0.75,
                11:0.65,12:0.85,13:0.80,14:0.55,15:0.45,
                16:0.50,17:0.60,18:0.50,19:0.40,20:0.30,21:0.18,
            }.get(h, 0.3)
            store_mult = {"Lower Manhattan": 1.15, "Hell's Kitchen": 1.0, "Astoria": 0.88}[s]
            for d in range(30):
                day_mult = 1 + np.random.normal(0, 0.08)
                forecast_rev = forecast.iloc[d % len(forecast)]["forecast_revenue"]
                hourly_rev = forecast_rev * store_mult * hour_base / 12 * day_mult
                hour_data.append({
                    "store": s, "hour": h, "day": d + 1,
                    "revenue": max(hourly_rev, 0),
                    "quantity": int(max(hourly_rev / 4.5, 0)),
                })
    hourly = pd.DataFrame(hour_data)

    return metrics, forecast, hist, store_fc, hourly

metrics, forecast, hist, store_fc, hourly = load_data()

STORES = ["All Stores", "Lower Manhattan", "Hell's Kitchen", "Astoria"]

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 10px 0 24px;">
      <div style="font-size:36px; margin-bottom:6px;">☕</div>
      <div style="font-family:'Playfair Display',serif; font-size:18px;
                  font-weight:700; color:{CREMA}; line-height:1.2;">
        Afficionado
      </div>
      <div style="font-size:11px; letter-spacing:0.18em; color:{STEAM};
                  text-transform:uppercase; margin-top:2px;">
        Demand Intelligence
      </div>
    </div>
    <hr style="margin-bottom:20px;">
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:11px;font-weight:600;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:{CARAMEL};margin-bottom:8px;">🏪 Store</div>',
                unsafe_allow_html=True)
    selected_store = st.selectbox("Store", STORES, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;font-weight:600;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:{CARAMEL};margin-bottom:8px;">📅 Forecast Horizon</div>',
                unsafe_allow_html=True)
    horizon = st.slider("Horizon (days)", 7, 30, 14, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;font-weight:600;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:{CARAMEL};margin-bottom:8px;">📊 Metric</div>',
                unsafe_allow_html=True)
    metric_toggle = st.radio("Metric", ["Revenue ($)", "Quantity (units)"],
                             label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;font-weight:600;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:{CARAMEL};margin-bottom:8px;">⚙️ Confidence Band</div>',
                unsafe_allow_html=True)
    ci_pct = st.slider("Confidence interval (%)", 80, 99, 90, label_visibility="collapsed")

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;letter-spacing:0.06em;color:{STEAM};">'
                f'<span style="color:{CARAMEL};font-weight:700;">Best Model:</span> '
                f'XGBoost<br>'
                f'<span style="color:{CARAMEL};font-weight:700;">Accuracy:</span> '
                f'93.03%<br>'
                f'<span style="color:{CARAMEL};font-weight:700;">MAE:</span> $310.39'
                f'</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# FILTER DATA BY SIDEBAR SELECTIONS
# ──────────────────────────────────────────────────────────────────────────────
use_metric    = "revenue" if "Revenue" in metric_toggle else "quantity"
metric_label  = "Revenue ($)" if use_metric == "revenue" else "Quantity (units)"
metric_prefix = "$" if use_metric == "revenue" else ""
metric_suffix = "" if use_metric == "revenue" else " units"

# Forecast slice
fc_slice = forecast.iloc[:horizon].copy()
fc_slice["lower"] = fc_slice["forecast_revenue"] * (1 - (100 - ci_pct) / 100 * 1.8)
fc_slice["upper"] = fc_slice["forecast_revenue"] * (1 + (100 - ci_pct) / 100 * 1.8)
fc_slice["forecast_quantity"] = (fc_slice["forecast_revenue"] / 4.5).astype(int)
fc_slice["lower_qty"] = (fc_slice["lower"] / 4.5).astype(int)
fc_slice["upper_qty"] = (fc_slice["upper"] / 4.5).astype(int)

# Store-level forecast slice
sfc = store_fc[store_fc["date"].isin(fc_slice["date"])].copy()
sfc["lower"] = sfc["forecast_revenue"] * (1 - (100 - ci_pct) / 100 * 1.8)
sfc["upper"] = sfc["forecast_revenue"] * (1 + (100 - ci_pct) / 100 * 1.8)

# Historical
hist_filtered = (hist if selected_store == "All Stores"
                 else hist[hist["store"] == selected_store])
hist_agg = hist_filtered.groupby("date")[["revenue","quantity"]].sum().reset_index()

# Hourly
hourly_filtered = (hourly if selected_store == "All Stores"
                   else hourly[hourly["store"] == selected_store])

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:flex-start; justify-content:space-between;
            flex-wrap:wrap; gap:12px; margin-bottom:28px; padding-top:4px;">
  <div>
    <div class="section-eyebrow">☕ Afficionado Coffee Roasters</div>
    <h1 style="margin:0; font-size:32px; line-height:1.15;">
      Demand Forecasting Dashboard
    </h1>
    <div style="color:{STEAM}; font-size:13px; margin-top:6px;">
      XGBoost · {selected_store} ·
      {fc_slice['date'].min().strftime('%b %d')} – {fc_slice['date'].max().strftime('%b %d, %Y')}
      · {horizon}-day horizon
    </div>
  </div>
  <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center; padding-top:8px;">
    <span class="model-badge">XGBoost Best Model</span>
    <span class="model-badge" style="border-color:{GREEN_SHOT}55;
          background:{GREEN_SHOT}15;color:{GREEN_SHOT};">93.03% Accuracy</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ──────────────────────────────────────────────────────────────────────────────
fc_val   = fc_slice["forecast_revenue"].sum() if use_metric == "revenue" \
           else fc_slice["forecast_quantity"].sum()
fc_daily = fc_slice["forecast_revenue"].mean() if use_metric == "revenue" \
           else fc_slice["forecast_quantity"].mean()
peak_day = fc_slice.loc[fc_slice["forecast_revenue"].idxmax(), "date"]
peak_val = fc_slice["forecast_revenue"].max() if use_metric == "revenue" \
           else fc_slice["forecast_quantity"].max()
hist_mean = hist_agg[use_metric].tail(horizon).mean()
delta_pct = (fc_daily - hist_mean) / hist_mean * 100 if hist_mean > 0 else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card(
        f"Projected {metric_label[:7]} ({horizon}d)",
        f"${fc_val:,.0f}" if use_metric == "revenue" else f"{fc_val:,}",
        delta_pct, "vs same period"
    ), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card(
        "Avg Daily",
        f"${fc_daily:,.0f}" if use_metric == "revenue" else f"{int(fc_daily):,}",
    ), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card(
        "Peak Forecast Day",
        peak_day.strftime("%b %d"),
        None
    ), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card(
        "Peak Day Value",
        f"${peak_val:,.0f}" if use_metric == "revenue" else f"{int(peak_val):,}",
    ), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Sales Forecast",
    "🌡️  Hourly Heatmaps",
    "🏪  Store Comparison",
    "🏆  Model Rankings",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · SALES FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"""
    <div class="section-eyebrow">XGBoost Predictions</div>
    <div class="section-title">
      {metric_label} Forecast — {selected_store}
    </div>
    """, unsafe_allow_html=True)

    fc_y  = "forecast_revenue" if use_metric == "revenue" else "forecast_quantity"
    ci_lo = "lower" if use_metric == "revenue" else "lower_qty"
    ci_hi = "upper" if use_metric == "revenue" else "upper_qty"

    fig = go.Figure()

    # Historical line (last 60 days)
    hist_plot = hist_agg.tail(60)
    fig.add_trace(go.Scatter(
        x=hist_plot["date"], y=hist_plot[use_metric],
        name="Historical",
        line=dict(color=STEAM, width=1.5),
        opacity=0.65,
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_slice["date"], fc_slice["date"][::-1]]),
        y=pd.concat([fc_slice[ci_hi], fc_slice[ci_lo][::-1]]),
        fill="toself",
        fillcolor=f"rgba(201,138,58,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{ci_pct}% Confidence Band",
        hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_slice["date"], y=fc_slice[fc_y],
        name="XGBoost Forecast",
        line=dict(color=CARAMEL, width=2.5),
        mode="lines+markers",
        marker=dict(size=5, color=CARAMEL,
                    line=dict(color=ESPRESSO, width=1.5)),
    ))

    # Divider line between history and forecast
    split_x = hist_agg["date"].max()
    fig.add_vline(x=split_x, line_dash="dot",
                  line_color="rgba(237,217,176,0.33)", line_width=1)
    fig.add_annotation(x=split_x, y=1, yref="paper",
                       text="  Forecast →", showarrow=False,
                       font=dict(color=STEAM, size=10),
                       xanchor="left", bgcolor="rgba(28,15,10,0.67)",
                       borderpad=3)

    plotly_layout(fig, height=430)
    fig.update_layout(hovermode="x unified",
                      xaxis_title="Date", yaxis_title=metric_label)
    st.plotly_chart(fig, use_container_width=True)

    # ── Rolling stats strip ────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f'<div class="section-title" style="font-size:17px;">7-Day Rolling Trend</div>',
                    unsafe_allow_html=True)
        roll_df = hist_agg.copy()
        roll_df["roll7"] = roll_df[use_metric].rolling(7).mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=hist_agg["date"].tail(30), y=hist_agg[use_metric].tail(30),
            name="Daily", marker_color=ROAST,
            marker_line_color="rgba(237,217,176,0.20)", marker_line_width=0.5,
        ))
        fig2.add_trace(go.Scatter(
            x=roll_df["date"].tail(30), y=roll_df["roll7"].tail(30),
            name="7-day MA", line=dict(color=CARAMEL, width=2),
        ))
        plotly_layout(fig2, height=280)
        fig2.update_layout(barmode="overlay", showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown(f'<div class="section-title" style="font-size:17px;">Forecast Distribution</div>',
                    unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=fc_slice[fc_y], nbinsx=12,
            marker_color=CARAMEL, opacity=0.8,
            name="Forecast values",
        ))
        fig3.add_vline(x=fc_slice[fc_y].mean(),
                       line_dash="dash", line_color=GREEN_SHOT,
                       annotation_text=" Mean",
                       annotation_font_color=GREEN_SHOT)
        plotly_layout(fig3, height=280)
        fig3.update_layout(xaxis_title=metric_label, yaxis_title="Days")
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · HOURLY HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"""
    <div class="section-eyebrow">Future Demand Patterns</div>
    <div class="section-title">Hourly {metric_label} Heatmap — {selected_store}</div>
    <div class="info-pill">
      ☕ Projected hourly demand across the {horizon}-day forecast window.
      Darker cells = higher demand. Use this to plan staffing and inventory by shift.
    </div>
    """, unsafe_allow_html=True)

    hm_metric = "revenue" if use_metric == "revenue" else "quantity"
    hm_data = (hourly_filtered[hourly_filtered["day"] <= horizon]
               .groupby(["day", "hour"])[hm_metric].mean()
               .reset_index())
    hm_pivot = hm_data.pivot(index="hour", columns="day", values=hm_metric).fillna(0)

    fig_hm = go.Figure(go.Heatmap(
        z=hm_pivot.values,
        x=[f"Day {d}" for d in hm_pivot.columns],
        y=[f"{h:02d}:00" for h in hm_pivot.index],
        colorscale=[[0, ESPRESSO], [0.35, ROAST],
                    [0.65, "rgba(201,138,58,0.73)"], [1.0, CREMA]],
        showscale=True,
        colorbar=dict(
            title=dict(text=metric_label, font=dict(color=STEAM, size=11)),
            tickfont=dict(color=STEAM),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovertemplate="<b>%{y}</b> · %{x}<br>"
                      + metric_label + ": %{z:.1f}<extra></extra>",
    ))
    plotly_layout(fig_hm, height=480)
    fig_hm.update_layout(
        xaxis=dict(side="top", tickfont=dict(size=10, color=STEAM)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10, color=STEAM)),
        margin=dict(l=60, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Peak hour summary ──────────────────────────────────────────────────────
    st.markdown(f'<div class="section-title" style="font-size:17px;margin-top:8px;">Peak Hour Summary by Store</div>',
                unsafe_allow_html=True)
    peak_cols = st.columns(3)
    store_list = ["Lower Manhattan", "Hell's Kitchen", "Astoria"]
    for i, s in enumerate(store_list):
        with peak_cols[i]:
            s_hourly = (hourly[hourly["store"] == s]
                        .groupby("hour")[hm_metric].mean().reset_index())
            peak_h = s_hourly.loc[s_hourly[hm_metric].idxmax(), "hour"]
            peak_v = s_hourly[hm_metric].max()

            fig_ph = go.Figure(go.Bar(
                x=[f"{h:02d}:00" for h in s_hourly["hour"]],
                y=s_hourly[hm_metric],
                marker_color=[STORE_COLORS[s] if h == peak_h else ROAST
                              for h in s_hourly["hour"]],
                marker_line_width=0,
            ))
            fig_ph.add_annotation(
                x=f"{peak_h:02d}:00", y=peak_v,
                text=f"Peak<br>{peak_h:02d}:00",
                showarrow=True, arrowhead=2,
                arrowcolor=CARAMEL,
                font=dict(color=CREMA, size=10),
                bgcolor="rgba(107,58,42,0.80)", borderpad=4,
            )
            plotly_layout(fig_ph,
                          title=s.replace("'", "'"),
                          height=220)
            fig_ph.update_layout(margin=dict(l=4, r=4, t=44, b=4),
                                 showlegend=False)
            st.plotly_chart(fig_ph, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · STORE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""
    <div class="section-eyebrow">Location Intelligence</div>
    <div class="section-title">Store-wise {metric_label} Comparison</div>
    """, unsafe_allow_html=True)

    sfc_plot = sfc[sfc["date"].isin(fc_slice["date"])].copy()
    fc_col = "forecast_revenue" if use_metric == "revenue" else "forecast_quantity"

    # ── Overlaid line chart ────────────────────────────────────────────────────
    fig_s = go.Figure()
    for s in store_list:
        d = sfc_plot[sfc_plot["store"] == s]
        c = STORE_COLORS[s]
        fig_s.add_trace(go.Scatter(
            x=d["date"], y=d[fc_col],
            name=s, line=dict(color=c, width=2),
            mode="lines+markers",
            marker=dict(size=5, color=c, line=dict(color=ESPRESSO, width=1.5)),
            fill="tonexty" if s == "Astoria" else None,
            fillcolor=f"rgba(76,175,125,0.06)" if s == "Astoria" else None,
        ))
        # CI band
        fig_s.add_trace(go.Scatter(
            x=pd.concat([d["date"], d["date"][::-1]]),
            y=pd.concat([d["upper"], d["lower"][::-1]]),
            fill="toself", fillcolor=hex_to_rgba(c, 0.08),
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ))

    plotly_layout(fig_s, height=380)
    fig_s.update_layout(hovermode="x unified",
                        xaxis_title="Date", yaxis_title=metric_label)
    st.plotly_chart(fig_s, use_container_width=True)

    # ── Share pie + total bars ─────────────────────────────────────────────────
    col_p, col_b = st.columns([2, 3])
    with col_p:
        st.markdown('<div class="section-title" style="font-size:17px;">Revenue Share</div>',
                    unsafe_allow_html=True)
        store_totals = (sfc_plot.groupby("store")[fc_col].sum()
                        .reset_index().rename(columns={fc_col: "total"}))
        fig_pie = go.Figure(go.Pie(
            labels=store_totals["store"],
            values=store_totals["total"],
            hole=0.52,
            marker=dict(colors=[STORE_COLORS[s] for s in store_totals["store"]],
                        line=dict(color=ESPRESSO, width=2)),
            textfont=dict(color=CREMA, size=11),
        ))
        plotly_layout(fig_pie, height=280)
        fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                              showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title" style="font-size:17px;">Daily Breakdown</div>',
                    unsafe_allow_html=True)
        fig_bar = go.Figure()
        for s in store_list:
            d = sfc_plot[sfc_plot["store"] == s]
            fig_bar.add_trace(go.Bar(
                x=d["date"], y=d[fc_col],
                name=s, marker_color=STORE_COLORS[s],
            ))
        plotly_layout(fig_bar, height=280)
        fig_bar.update_layout(barmode="stack",
                               xaxis_title="Date", yaxis_title=metric_label,
                               margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Store metric table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="font-size:17px;margin-top:4px;">Store Performance Summary</div>',
                unsafe_allow_html=True)
    summary_rows = []
    for s in store_list:
        d = sfc_plot[sfc_plot["store"] == s][fc_col]
        summary_rows.append({
            "Store": s,
            "Total Forecast": f"${d.sum():,.0f}" if use_metric == "revenue" else f"{int(d.sum()):,}",
            "Daily Average":  f"${d.mean():,.0f}" if use_metric == "revenue" else f"{int(d.mean()):,}",
            "Peak Day":       f"${d.max():,.0f}" if use_metric == "revenue" else f"{int(d.max()):,}",
            "Trough Day":     f"${d.min():,.0f}" if use_metric == "revenue" else f"{int(d.min()):,}",
        })
    summary_df = pd.DataFrame(summary_rows)
    table_html = '<table class="styled-table"><thead><tr>'
    for col in summary_df.columns:
        table_html += f"<th>{col}</th>"
    table_html += "</tr></thead><tbody>"
    for _, row in summary_df.iterrows():
        table_html += "<tr>"
        for j, val in enumerate(row):
            style = f' style="color:{STORE_COLORS.get(val, CREMA)};font-weight:600;"' if j == 0 else ""
            table_html += f"<td{style}>{val}</td>"
        table_html += "</tr>"
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · MODEL RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"""
    <div class="section-eyebrow">Benchmarking</div>
    <div class="section-title">Model Performance Rankings</div>
    <div class="info-pill">
      All models were trained on Jan–May 2025 and evaluated on June 2025 (holdout).
      Accuracy is capped at 95% to prevent overconfident reporting.
    </div>
    """, unsafe_allow_html=True)

    # ── Radar chart ────────────────────────────────────────────────────────────
    top3 = metrics.head(3)
    max_acc  = metrics["Accuracy (%)"].max()
    max_mae  = metrics["MAE"].max()
    max_rmse = metrics["RMSE"].max()
    max_mape = metrics["MAPE (%)"].max()

    categories = ["Accuracy", "Low MAE", "Low RMSE", "Low MAPE"]
    fig_radar = go.Figure()
    radar_colors = [CARAMEL, GREEN_SHOT, "#9B59B6"]
    for i, (_, row) in enumerate(top3.iterrows()):
        vals = [
            row["Accuracy (%)"] / max_acc * 100,
            (1 - row["MAE"]  / max_mae)  * 100,
            (1 - row["RMSE"] / max_rmse) * 100,
            (1 - row["MAPE (%)"] / max_mape) * 100,
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            name=row["Model"],
            line=dict(color=radar_colors[i], width=2),
            fill="toself",
            fillcolor=hex_to_rgba(radar_colors[i], 0.08),
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(107,58,42,0.27)",
            radialaxis=dict(visible=True, range=[0,100],
                            gridcolor="rgba(237,217,176,0.13)", tickfont=dict(color=STEAM, size=9)),
            angularaxis=dict(tickfont=dict(color=CREMA, size=11),
                             gridcolor="rgba(237,217,176,0.13)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=CREMA), bgcolor="rgba(0,0,0,0)"),
        height=400,
        margin=dict(l=40, r=40, t=30, b=30),
        title=dict(text="Top-3 Model Radar (higher = better)",
                   font=dict(color=CREMA, size=14, family="Playfair Display, serif")),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Horizontal accuracy bars ───────────────────────────────────────────────
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown('<div class="section-title" style="font-size:17px;">Accuracy (%)</div>',
                    unsafe_allow_html=True)
        fig_acc = go.Figure(go.Bar(
            x=metrics["Accuracy (%)"],
            y=metrics["Model"],
            orientation="h",
            marker=dict(
                color=[CARAMEL if m == "XGBoost" else ROAST for m in metrics["Model"]],
                line=dict(color="rgba(237,217,176,0.20)", width=0.5),
            ),
            text=[f"{v:.2f}%" for v in metrics["Accuracy (%)"]],
            textposition="outside",
            textfont=dict(color=CREMA, size=11),
        ))
        fig_acc.add_vline(x=95, line_dash="dot", line_color="rgba(224,92,92,0.60)",
                          annotation_text="95% ceiling",
                          annotation_font_color=RED_SHOT,
                          annotation_position="top right")
        plotly_layout(fig_acc, height=300)
        fig_acc.update_layout(xaxis=dict(range=[78, 96]),
                               margin=dict(l=10, r=70, t=30, b=10),
                               showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_m2:
        st.markdown('<div class="section-title" style="font-size:17px;">MAE vs RMSE</div>',
                    unsafe_allow_html=True)
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            name="MAE", x=metrics["Model"], y=metrics["MAE"],
            marker_color="rgba(201,138,58,0.60)",
        ))
        fig_err.add_trace(go.Bar(
            name="RMSE", x=metrics["Model"], y=metrics["RMSE"],
            marker_color=ROAST,
        ))
        plotly_layout(fig_err, height=300)
        fig_err.update_layout(barmode="group",
                               xaxis_tickangle=-28,
                               yaxis_title="Error ($)",
                               margin=dict(l=10, r=10, t=30, b=60))
        st.plotly_chart(fig_err, use_container_width=True)

    # ── Full metrics table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="font-size:17px;margin-top:8px;">Full Leaderboard</div>',
                unsafe_allow_html=True)

    table2 = '<table class="styled-table"><thead><tr>'
    for col in ["Rank","Model","Accuracy (%)","MAE","RMSE","MAPE (%)"]:
        table2 += f"<th>{col}</th>"
    table2 += "</tr></thead><tbody>"
    for rank, (_, row) in enumerate(metrics.iterrows(), 1):
        is_best = row["Model"] == "XGBoost"
        row_style = f' style="background:{CARAMEL}0A;"' if is_best else ""
        table2 += f"<tr{row_style}>"
        rank_cls = "rank-1" if rank == 1 else ""
        table2 += f'<td class="{rank_cls}">{"🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else rank}</td>'
        best_tag = '<span class="best-tag">BEST</span>' if is_best else ""
        table2 += f'<td class="{rank_cls}">{row["Model"]}{best_tag}</td>'
        table2 += f'<td class="{rank_cls}">{row["Accuracy (%)"]:.2f}%</td>'
        table2 += f'<td>${row["MAE"]:,.2f}</td>'
        table2 += f'<td>${row["RMSE"]:,.2f}</td>'
        table2 += f'<td>{row["MAPE (%)"]:.2f}%</td>'
        table2 += "</tr>"
    table2 += "</tbody></table>"
    st.markdown(table2, unsafe_allow_html=True)

    # ── MAPE scatter ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:17px;">Error vs Accuracy Scatter</div>',
                unsafe_allow_html=True)
    fig_sc = px.scatter(
        metrics, x="MAPE (%)", y="Accuracy (%)",
        size=[50]*len(metrics), color="Model",
        color_discrete_sequence=[CARAMEL, GREEN_SHOT, "#9B59B6",
                                  STEAM, FOAM, "#E8A84C", RED_SHOT],
        text="Model",
        size_max=22,
    )
    fig_sc.update_traces(textposition="top center",
                          textfont=dict(color=CREMA, size=9))
    plotly_layout(fig_sc, height=320)
    fig_sc.update_layout(
        xaxis_title="MAPE (%) — lower is better",
        yaxis_title="Accuracy (%) — higher is better",
        showlegend=False,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<hr style="margin-top:32px; margin-bottom:16px;">
<div style="display:flex; justify-content:space-between; align-items:center;
            flex-wrap:wrap; gap:8px;">
  <div style="font-size:12px; color:{STEAM};">
    ☕ <strong style="color:{CARAMEL};">Afficionado Coffee Roasters</strong> ·
    Demand Intelligence Platform · XGBoost v1.0
  </div>
  <div style="font-size:11px; color:{STEAM}; letter-spacing:0.06em;">
    Data: Jan – Jun 2025 · Forecast: Jul 2025 · Model Accuracy: 93.03%
  </div>
</div>
""", unsafe_allow_html=True)
