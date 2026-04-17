"""
N1 Performance Lab — Force Plate Analysis Dashboard v4
8-tab system: Coach View | Data Quality | Metrics | PCA & Clustering | Trends | Athlete Profiles | Summary | About & References
Run: streamlit run 02_Logic/n1_forceplate_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings, os, io
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="N1 Force Plate Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── N1 Brutus Design System ────────────────────────────────────────────────────
C = dict(
    bg="#FFFFFF", text="#1A1A1A", border="#1A1A1A",
    pos="#2ECC71", warn="#F39C12", neg="#E74C3C",
    neu="#BDC3C7", line="#1A1A1A", band="rgba(26,26,26,0.07)",
    grid="#F0F0F0",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#fff;color:#1A1A1A}
h1,h2,h3{font-weight:700;letter-spacing:-.02em}
.kpi{background:#F8F8F8;border-left:3px solid #1A1A1A;padding:14px 18px;border-radius:4px;margin-bottom:8px}
.lbl{font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#888;margin-bottom:4px}
.badge-g{display:inline-block;background:#2ECC71;color:#fff;font-size:10px;font-weight:700;
         padding:2px 7px;border-radius:2px}
.badge-r{display:inline-block;background:#E74C3C;color:#fff;font-size:10px;font-weight:700;
         padding:2px 7px;border-radius:2px}
.badge-a{display:inline-block;background:#F39C12;color:#fff;font-size:10px;font-weight:700;
         padding:2px 7px;border-radius:2px}
.badge-n{display:inline-block;background:#BDC3C7;color:#fff;font-size:10px;font-weight:700;
         padding:2px 7px;border-radius:2px}
/* ── Coach View cards ─────────────────────────────────────────────── */
.cv-card{border-radius:8px;padding:20px 22px;margin-bottom:4px;border:1px solid #E8E8E8}
.cv-name{font-size:18px;font-weight:700;letter-spacing:-.01em;margin-bottom:2px}
.cv-status{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px}
.cv-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
.cv-metric{display:inline-block;font-size:12px;margin-right:14px;margin-top:6px}
.cv-metric-label{font-size:10px;color:#888;display:block;margin-bottom:1px}
.cv-metric-val{font-weight:700;font-size:14px}
.cv-flag{font-size:10px;font-weight:600}
.cv-rec{font-size:12px;color:#444;margin-top:10px;padding-top:10px;border-top:1px solid #F0F0F0;line-height:1.5}
.cv-rec-label{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#888;margin-bottom:3px}
.cv-asym-bar{display:inline-block;height:8px;border-radius:2px;vertical-align:middle}
.cv-section{font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
            color:#888;margin:18px 0 8px 0}
</style>
""", unsafe_allow_html=True)

# ── Metric taxonomy — keyword-based so it works with any CSV format ────────────
OUTPUT_KW   = ["jump height","rsi","flight time"]
DRIVER_KW   = ["peak force","braking rfd","impulse","takeoff"]
STRATEGY_KW = ["duration","depth","peak velocity","contraction time"]
ASYM_KW     = ["asym"]
POWER_KW    = ["power"]

def classify_col(col):
    c = col.lower()
    if any(k in c for k in ASYM_KW):   return "Asymmetry"
    if any(k in c for k in OUTPUT_KW):  return "Output"
    if any(k in c for k in DRIVER_KW):  return "Driver"
    if any(k in c for k in STRATEGY_KW):return "Strategy"
    if any(k in c for k in POWER_KW):   return "Power"
    return "Other"

def get_cols(df, keywords):
    """Return numeric columns whose names contain any of the keywords."""
    result = []
    for c in df.columns:
        if df[c].dtype not in [np.float64, np.int64, np.float32]: continue
        if any(k in c.lower() for k in keywords):
            result.append(c)
    return result

# Canonical lists — populated after data load (see sidebar)
OUTPUT_M = DRIVER_M = STRATEGY_M = ASYM_M = POWER_M = []

QUICK_FILTER = {
    "All":       [],
    "Output":    [],
    "Driver":    [],
    "Strategy":  [],
    "Asymmetry": [],
    "Power":     [],
}

# ── Stat helpers ───────────────────────────────────────────────────────────────
def swc(s): return 0.2 * s.std()
def cv_pct(s): return (s.std() / s.mean() * 100) if s.mean() != 0 else np.nan
def percentile_rank(val, series): return stats.percentileofscore(series.dropna(), val, kind="rank")

def flag(val, baseline, swc_val):
    if swc_val == 0 or np.isnan(val): return "n"
    r = (val - baseline) / swc_val
    if r >= 1: return "g"
    if r <= -1: return "r"
    if abs(r) >= 0.5: return "a"
    return "n"

FLAG_LABEL = {"g": "▲ Above SWC", "r": "▼ Below SWC", "a": "~ Marginal", "n": "— Stable"}
FLAG_COLOR = {"g": C["pos"], "r": C["neg"], "a": C["warn"], "n": C["neu"]}
FLAG_BADGE = {"g": "badge-g", "r": "badge-r", "a": "badge-a", "n": "badge-n"}

# ── Data loading ───────────────────────────────────────────────────────────────
def load(src) -> pd.DataFrame:
    df = pd.read_csv(src) if isinstance(src, str) else pd.read_csv(src)
    df.columns = [c.strip() for c in df.columns]
    for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"]:
        try: df["Date"] = pd.to_datetime(df["Date"], format=fmt); break
        except: continue
    num_skip = {"Name","ExternalId","Test Type","Date","Tags"}
    for c in df.columns:
        if c not in num_skip:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.sort_values(["Name","Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def available(df, pool):
    return [m for m in pool if m in df.columns and df[m].notna().any()]

# ── Plot helpers ───────────────────────────────────────────────────────────────
def layout(fig, h=220, t=36):
    fig.update_layout(
        plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
        margin=dict(l=8,r=8,t=t,b=8), height=h, showlegend=False,
        font=dict(family="Inter", color=C["text"]),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=10)),
    )
    return fig

def trend_fig(adf, metric):
    s = adf[metric].dropna()
    if len(s) < 2: return None
    bl, sw = s.mean(), swc(s)
    colors = [FLAG_COLOR[flag(v, bl, sw)] for v in adf[metric]]
    fig = go.Figure()
    fig.add_hrect(y0=bl-sw, y1=bl+sw, fillcolor=C["band"], line_width=0)
    fig.add_hline(y=bl, line_dash="dot", line_color="#aaa", line_width=1)
    fig.add_trace(go.Scatter(
        x=adf["Date"], y=adf[metric],
        mode="lines+markers",
        line=dict(color=C["line"], width=2),
        marker=dict(color=colors, size=9, line=dict(color=C["line"], width=1)),
        hovertemplate="%{x|%d %b}<br><b>%{y:.2f}</b><extra></extra>",
    ))
    fig.update_layout(title=dict(text=metric, font=dict(size=12, family="Inter")))
    return layout(fig)

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## N1 Performance Lab")
    st.markdown('<p class="lbl">Force Plate Dashboard</p>', unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader("Upload CSV (Hawkin / ForceDecks)", type=["csv"],
        help="Wide format: each row = one test. Columns: Name, Date, Jump Height (Imp-Mom) [cm], etc.")
    use_sample = st.checkbox("Use sample dataset", value=uploaded is None)

    if uploaded:
        df = load(uploaded)
    elif use_sample:
        sp = os.path.join(os.path.dirname(__file__), "../01_Raw_Data/synthetic_forcedecks_CMJ_dataset.csv")
        if not os.path.exists(sp):
            st.error("Sample not found. Upload a CSV."); st.stop()
        df = load(sp)
    else:
        st.info("Upload a CSV to begin."); st.stop()

    # Auto-classify columns from whatever CSV was loaded
    OUTPUT_M   = get_cols(df, OUTPUT_KW)
    DRIVER_M   = get_cols(df, DRIVER_KW)
    STRATEGY_M = get_cols(df, STRATEGY_KW)
    ASYM_M     = get_cols(df, ASYM_KW)
    POWER_M    = get_cols(df, POWER_KW)
    QUICK_FILTER["Output"]    = OUTPUT_M
    QUICK_FILTER["Driver"]    = DRIVER_M
    QUICK_FILTER["Strategy"]  = STRATEGY_M
    QUICK_FILTER["Asymmetry"] = ASYM_M
    QUICK_FILTER["Power"]     = POWER_M

    athletes = sorted(df["Name"].unique().tolist())
    st.divider()
    st.markdown("### Filters")
    sel_athletes = st.multiselect("Athletes", athletes, default=athletes[:10],
                                  help="2–20 recommended for PCA")
    sel_athlete  = st.selectbox("Single athlete (Dive / Profile)", ["— Squad —"] + athletes)
    asym_thresh  = st.slider("Asymmetry flag threshold (%)", 5, 20, 10)
    n_clusters   = st.slider("PCA clusters (K-means)", 2, 6, 3)
    st.divider()
    st.markdown('<p class="lbl">Quick metric filter</p>', unsafe_allow_html=True)
    quick = st.selectbox("Category", list(QUICK_FILTER.keys()))

if len(sel_athletes) < 2:
    st.warning("Select at least 2 athletes."); st.stop()

fdf = df[df["Name"].isin(sel_athletes)].copy()

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
t0, t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "🏐 Coach View",
    "Data Quality", "Metrics Analysis", "PCA & Clustering",
    "Performance Trends", "Athlete Profiles", "Summary Report",
    "About & References"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 0 — COACH VIEW
# ════════════════════════════════════════════════════════════════════════════════
with t0:
    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='margin-bottom:2px'>Squad Readiness</h2>"
        "<p style='color:#888;font-size:13px;margin-top:0'>Latest session · SWC-referenced · Hawkin Dynamics CMJ</p>",
        unsafe_allow_html=True)

    # Key metrics to surface in coach cards — auto-detected from loaded data
    CV_OUTPUT  = ["jump height", "mrsi", "rsi", "flight time", "takeoff velocity"]
    CV_ASYM    = ["l|r", "asym"]
    CV_DRIVER  = ["braking rfd", "propulsive net", "impulse ratio"]

    def cv_find(df, keywords):
        for c in df.columns:
            if df[c].dtype not in [np.float64, np.int64, np.float32]: continue
            if any(k in c.lower() for k in keywords) and df[c].notna().any():
                return c
        return None

    jh_col   = cv_find(fdf, ["jump height"])
    mrsi_col = cv_find(fdf, ["mrsi","m_rsi","modified"])
    rsi_col  = cv_find(fdf, ["rsi"]) if not mrsi_col else None
    asym_col = cv_find(fdf, ["l|r braking impulse","braking impulse asym"])
    rfd_col  = cv_find(fdf, ["braking rfd"])
    tv_col   = cv_find(fdf, ["takeoff velocity"])

    # ── Readiness colour logic ──────────────────────────────────────────────────
    def athlete_readiness(adf, all_athletes_df):
        """Return (status_label, color_hex, dot_color, flags_list, rec_text)."""
        red_flags, amber_flags = [], []

        def check(col, label, higher_is_better=True):
            if col is None or col not in adf.columns: return
            s = adf[col].dropna()
            if len(s) < 2: return
            last_v = s.iloc[-1]; bl = s.mean(); sw = swc(s)
            if sw == 0: return
            r = (last_v - bl) / sw
            direction = r if higher_is_better else -r
            if direction <= -1:
                red_flags.append(f"{label} ▼")
            elif direction <= -0.5:
                amber_flags.append(f"{label} ~")

        check(jh_col,   "Jump Height")
        check(mrsi_col, "mRSI")
        check(rsi_col,  "RSI")
        check(tv_col,   "Takeoff Velocity")
        check(rfd_col,  "Braking RFD")

        # Asymmetry flag — absolute value > threshold
        asym_flag = False
        asym_val  = None
        if asym_col and asym_col in adf.columns:
            last_asym = adf[asym_col].dropna()
            if not last_asym.empty:
                asym_val = abs(last_asym.iloc[-1])
                if asym_val >= 15:
                    red_flags.append(f"Asymmetry {asym_val:.1f}% ⚠")
                    asym_flag = True
                elif asym_val >= 10:
                    amber_flags.append(f"Asymmetry {asym_val:.1f}%")

        n_red = len(red_flags)
        if n_red >= 2 or asym_flag:
            status = "HIGH LOAD"
            bg     = "#FFF0F0"
            border = "#E74C3C"
            dot    = "#E74C3C"
        elif n_red == 1 or len(amber_flags) >= 2:
            status = "MONITOR"
            bg     = "#FFFBF0"
            border = "#F39C12"
            dot    = "#F39C12"
        elif amber_flags:
            status = "CAUTION"
            bg     = "#FFFBF0"
            border = "#F39C12"
            dot    = "#F39C12"
        else:
            status = "READY"
            bg     = "#F0FFF6"
            border = "#2ECC71"
            dot    = "#2ECC71"

        all_flags = red_flags + amber_flags
        flags_str = " · ".join(all_flags) if all_flags else "No flags"

        # Recommendation
        if status == "HIGH LOAD":
            rec = "Reduce jump volume and high-intensity work. Prioritise recovery before next session."
            if asym_flag:
                rec = f"Asymmetry at {asym_val:.1f}% — limit deceleration and COD tasks. " + rec
        elif status in ("MONITOR", "CAUTION"):
            rec = "Manageable load. Monitor during session. Reduce intensity if output drops further."
        else:
            rec = "Available for full training. No restrictions indicated."

        return status, bg, border, dot, flags_str, rec, asym_val

    # ── Latest metric values ────────────────────────────────────────────────────
    def latest_metric(adf, col):
        if col is None or col not in adf.columns: return None
        s = adf[col].dropna()
        return round(s.iloc[-1], 2) if not s.empty else None

    def flag_color_hex(adf, col, higher_is_better=True):
        if col is None or col not in adf.columns: return "#BDC3C7"
        s = adf[col].dropna()
        if len(s) < 2: return "#BDC3C7"
        last_v = s.iloc[-1]; bl = s.mean(); sw = swc(s)
        if sw == 0: return "#BDC3C7"
        r = (last_v - bl) / sw
        direction = r if higher_is_better else -r
        if direction >= 1:  return "#2ECC71"
        if direction <= -1: return "#E74C3C"
        if abs(direction) >= 0.5: return "#F39C12"
        return "#1A1A1A"

    # ── Sort athletes: High Load → Monitor → Caution → Ready ───────────────────
    STATUS_ORDER = {"HIGH LOAD": 0, "MONITOR": 1, "CAUTION": 2, "READY": 3}
    athlete_cards = []
    for a in sel_athletes:
        adf = fdf[fdf["Name"] == a].sort_values("Date")
        status, bg, border, dot, flags_str, rec, asym_val = athlete_readiness(adf, fdf)
        jh   = latest_metric(adf, jh_col)
        mrsi = latest_metric(adf, mrsi_col)
        tv   = latest_metric(adf, tv_col)
        asym = latest_metric(adf, asym_col)
        athlete_cards.append((STATUS_ORDER[status], a, status, bg, border,
                               dot, flags_str, rec, jh, mrsi, tv, asym, adf))
    athlete_cards.sort(key=lambda x: x[0])

    # ── Squad summary bar ───────────────────────────────────────────────────────
    counts = {"READY": 0, "CAUTION": 0, "MONITOR": 0, "HIGH LOAD": 0}
    for card in athlete_cards:
        counts[card[2]] += 1

    bar_cols = st.columns(4)
    status_cfg = [
        ("READY",     "#2ECC71", "✓ Ready",     "Full training. No restrictions."),
        ("CAUTION",   "#F39C12", "~ Caution",   "Monitor during session."),
        ("MONITOR",   "#F39C12", "⚠ Monitor",   "Reduce intensity or volume."),
        ("HIGH LOAD", "#E74C3C", "✕ High Load", "Limit high-intensity work."),
    ]
    for i, (s, col_hex, label, desc) in enumerate(status_cfg):
        with bar_cols[i]:
            st.markdown(
                f"<div style='text-align:center;padding:14px 8px;background:#F8F8F8;"
                f"border-top:3px solid {col_hex};border-radius:4px'>"
                f"<div style='font-size:28px;font-weight:700;color:{col_hex}'>{counts[s]}</div>"
                f"<div style='font-size:10px;font-weight:700;letter-spacing:.1em;"
                f"text-transform:uppercase;color:#555;margin-top:2px'>{label}</div>"
                f"<div style='font-size:10px;color:#888;margin-top:3px'>{desc}</div>"
                f"</div>", unsafe_allow_html=True)

    st.markdown("<div class='cv-section'>INDIVIDUAL ATHLETE STATUS</div>",
                unsafe_allow_html=True)

    # ── Athlete cards ───────────────────────────────────────────────────────────
    for (_, a, status, bg, border, dot, flags_str, rec,
         jh, mrsi, tv, asym, adf) in athlete_cards:

        jh_c   = flag_color_hex(adf, jh_col)
        mrsi_c = flag_color_hex(adf, mrsi_col)
        tv_c   = flag_color_hex(adf, tv_col)

        # Asymmetry bar visual
        asym_bar = ""
        if asym is not None:
            pct = min(abs(asym), 30)
            asym_c = "#E74C3C" if abs(asym) >= 15 else "#F39C12" if abs(asym) >= 10 else "#2ECC71"
            asym_bar = (
                f"<span class='cv-metric'>"
                f"<span class='cv-metric-label'>L|R Asymmetry</span>"
                f"<span class='cv-metric-val' style='color:{asym_c}'>{asym:+.1f}%</span>"
                f"</span>"
            )

        metrics_html = ""
        if jh is not None:
            metrics_html += (f"<span class='cv-metric'>"
                             f"<span class='cv-metric-label'>Jump Height</span>"
                             f"<span class='cv-metric-val' style='color:{jh_c}'>{jh} m</span>"
                             f"</span>")
        if mrsi is not None:
            metrics_html += (f"<span class='cv-metric'>"
                             f"<span class='cv-metric-label'>mRSI</span>"
                             f"<span class='cv-metric-val' style='color:{mrsi_c}'>{mrsi}</span>"
                             f"</span>")
        if tv is not None:
            metrics_html += (f"<span class='cv-metric'>"
                             f"<span class='cv-metric-label'>Takeoff Velocity</span>"
                             f"<span class='cv-metric-val' style='color:{tv_c}'>{tv} m/s</span>"
                             f"</span>")
        metrics_html += asym_bar

        st.markdown(
            f"<div class='cv-card' style='background:{bg};border-left:4px solid {border}'>"
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
            f"<div>"
            f"<div class='cv-name'>{a}</div>"
            f"<div class='cv-status' style='color:{dot}'>"
            f"<span class='cv-dot' style='background:{dot}'></span>{status}"
            f"</div>"
            f"</div>"
            f"<div style='text-align:right;font-size:11px;color:#888;max-width:50%'>"
            f"<b style='color:{dot}'>{flags_str}</b>"
            f"</div>"
            f"</div>"
            f"{metrics_html}"
            f"<div class='cv-rec'>"
            f"<div class='cv-rec-label'>Recommendation</div>"
            f"{rec}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True)

    # ── Jump height squad bar chart ─────────────────────────────────────────────
    if jh_col:
        st.markdown("<div class='cv-section'>SQUAD JUMP HEIGHT — LATEST VS BASELINE</div>",
                    unsafe_allow_html=True)
        bar_names, bar_latest, bar_baseline, bar_colors = [], [], [], []
        for (_, a, status, *rest, adf) in athlete_cards:
            s = adf[jh_col].dropna()
            if len(s) < 2: continue
            bar_names.append(a.split()[0])   # first name only
            bar_latest.append(round(s.iloc[-1], 3))
            bar_baseline.append(round(s.mean(), 3))
            bar_colors.append(
                "#E74C3C" if status == "HIGH LOAD" else
                "#F39C12" if status in ("MONITOR", "CAUTION") else "#2ECC71")

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Baseline (mean)",
            x=bar_names, y=bar_baseline,
            marker_color="#E8E8E8",
            text=[f"{v:.2f}" for v in bar_baseline],
            textposition="inside",
            textfont=dict(color="#888", size=10),
        ))
        fig_bar.add_trace(go.Bar(
            name="Latest",
            x=bar_names, y=bar_latest,
            marker_color=bar_colors,
            text=[f"{v:.2f}" for v in bar_latest],
            textposition="outside",
            textfont=dict(size=11, color="#1A1A1A"),
        ))
        fig_bar.update_layout(
            barmode="overlay",
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            height=260,
            margin=dict(l=8, r=8, t=10, b=8),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(size=10)),
            xaxis=dict(showgrid=False, tickfont=dict(size=11, family="Inter")),
            yaxis=dict(showgrid=True, gridcolor="#F0F0F0",
                       title="Jump Height (m)", tickfont=dict(size=9)),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<p style='font-size:10px;color:#aaa;text-align:center'>"
        "N1 Performance Lab · Coach View · SWC thresholds (Hopkins) · "
        "Red = ≥2 flags or asymmetry >15% · Amber = 1 flag or asymmetry 10–15%"
        "</p>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA QUALITY
# ════════════════════════════════════════════════════════════════════════════════
with t1:
    st.markdown("### Data Quality Dashboard")
    st.markdown('<p class="lbl">Completeness · Missing data · Quality scores</p>',
                unsafe_allow_html=True)

    all_metrics = [c for c in df.columns if c not in
                   {"Name","ExternalId","Test Type","Date","Time","BW [KG]",
                    "Reps","Tags","Additional Load [kg]"}]

    # Completeness matrix: athletes × metrics
    comp_rows = []
    for a in sel_athletes:
        adf = fdf[fdf["Name"] == a]
        row = {"Athlete": a, "Sessions": len(adf)}
        for m in all_metrics:
            pct = adf[m].notna().mean() * 100
            row[m] = round(pct, 0)
        row["Quality Score"] = round(
            np.mean([adf[m].notna().mean() for m in all_metrics if m in adf.columns]) * 100, 1
        )
        comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows)

    # KPIs
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        st.metric("Athletes selected", len(sel_athletes))
    with kc2:
        st.metric("Total sessions", len(fdf))
    with kc3:
        st.metric("Metrics available", len(all_metrics))
    with kc4:
        avg_q = comp_df["Quality Score"].mean()
        st.metric("Avg quality score", f"{avg_q:.1f}%",
                  delta="Good" if avg_q >= 80 else "Review")

    st.divider()

    # Heatmap: completeness per athlete × metric
    st.markdown("#### Data Completeness Heatmap")
    heat_metrics = [m for m in all_metrics if m in comp_df.columns][:30]
    z = comp_df[heat_metrics].values
    fig_heat = go.Figure(go.Heatmap(
        z=z, x=heat_metrics, y=comp_df["Athlete"].tolist(),
        colorscale=[[0,"#E74C3C"],[0.5,"#F39C12"],[1,"#2ECC71"]],
        zmin=0, zmax=100,
        text=[[f"{v:.0f}%" for v in row] for row in z],
        texttemplate="%{text}",
        colorbar=dict(title="Complete %", tickfont=dict(size=10)),
        hovertemplate="%{y} — %{x}<br>%{text}<extra></extra>",
    ))
    fig_heat.update_layout(
        plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
        height=max(300, len(sel_athletes)*28+60),
        margin=dict(l=8,r=8,t=20,b=8),
        xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10)),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Athletes with missing data
    st.markdown("#### Athletes with Missing Data")
    missing = comp_df[comp_df["Quality Score"] < 80][["Athlete","Sessions","Quality Score"]].copy()
    if missing.empty:
        st.success("All athletes have ≥80% data completeness.")
    else:
        st.dataframe(missing, use_container_width=True, hide_index=True)

    st.markdown("#### Full Quality Table")
    st.dataframe(comp_df[["Athlete","Sessions","Quality Score"]], use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — METRICS ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("### Metrics Analysis")
    st.markdown('<p class="lbl">Search · Filter · Compare across 100+ metrics</p>',
                unsafe_allow_html=True)

    num_cols = [c for c in fdf.columns if fdf[c].dtype in [np.float64, np.int64]
                and fdf[c].notna().any() and c != "BW [KG]"]

    # Quick filter + search
    mc1, mc2 = st.columns([2,1])
    with mc1:
        search = st.text_input("Search metric", placeholder="e.g. RFD, impulse, velocity")
    with mc2:
        cat_filter = st.selectbox("Category", list(QUICK_FILTER.keys()), key="m_cat")

    pool = QUICK_FILTER[cat_filter] if cat_filter != "All" else num_cols
    pool = [m for m in pool if m in num_cols]
    if search:
        pool = [m for m in pool if search.lower() in m.lower()]
    if not pool:
        pool = num_cols

    selected_metrics = st.multiselect(
        f"Select metrics to analyze ({len(pool)} available)",
        pool, default=pool[:4] if len(pool) >= 4 else pool
    )

    if not selected_metrics:
        st.info("Select at least one metric.")
    else:
        # Squad latest table
        st.markdown("#### Squad Comparison — Latest Session")
        rows = []
        for a in sel_athletes:
            adf = fdf[fdf["Name"]==a].sort_values("Date")
            last = adf.iloc[-1]
            row = {"Athlete": a}
            for m in selected_metrics:
                s = adf[m].dropna()
                if len(s) == 0: row[m] = "—"; continue
                val = last[m]
                bl, sw = s.mean(), swc(s)
                f = flag(val, bl, sw)
                row[m] = f"{val:.2f}"
                row[f"{m} Flag"] = FLAG_LABEL[f]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Distribution box plots
        st.markdown("#### Distribution Across Squad (All Sessions)")
        for m in selected_metrics:
            fig = go.Figure()
            for a in sel_athletes:
                vals = fdf[fdf["Name"]==a][m].dropna()
                if vals.empty: continue
                fig.add_trace(go.Box(
                    y=vals, name=a, boxpoints="all",
                    jitter=0.3, pointpos=-1.5,
                    line=dict(color=C["line"], width=1.5),
                    marker=dict(size=5, opacity=0.6),
                    hovertemplate=f"<b>{a}</b><br>{m}: %{{y:.2f}}<extra></extra>",
                ))
            fig.update_layout(
                title=dict(text=m, font=dict(size=12, family="Inter")),
                plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                height=300, margin=dict(l=8,r=8,t=36,b=8),
                xaxis=dict(tickfont=dict(size=9)),
                yaxis=dict(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=10)),
                font=dict(family="Inter"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — PCA & CLUSTERING
# ════════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown("### PCA & Clustering")
    st.markdown('<p class="lbl">Athlete similarity · Movement archetypes · K-means grouping</p>',
                unsafe_allow_html=True)

    pca_pool  = [c for c in fdf.columns if fdf[c].dtype in [np.float64, np.int64]
                 and fdf[c].notna().sum() >= len(sel_athletes) * 0.5 and c != "BW [KG]"]
    pca_mets  = st.multiselect("Metrics for PCA", pca_pool,
                               default=[m for m in (OUTPUT_M+DRIVER_M+STRATEGY_M) if m in pca_pool])

    if len(pca_mets) < 3:
        st.warning("Select at least 3 metrics for PCA.")
    elif len(sel_athletes) < 3:
        st.warning("Select at least 3 athletes.")
    else:
        # Build athlete matrix (mean per athlete)
        mat, names = [], []
        for a in sel_athletes:
            adf = fdf[fdf["Name"]==a][pca_mets].dropna(how="all")
            if adf.empty: continue
            row = [adf[m].mean() if m in adf.columns else np.nan for m in pca_mets]
            if sum(np.isnan(row)) <= len(pca_mets) * 0.3:
                # median impute
                row = [np.nanmedian(fdf[fdf["Name"].isin(sel_athletes)][m]) if np.isnan(v) else v
                       for v, m in zip(row, pca_mets)]
                mat.append(row); names.append(a)

        if len(mat) < 3:
            st.error("Not enough athletes with sufficient data.")
        else:
            X = np.array(mat)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            # Correlation heatmap
            with st.expander("Metric Correlation Matrix (validates PCA suitability)", expanded=False):
                corr = pd.DataFrame(Xs, columns=pca_mets).corr()
                fig_corr = go.Figure(go.Heatmap(
                    z=corr.values, x=pca_mets, y=pca_mets,
                    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                    text=corr.values.round(2),
                    texttemplate="%{text}",
                    colorbar=dict(title="r", tickfont=dict(size=9)),
                ))
                fig_corr.update_layout(
                    plot_bgcolor=C["bg"], paper_bgcolor=C["bg"], height=420,
                    margin=dict(l=8,r=8,t=20,b=8),
                    xaxis=dict(tickangle=-40, tickfont=dict(size=8)),
                    yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            # PCA
            n_comp = min(len(names), len(pca_mets), 5)
            pca    = PCA(n_components=n_comp)
            Xpca   = pca.fit_transform(Xs)
            var    = pca.explained_variance_ratio_

            # K-means
            k = min(n_clusters, len(names)-1)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = km.fit_predict(Xpca[:, :min(3, n_comp)])
            sil = silhouette_score(Xpca[:, :min(3, n_comp)], clusters) if k > 1 else 0

            COLORS = ["#1A1A1A","#E74C3C","#2ECC71","#F39C12","#3498DB","#9B59B6"]

            pc1, pc2 = st.columns(2)

            # PCA scatter
            with pc1:
                fig_pca = go.Figure()
                for ci in range(k):
                    mask = clusters == ci
                    fig_pca.add_trace(go.Scatter(
                        x=Xpca[mask, 0], y=Xpca[mask, 1],
                        mode="markers+text",
                        text=[names[i] for i in range(len(names)) if mask[i]],
                        textposition="top center",
                        textfont=dict(size=9),
                        marker=dict(color=COLORS[ci], size=12,
                                    line=dict(color=C["line"], width=1)),
                        name=f"Cluster {ci+1}",
                    ))
                fig_pca.update_layout(
                    title=dict(text=f"PCA — Athlete Similarity Map<br><sup>PC1 {var[0]:.1%} | PC2 {var[1]:.1%} | Silhouette {sil:.2f}</sup>",
                               font=dict(size=12, family="Inter")),
                    plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                    xaxis=dict(title=f"PC1 ({var[0]:.1%})", showgrid=True, gridcolor=C["grid"]),
                    yaxis=dict(title=f"PC2 ({var[1]:.1%})", showgrid=True, gridcolor=C["grid"]),
                    height=420, margin=dict(l=8,r=8,t=60,b=8),
                    legend=dict(font=dict(size=10)), font=dict(family="Inter"),
                )
                st.plotly_chart(fig_pca, use_container_width=True)

            # Feature importance
            with pc2:
                weights = pca.explained_variance_ratio_
                importance = (np.abs(pca.components_) * weights[:, np.newaxis]).sum(axis=0)
                imp_df = pd.DataFrame({"Metric": pca_mets, "Importance": importance})
                imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)
                fig_imp = go.Figure(go.Bar(
                    x=imp_df["Importance"], y=imp_df["Metric"],
                    orientation="h",
                    marker=dict(color=C["line"]),
                    text=imp_df["Importance"].round(3),
                    textposition="outside",
                ))
                fig_imp.update_layout(
                    title=dict(text="Feature Importance (weighted PCA loadings)",
                               font=dict(size=12, family="Inter")),
                    plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                    xaxis=dict(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=9)),
                    yaxis=dict(tickfont=dict(size=9)),
                    height=420, margin=dict(l=8,r=8,t=40,b=8),
                    font=dict(family="Inter"), showlegend=False,
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            # Scree plot
            st.markdown("#### Scree Plot — Variance Explained")
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(var))],
                y=var * 100,
                marker=dict(color=C["line"]),
                name="Individual",
            ))
            fig_scree.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(var))],
                y=np.cumsum(var) * 100,
                mode="lines+markers",
                line=dict(color=C["pos"], width=2),
                marker=dict(size=7),
                name="Cumulative",
                yaxis="y2",
            ))
            fig_scree.update_layout(
                plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                yaxis=dict(title="Variance (%)", showgrid=True, gridcolor=C["grid"]),
                yaxis2=dict(title="Cumulative (%)", overlaying="y", side="right",
                            range=[0,105], showgrid=False),
                height=300, margin=dict(l=8,r=8,t=20,b=8),
                legend=dict(font=dict(size=10)), font=dict(family="Inter"),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_scree, use_container_width=True)

            # Cluster membership table
            st.markdown("#### Cluster Assignments")
            cdf = pd.DataFrame({"Athlete": names, "Cluster": [f"Cluster {c+1}" for c in clusters]})
            st.dataframe(cdf, use_container_width=True, hide_index=True)
            buf = io.BytesIO(); cdf.to_csv(buf, index=False)
            st.download_button("Download Cluster CSV", buf.getvalue(),
                               "n1_pca_clusters.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — PERFORMANCE TRENDS
# ════════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("### Performance Trends")
    st.markdown('<p class="lbl">Longitudinal tracking · Statistical testing · Top improvers</p>',
                unsafe_allow_html=True)

    trend_pool = [c for c in fdf.columns if fdf[c].dtype in [np.float64, np.int64]
                  and fdf[c].notna().any() and c != "BW [KG]"]
    trend_met  = st.selectbox("Select metric", trend_pool,
                              index=trend_pool.index("Jump Height (Imp-Mom) [cm]")
                              if "Jump Height (Imp-Mom) [cm]" in trend_pool else 0)

    # Improvement heatmap (latest vs first session per athlete)
    st.markdown(f"#### {trend_met} — Improvement Heatmap (First → Latest)")
    imp_rows = []
    for a in sel_athletes:
        adf = fdf[fdf["Name"]==a].sort_values("Date")[trend_met].dropna()
        if len(adf) < 2: continue
        first, last_v = adf.iloc[0], adf.iloc[-1]
        delta = last_v - first
        pct   = delta / first * 100 if first != 0 else 0
        sw    = swc(adf)
        swc_d = delta / sw if sw > 0 else 0
        # Significance test (linear regression slope p-value)
        idx = np.arange(len(adf))
        slope, _, _, p, _ = stats.linregress(idx, adf.values)
        imp_rows.append({
            "Athlete": a, "First": round(first,2), "Latest": round(last_v,2),
            "Delta": round(delta,2), "Delta (%)": round(pct,1),
            "SWC Δ": round(swc_d,2), "p-value": round(p,3),
            "Trend": "↑ Sig" if slope>0 and p<0.05 else "↓ Sig" if slope<0 and p<0.05 else "~",
        })

    if imp_rows:
        imp_df = pd.DataFrame(imp_rows).sort_values("SWC Δ", ascending=False)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

        # Top improvers bar
        st.markdown("#### Top Improvers")
        top = imp_df.head(10)
        colors = [C["pos"] if v>=1 else C["warn"] if v>=0 else C["neg"]
                  for v in top["SWC Δ"]]
        fig_top = go.Figure(go.Bar(
            x=top["Athlete"], y=top["SWC Δ"],
            marker_color=colors,
            text=[f"{v:+.2f}" for v in top["SWC Δ"]],
            textposition="outside",
            hovertemplate="%{x}<br>SWC Δ: <b>%{y:+.2f}</b><extra></extra>",
        ))
        fig_top.add_hline(y=1, line_dash="dash", line_color=C["pos"], line_width=1,
                          annotation_text="≥1 SWC")
        fig_top.add_hline(y=-1, line_dash="dash", line_color=C["neg"], line_width=1)
        fig_top.update_layout(
            plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
            yaxis=dict(title="SWC Δ", showgrid=True, gridcolor=C["grid"]),
            xaxis=dict(showgrid=False),
            height=300, margin=dict(l=8,r=8,t=20,b=8),
            font=dict(family="Inter"), showlegend=False,
        )
        st.plotly_chart(fig_top, use_container_width=True)

    # Multi-athlete trend overlay
    st.markdown(f"#### {trend_met} — Squad Trend Overlay")
    fig_ov = go.Figure()
    for a in sel_athletes:
        adf = fdf[fdf["Name"]==a].sort_values("Date")
        vals = adf[trend_met].dropna()
        if vals.empty: continue
        fig_ov.add_trace(go.Scatter(
            x=adf["Date"], y=adf[trend_met],
            mode="lines+markers", name=a,
            line=dict(width=1.5), marker=dict(size=6),
            hovertemplate=f"<b>{a}</b><br>%{{x|%d %b}}: %{{y:.2f}}<extra></extra>",
        ))
    fig_ov.update_layout(
        plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=C["grid"], title=trend_met),
        height=360, margin=dict(l=8,r=8,t=20,b=8),
        legend=dict(font=dict(size=9)), font=dict(family="Inter"),
    )
    st.plotly_chart(fig_ov, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — ATHLETE PROFILES
# ════════════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown("### Athlete Profiles")
    st.markdown('<p class="lbl">Percentile rankings · Radar chart · Head-to-head</p>',
                unsafe_allow_html=True)

    if sel_athlete == "— Squad —":
        st.info("Select a single athlete from the sidebar.")
    else:
        adf = fdf[fdf["Name"]==sel_athlete].sort_values("Date")
        st.markdown(f"#### {sel_athlete} — Individual Report")

        # KPI row
        kpi_mets = [m for m in (OUTPUT_M + DRIVER_M) if m in adf.columns][:6]
        if not kpi_mets:
            # fallback: take first 6 numeric columns
            kpi_mets = [c for c in adf.columns if adf[c].dtype in [np.float64, np.int64]
                        and adf[c].notna().any()][:6]
        if not kpi_mets:
            st.info("No numeric metrics found for this athlete.")
        else:
            cols = st.columns(len(kpi_mets))
        for i, m in enumerate(kpi_mets):
            s = adf[m].dropna()
            if s.empty: continue
            last_v = s.iloc[-1]; bl = s.mean(); sw = swc(s)
            f = flag(last_v, bl, sw)
            col_str = FLAG_COLOR[f]
            with cols[i]:
                st.markdown(f"""
                <div class="kpi" style="border-left-color:{col_str}">
                    <div class="lbl">{m[:28]}</div>
                    <div style="font-size:20px;font-weight:700">{last_v:.2f}</div>
                    <div style="font-size:11px;color:{col_str}">Δ {last_v-bl:+.2f} | SWC ±{sw:.2f}</div>
                </div>""", unsafe_allow_html=True)

        # Percentile radar
        st.markdown("#### Percentile Profile vs Squad")
        radar_pool = [m for m in (OUTPUT_M + DRIVER_M + STRATEGY_M) if m in fdf.columns]
        radar_vals, radar_labels = [], []
        for m in radar_pool:
            s = fdf[m].dropna()
            last = adf[m].dropna()
            if s.empty or last.empty: continue
            pct = percentile_rank(last.iloc[-1], s)
            radar_vals.append(pct)
            radar_labels.append(m[:25])

        if radar_vals:
            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                fillcolor="rgba(26,26,26,0.12)",
                line=dict(color=C["line"], width=2),
                marker=dict(size=7),
                hovertemplate="%{theta}<br>Percentile: <b>%{r:.0f}</b><extra></extra>",
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0,100],
                                   tickfont=dict(size=9), gridcolor=C["grid"]),
                    angularaxis=dict(tickfont=dict(size=9)),
                    bgcolor=C["bg"],
                ),
                plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                height=420, margin=dict(l=40,r=40,t=20,b=20),
                font=dict(family="Inter"), showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Head-to-head comparison
        st.markdown("#### Head-to-Head Comparison")
        h2h_options = [a for a in sel_athletes if a != sel_athlete]
        if not h2h_options:
            st.info("Select additional athletes in the sidebar to enable head-to-head comparison.")
        else:
            opponent = st.selectbox("Compare against", h2h_options)
            odf = fdf[fdf["Name"]==opponent].sort_values("Date")
            h2h_pool = [m for m in radar_pool if m in adf.columns and m in odf.columns]
            h2h_rows = []
            for m in h2h_pool:
                av = adf[m].dropna(); ov = odf[m].dropna()
                if av.empty or ov.empty: continue
                a_last, o_last = av.iloc[-1], ov.iloc[-1]
                win = sel_athlete if a_last > o_last else opponent if o_last > a_last else "Tie"
                h2h_rows.append({
                    "Metric": m, sel_athlete: round(a_last,2), opponent: round(o_last,2),
                    "Δ": round(a_last-o_last,2), "Winner": win
                })
            if h2h_rows:
                h2h_df = pd.DataFrame(h2h_rows)
                st.dataframe(h2h_df, use_container_width=True, hide_index=True)

        # Trend section
        st.markdown("#### Longitudinal Trends")
        all_trend = [m for m in (OUTPUT_M + DRIVER_M + STRATEGY_M + ASYM_M) if m in adf.columns]
        c1, c2 = st.columns(2)
        for i, m in enumerate(all_trend):
            fig = trend_fig(adf, m)
            if fig:
                with (c1 if i%2==0 else c2):
                    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — SUMMARY REPORT
# ════════════════════════════════════════════════════════════════════════════════
with t6:
    st.markdown("### Summary Report")
    st.markdown('<p class="lbl">Overall statistics · Insights · Export all results</p>',
                unsafe_allow_html=True)

    # Overall stats
    st.markdown("#### Overall Statistics")
    sum_mets = [m for m in (OUTPUT_M + DRIVER_M) if m in fdf.columns]
    sum_rows = []
    for m in sum_mets:
        s = fdf[m].dropna()
        if s.empty: continue
        sum_rows.append({
            "Metric": m, "N": len(s),
            "Mean": round(s.mean(),2), "SD": round(s.std(),2),
            "Min": round(s.min(),2), "Max": round(s.max(),2),
            "CV (%)": round(cv_pct(s),1), "SWC": round(swc(s),2),
        })
    sum_df = pd.DataFrame(sum_rows)
    st.dataframe(sum_df, use_container_width=True, hide_index=True)

    st.divider()

    # SWC flag summary per athlete
    st.markdown("#### Squad Readiness Summary — Latest vs Baseline")
    key_mets = [m for m in (OUTPUT_M + DRIVER_M) if m in fdf.columns]
    read_rows = []
    for a in sel_athletes:
        adf = fdf[fdf["Name"]==a].sort_values("Date")
        row = {"Athlete": a}
        flags = []
        for m in key_mets:
            s = adf[m].dropna()
            if len(s) < 2: continue
            last_v = s.iloc[-1]; bl = s.mean(); sw = swc(s)
            f = flag(last_v, bl, sw)
            flags.append(f)
            row[m[:20]] = f"{last_v:.2f} ({FLAG_LABEL[f]})"
        n_red = flags.count("r"); n_grn = flags.count("g")
        row["Overall"] = "⚠ Monitor" if n_red >= 2 else "▲ Elevated" if n_grn >= 2 else "Stable"
        read_rows.append(row)
    read_df = pd.DataFrame(read_rows)
    st.dataframe(read_df, use_container_width=True, hide_index=True)

    st.divider()

    # Exports
    st.markdown("#### Export All Results")
    ec1, ec2, ec3 = st.columns(3)

    # 1 — SWC summary
    swc_rows = []
    for a in sel_athletes:
        adf = fdf[fdf["Name"]==a].sort_values("Date")
        for m in [c for c in fdf.columns if fdf[c].dtype in [np.float64,np.int64]]:
            s = adf[m].dropna()
            if len(s) < 2: continue
            last_v = s.iloc[-1]; bl = s.mean(); sw = swc(s); cv_v = cv_pct(s)
            f = flag(last_v, bl, sw)
            swc_rows.append({
                "Athlete":a,"Metric":m,"Latest":round(last_v,3),
                "Baseline":round(bl,3),"SWC":round(sw,3),
                "CV (%)":round(cv_v,2) if not np.isnan(cv_v) else None,
                "Delta":round(last_v-bl,3),"Flag":FLAG_LABEL[f],
            })
    swc_csv = pd.DataFrame(swc_rows).to_csv(index=False)
    with ec1:
        st.download_button("SWC Summary CSV", swc_csv, "n1_swc_summary.csv", "text/csv")

    # 2 — Overall stats
    with ec2:
        st.download_button("Overall Stats CSV", sum_df.to_csv(index=False),
                           "n1_overall_stats.csv", "text/csv")

    # 3 — Cleaned raw
    with ec3:
        st.download_button("Cleaned Raw CSV", fdf.to_csv(index=False),
                           "n1_cleaned_data.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — ABOUT & REFERENCES
# ════════════════════════════════════════════════════════════════════════════════
with t7:
    st.markdown("### About This Dashboard")
    st.markdown(
        "The N1 Force Plate Dashboard is a web-based analysis platform built exclusively "
        "for **Hawkin Dynamics** CSV exports. Upload your data, select your focus, and move "
        "from raw numbers to training decisions — in under ten minutes."
    )

    st.divider()

    # ── 10-Minute Workflow ──────────────────────────────────────────────────────
    st.markdown("#### The 10-Minute Workflow")
    steps = [
        ("1. Upload your Hawkin Dynamics CSV export",
         "Drag and drop your export file. The dashboard auto-detects column names — no formatting required."),
        ("2. Select your analysis focus",
         "Use the category filter (Output, Driver, Strategy, Asymmetry, Power) or search by keyword to narrow to the metrics that matter for your current block."),
        ("3. Choose optimal cluster count",
         "The app suggests a silhouette score for each K — pick the cluster count that produces meaningful, coaching-relevant groups."),
        ("4. Review PCA variance explanation",
         "The scree plot shows how much of your data story is captured by each principal component. Aim for ≥70% cumulative variance before drawing conclusions."),
        ("5. Explore athlete groupings",
         "Interactive scatter plots let you hover, zoom, and discuss clusters in team meetings without leaving the browser."),
        ("6. Export insights",
         "Download cluster assignments, SWC summaries, or cleaned raw data — ready for program adjustment or longitudinal tracking."),
    ]
    for title, desc in steps:
        with st.expander(title):
            st.markdown(desc)

    st.divider()

    # ── Longitudinal Integration ────────────────────────────────────────────────
    st.markdown("#### Making It Stick in Your Workflow")
    st.markdown("""
- **Monthly cluster reviews** during periodization planning — track whether your squad profile is shifting as intended.
- **New athlete assessments** — benchmark incoming athletes against existing clusters immediately.
- **Progress tracking** — watch athletes migrate between clusters as they develop; cluster migration is your objective evidence of adaptation.
- **Real-time discussions** — pull up the app during team meetings to ground coaching decisions in data.
""")

    st.divider()

    # ── Performance Tracking ────────────────────────────────────────────────────
    st.markdown("#### Performance Tracking Over Time")
    st.markdown("""
The **Performance Trends** tab provides:

| Feature | Description |
|---|---|
| Trend analysis | Is your training moving athletes in the right direction? |
| Cluster migration | Watch athletes develop and move between performance groups. |
| Program effectiveness | Validate coaching interventions with objective SWC-referenced data. |
| Individual trajectories | Track specific athlete development patterns session by session. |
""")

    st.divider()

    # ── Technical Integration ───────────────────────────────────────────────────
    st.markdown("#### Technical Integration")
    st.markdown("""
- **Platform:** Streamlit — secure, web-based access
- **File limit:** 200 MB (~two years of standard force plate data)
- **Compatible systems:** Hawkin Dynamics only (other vendors organise columns differently)
- **Export:** SWC summary, overall stats, and cleaned raw CSV available on the Summary tab
- **Direct integration:** Contact N1 Performance Lab for automated data ingestion options
""")

    st.divider()

    # ── FAQs ───────────────────────────────────────────────────────────────────
    st.markdown("#### FAQs")
    faqs = [
        ("What is Principal Component Analysis?",
         "PCA is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional form while preserving as much variance as possible. It reveals which combinations of metrics best separate your athletes."),
        ("What is K-Means Clustering?",
         "K-Means is an unsupervised machine learning algorithm that groups data points based on inherent similarity — no labels required. For example, it might identify 'explosive-power athletes', 'impulse-dominant athletes', and 'strategy-reliant athletes' without being told those categories exist."),
        ("What exactly am I looking at in the PCA plot?",
         "Each dot is one athlete. Dots close together share similar force plate profiles. Athletes in the same coloured cluster share characteristics that could inform shared training blocks."),
        ("How many clusters should I use?",
         "Start with 3–4 for most teams. The silhouette score (shown in the plot title) quantifies cluster separation — higher is better. Use your coaching eye too: if fast and slow athletes land in the same cluster, try one more cluster."),
        ("What if my best athlete lands in the 'worst' cluster?",
         "Clusters are not rankings — they are profiles. Elite athletes sometimes succeed through technique or factors the force plate does not capture. Use the grouping as a discussion point, not a verdict."),
        ("How much data do I need?",
         "Minimum 10 athletes; 20+ is better for reliable patterns. For longitudinal analysis, at least 3–4 testing sessions per athlete."),
        ("If an athlete moves between clusters over time, is that good or bad?",
         "It depends on your training goals. Moving toward your target performance profile is positive. Use cluster migration to track whether programming is pushing athletes in the intended direction."),
        ("Can I use data from other force plate systems?",
         "Currently Hawkin Dynamics only — other systems organise columns differently so uploads will not parse correctly. Contact N1 Performance Lab to discuss expanding compatibility."),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.markdown(a)

    st.divider()

    # ── References ─────────────────────────────────────────────────────────────
    st.markdown("#### References")
    refs = [
        "Stone, J. D., Merrigan, J. J., Ramadan, J., Brown, R. S., Cheng, G. T., Hornsby, W. G., ... & Hagen, J. A. (2022). Simplifying external load data in NCAA Division-I men's basketball competitions: A principal component analysis. *Frontiers in Sports and Active Living*, 4, 795897.",
        "Parmar, N., James, N., Hearne, G., & Jones, B. (2018). Using principal component analysis to develop performance indicators in professional rugby league. *International Journal of Performance Analysis in Sport*, 18(6), 938–949.",
        "Vagner, M., Cleather, D. J., Kubový, P., Hojka, V., & Stastny, P. (2022). Principal component analysis can be used to discriminate between elite and sub-elite kicking performance. *Motor Control*, 27(2), 354–372.",
        "Bazmara, M., & Jafari, S. (2013). K nearest neighbor algorithm for finding soccer talent. *Journal of Basic and Applied Scientific Research*, 3(4), 981–986.",
        "Shelly, Z., Burch, R. F., Tian, W., Strawderman, L., Piroli, A., & Bichey, C. (2020). Using K-means clustering to create training groups for elite American football student-athletes based on game demands. *International Journal of Kinesiology and Sports Science*, 8(2), 47–63.",
        "GeeksforGeeks. (n.d.). K-Means Clustering — Introduction. https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/",
        "StatQuest with Josh Starmer. (2018). StatQuest: K-means clustering [Video]. YouTube. https://www.youtube.com/watch?v=FgakZw6K1QQ",
    ]
    for i, ref in enumerate(refs, 1):
        st.markdown(f"{i}. {ref}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="font-size:10px;color:#888;text-align:right">'
    'N1 Performance Lab · Force Plate Dashboard v3 · '
    'ODS Framework (Lake) · SWC (Hopkins) · PCA (Merrigan) · '
    'Compatible with Hawkin Dynamics exports</p>',
    unsafe_allow_html=True
)
