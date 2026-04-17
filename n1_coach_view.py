"""
N1 Performance Lab — Coach View
Standalone squad readiness app for coaching staff.
Reads from 01_Raw_Data/latest_squad.csv (Hawkin Dynamics summary export format).
Run: streamlit run n1_coach_view.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, re, sys
from datetime import datetime

# Engine import — works both locally and on Streamlit Cloud
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "02_Logic"))
from n1_predictive_engine import run_pipeline, EngineOutput

st.set_page_config(
    page_title="Squad Readiness · N1",
    page_icon="🏐",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#F9F9F9;color:#1A1A1A}

.header-band{
    background:#1A1A1A;color:#fff;
    padding:22px 28px 16px 28px;
    border-radius:10px;margin-bottom:20px
}
.header-title{font-size:22px;font-weight:800;letter-spacing:-.02em;margin-bottom:2px}
.header-sub{font-size:12px;color:#aaa;font-weight:400}

.summary-grid{display:flex;gap:10px;margin-bottom:20px}
.summary-box{
    flex:1;text-align:center;padding:14px 8px;
    background:#fff;border-radius:8px;
    border-top:3px solid var(--c);
    box-shadow:0 1px 4px rgba(0,0,0,.06)
}
.summary-num{font-size:30px;font-weight:800;color:var(--c)}
.summary-lbl{font-size:10px;font-weight:700;letter-spacing:.1em;
             text-transform:uppercase;color:#777;margin-top:2px}

.card{
    background:#fff;border-radius:10px;
    border-left:5px solid var(--border);
    padding:18px 20px 14px 20px;
    margin-bottom:10px;
    box-shadow:0 1px 6px rgba(0,0,0,.06)
}
.card-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.card-name{font-size:18px;font-weight:800;letter-spacing:-.01em}
.card-badge{
    font-size:10px;font-weight:800;letter-spacing:.1em;
    text-transform:uppercase;padding:4px 10px;
    border-radius:20px;color:#fff;background:var(--border)
}
.metrics-row{display:flex;gap:16px;flex-wrap:wrap;margin:10px 0}
.metric-box{min-width:80px}
.metric-label{font-size:10px;color:#999;font-weight:600;
              letter-spacing:.06em;text-transform:uppercase;margin-bottom:2px}
.metric-value{font-size:18px;font-weight:800;color:var(--vc)}
.metric-flag{font-size:9px;font-weight:700;color:var(--vc);margin-top:1px}
.asym-chip{
    display:inline-block;font-size:11px;font-weight:700;
    padding:3px 10px;border-radius:12px;
    background:var(--ac);color:#fff;margin-top:6px
}
.rec-box{
    margin-top:12px;padding-top:12px;
    border-top:1px solid #F0F0F0;
    font-size:12px;color:#444;line-height:1.55
}
.rec-label{font-size:9px;font-weight:800;letter-spacing:.12em;
           text-transform:uppercase;color:#bbb;margin-bottom:3px}
.section-label{
    font-size:10px;font-weight:800;letter-spacing:.14em;
    text-transform:uppercase;color:#aaa;
    margin:22px 0 10px 2px
}
.morning-note{
    background:#FFF8E1;border-left:3px solid #F39C12;
    padding:10px 14px;border-radius:6px;
    font-size:11px;color:#7A4F00;line-height:1.55;
    margin-bottom:20px
}
footer-note{font-size:10px;color:#bbb;text-align:center;margin-top:24px}
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "01_Raw_Data/latest_squad.csv")

@st.cache_data(ttl=300)
def load_summary(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Drop blank/squad-level rows
    name_col = df.columns[0]
    df = df[df[name_col].str.strip().ne("") & df[name_col].notna()].copy()
    df[name_col] = df[name_col].str.strip()
    df = df[df[name_col] != ""].reset_index(drop=True)
    return df

def parse_val(cell):
    """Extract numeric value from 'X.XX (▲ Above SWC)' format."""
    if pd.isna(cell) or str(cell).strip() == "":
        return None, "neutral"
    cell = str(cell)
    match = re.match(r"^\s*([+-]?\d+\.?\d*)", cell)
    val = float(match.group(1)) if match else None
    if "▲" in cell or "Above" in cell:
        direction = "up"
    elif "▼" in cell or "Below" in cell:
        direction = "down"
    else:
        direction = "neutral"
    return val, direction

def find_col(df, keywords):
    """Find first column matching any keyword (case-insensitive)."""
    for c in df.columns:
        cl = c.lower().strip()
        if any(k in cl for k in keywords):
            return c
    return None

if not os.path.exists(DATA_PATH):
    st.error("No squad data found. Ask your performance analyst to update the dashboard.")
    st.stop()

df = load_summary(DATA_PATH)
name_col = df.columns[0]
athletes  = df[name_col].tolist()

# Column detection
jh_col    = find_col(df, ["jump height"])
mrsi_col  = find_col(df, ["mrsi", "m_rsi", "modified rsi"])
rsi_col   = find_col(df, ["rsi"]) if not mrsi_col else None
tv_col    = find_col(df, ["takeoff velocity"])
asym_col  = find_col(df, ["l|r braking impulse", "braking impulse "])
overall_col = df.columns[1] if len(df.columns) > 1 else None

# File modification time → "last updated"
mod_time = datetime.fromtimestamp(os.path.getmtime(DATA_PATH))
last_updated = mod_time.strftime("%d %B %Y · %H:%M")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-band">
  <div class="header-title">🏐 Squad Readiness</div>
  <div class="header-sub">NU Men's Volleyball · Force Plate Assessment · Updated {last_updated}</div>
</div>
""", unsafe_allow_html=True)

# Morning note
st.markdown("""
<div class="morning-note">
  <b>⏱ Morning pre-session data.</b>
  Jump height and mRSI may read slightly lower than usual due to time-of-day effects.
  Asymmetry flags are the most reliable signal — treat those as priority.
</div>
""", unsafe_allow_html=True)

# ── Run integrated engine pipeline ────────────────────────────────────────────
PRIORITY_ORDER = {
    "CRITICAL: MANDATORY REST": 0,
    "MAINTENANCE REQUIRED":     1,
    "TECHNICAL INTERVENTION":   2,
    "READY":                    3,
}

engine_outputs: list[EngineOutput] = []
for _, row in df.iterrows():
    name = str(row[name_col]).strip()
    if not name: continue
    row_dict = row.to_dict()
    out = run_pipeline(name, row_dict)
    engine_outputs.append(out)

engine_outputs.sort(key=lambda x: PRIORITY_ORDER[x.harmonized.classification])

# ── Summary counts ─────────────────────────────────────────────────────────────
count_cfg = [
    ("CRITICAL: MANDATORY REST", "#E74C3C", "🚨 Critical"),
    ("MAINTENANCE REQUIRED",     "#3498DB", "🔧 Maintenance"),
    ("TECHNICAL INTERVENTION",   "#F39C12", "⚙️ Technical"),
    ("READY",                    "#2ECC71", "✅ Ready"),
]
counts = {s: sum(1 for o in engine_outputs if o.harmonized.classification == s)
          for s, _, _ in count_cfg}

cols = st.columns(4)
for i, (s, hex_c, lbl) in enumerate(count_cfg):
    with cols[i]:
        st.markdown(f"""
        <div class="summary-box" style="--c:{hex_c}">
          <div class="summary-num">{counts[s]}</div>
          <div class="summary-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

# ── Athlete cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Individual Status</div>', unsafe_allow_html=True)

DIR_COLOR = {"up": "#2ECC71", "down": "#E74C3C", "neutral": "#1A1A1A"}
DIR_ARROW = {"up": "▲", "down": "▼", "neutral": "—"}

for out in engine_outputs:
    h  = out.harmonized
    f  = out.forensics
    pr = out.predictive
    p  = out.profile
    border = h.color

    # Key metrics
    metrics_html = ""
    for label, metric, unit in [
        ("Jump Height", p.jump_height,      " m"),
        ("mRSI",        p.mrsi,             ""),
        ("Takeoff Vel", p.takeoff_velocity,  " m/s"),
    ]:
        if metric is None or metric.value is None: continue
        vc    = DIR_COLOR[metric.direction]
        arrow = DIR_ARROW[metric.direction]
        direction_lbl = "Above" if metric.direction == "up" else "Below" if metric.direction == "down" else "Stable"
        metrics_html += f"""
        <div class="metric-box">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="--vc:{vc}">{metric.value}{unit}</div>
          <div class="metric-flag" style="--vc:{vc}">{arrow} {direction_lbl}</div>
        </div>"""

    # Asymmetry chip
    asym_html = ""
    if p.lr_braking_imp and p.lr_braking_imp.value is not None:
        av = p.lr_braking_imp.value
        ac = "#E74C3C" if abs(av) >= 15 else "#F39C12" if abs(av) >= 10 else "#2ECC71"
        asym_html = f'<div class="asym-chip" style="--ac:{ac}">L|R Braking {av:+.1f}%</div>'

    # Engine badges
    n1_color   = "#E74C3C" if f.status == "COMPENSATING" else "#2ECC71"
    pred_color = ("#E74C3C" if pr.risk_level == "HIGH RISK"
                  else "#F39C12" if pr.risk_level == "MODERATE RISK" else "#2ECC71")

    engine_html = f"""
    <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
      <span style="font-size:10px;font-weight:700;padding:3px 9px;border-radius:10px;
                   background:{n1_color}22;color:{n1_color};border:1px solid {n1_color}44">
        N1: {f.status} ({f.severity})
      </span>
      <span style="font-size:10px;font-weight:700;padding:3px 9px;border-radius:10px;
                   background:{pred_color}22;color:{pred_color};border:1px solid {pred_color}44">
        LOAD: {pr.risk_level} · {pr.probability}%
      </span>
    </div>"""

    first_name = out.name.split()[0].title()
    last_name  = " ".join(out.name.split()[1:]).title()

    st.markdown(f"""
    <div class="card" style="--border:{border}">
      <div class="card-top">
        <div class="card-name">{first_name} <span style="font-weight:400;color:#888">{last_name}</span></div>
        <div class="card-badge" style="--border:{border}">{h.icon} {h.classification}</div>
      </div>
      <div class="metrics-row">{metrics_html}</div>
      {asym_html}
      {engine_html}
      <div class="rec-box">
        <div class="rec-label">Action</div>
        {h.action}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Expandable detail for coaches who want more
    with st.expander(f"Why — {first_name} {last_name}"):
        st.markdown(f"**Harmonizer Rationale**\n\n{h.rationale}")
        if f.indicators:
            st.markdown("**Biomechanical Flags (N1 V10)**")
            for ind in f.indicators:
                st.markdown(f"- {ind}")
        st.markdown(f"**Load-Probability Drivers**\n\n{pr.narrative}")
        if pr.drivers:
            st.markdown(" · ".join(pr.drivers))

# ── Jump height bar chart ──────────────────────────────────────────────────────
jh_data = [(out.name.split()[0].title(), out.profile.jump_height, out.harmonized.color)
           for out in engine_outputs
           if out.profile.jump_height and out.profile.jump_height.value is not None]

if jh_data:
    st.markdown('<div class="section-label">Jump Height — Squad Overview</div>',
                unsafe_allow_html=True)
    names  = [d[0] for d in jh_data]
    values = [d[1].value for d in jh_data]
    colors = [d[2] for d in jh_data]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors,
        text=[f"{v:.2f} m" for v in values],
        textposition="outside",
        textfont=dict(size=11, family="Inter"),
    ))
    fig.update_layout(
        plot_bgcolor="#fff", paper_bgcolor="#F9F9F9",
        height=220, margin=dict(l=4, r=4, t=8, b=4),
        xaxis=dict(showgrid=False, tickfont=dict(size=12, family="Inter")),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0",
                   title="Jump Height (m)", tickfont=dict(size=9),
                   range=[0, max(values) * 1.18]),
        font=dict(family="Inter"), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-size:10px;color:#ccc;margin-top:28px;padding-top:16px;
border-top:1px solid #EEEEEE">
N1 Performance Lab · Hawkin Dynamics CMJ · SWC Method (Hopkins)<br>
🚨 Critical = N1 Compensating + High Load &nbsp;·&nbsp;
🔧 Maintenance = Stable + High Load &nbsp;·&nbsp;
⚙️ Technical = Compensating + Low Load &nbsp;·&nbsp;
✅ Ready = Stable + Low Load
</div>
""", unsafe_allow_html=True)
