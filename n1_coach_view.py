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
import os, re
from datetime import datetime

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

# ── Parse each athlete ─────────────────────────────────────────────────────────
STATUS_ORDER = {"HIGH LOAD": 0, "MONITOR": 1, "CAUTION": 2, "READY": 3}

def classify_athlete(row):
    """Return status, colour, recommendation based on SWC flags in the row."""
    red_count, amber_count = 0, 0
    flags = []
    asym_val = None
    asym_flag = False

    def check(col, label, higher_is_better=True):
        nonlocal red_count, amber_count
        if col is None or col not in row.index: return
        val, direction = parse_val(row[col])
        if val is None: return
        is_bad = (direction == "down") if higher_is_better else (direction == "up")
        if is_bad:
            red_count += 1
            flags.append(f"{label} ▼")

    check(jh_col,   "Jump Height")
    check(mrsi_col, "mRSI")
    check(rsi_col,  "RSI")
    check(tv_col,   "Takeoff Velocity")

    # Asymmetry — absolute threshold
    if asym_col and asym_col in row.index:
        val, _ = parse_val(row[asym_col])
        if val is not None:
            asym_val = val
            if abs(val) >= 15:
                flags.append(f"Asymmetry {val:+.1f}% ⚠")
                asym_flag = True
                red_count += 1
            elif abs(val) >= 10:
                flags.append(f"Asymmetry {val:+.1f}%")
                amber_count += 1

    if red_count >= 2 or asym_flag:
        status = "HIGH LOAD"
        color  = "#E74C3C"
    elif red_count == 1 or amber_count >= 2:
        status = "MONITOR"
        color  = "#F39C12"
    elif amber_count >= 1:
        status = "CAUTION"
        color  = "#F39C12"
    else:
        status = "READY"
        color  = "#2ECC71"

    # Recommendation
    if status == "HIGH LOAD":
        if asym_flag:
            rec = (f"Asymmetry at {asym_val:+.1f}% — limit deceleration and change-of-direction "
                   f"tasks. Reduce jump volume. Prioritise recovery.")
        else:
            rec = "Multiple output metrics below baseline. Reduce intensity and jump volume. Recovery focus."
    elif status in ("MONITOR", "CAUTION"):
        rec = "One or more metrics below baseline. Manage load during session. Reduce intensity if fatigue shows."
    else:
        rec = "No significant flags. Available for full training."

    flag_str = " · ".join(flags) if flags else "No flags"
    return status, color, flag_str, rec, asym_val

cards = []
for _, row in df.iterrows():
    name = str(row[name_col]).strip()
    if not name: continue
    status, color, flag_str, rec, asym_val = classify_athlete(row)

    jh_val,   jh_dir   = parse_val(row[jh_col])   if jh_col   else (None, "neutral")
    mrsi_val, mrsi_dir = parse_val(row[mrsi_col])  if mrsi_col else (None, "neutral")
    tv_val,   tv_dir   = parse_val(row[tv_col])    if tv_col   else (None, "neutral")

    cards.append(dict(name=name, status=status, color=color,
                      flag_str=flag_str, rec=rec, asym_val=asym_val,
                      jh=(jh_val, jh_dir), mrsi=(mrsi_val, mrsi_dir),
                      tv=(tv_val, tv_dir)))

cards.sort(key=lambda x: STATUS_ORDER[x["status"]])

# ── Summary counts ─────────────────────────────────────────────────────────────
counts = {s: sum(1 for c in cards if c["status"] == s)
          for s in ["HIGH LOAD", "MONITOR", "CAUTION", "READY"]}

cfg = [
    ("HIGH LOAD", "#E74C3C", "✕ High Load"),
    ("MONITOR",   "#F39C12", "⚠ Monitor"),
    ("CAUTION",   "#F39C12", "~ Caution"),
    ("READY",     "#2ECC71", "✓ Ready"),
]
cols = st.columns(4)
for i, (s, hex_c, lbl) in enumerate(cfg):
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

for c in cards:
    border = c["color"]
    badge_bg = c["color"]

    # Metric HTML
    metrics_html = ""
    for label, (val, direction), unit in [
        ("Jump Height", c["jh"],   " m"),
        ("mRSI",        c["mrsi"], ""),
        ("Takeoff Vel", c["tv"],   " m/s"),
    ]:
        if val is None: continue
        vc = DIR_COLOR[direction]
        arrow = DIR_ARROW[direction]
        metrics_html += f"""
        <div class="metric-box">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="--vc:{vc}">{val}{unit}</div>
          <div class="metric-flag" style="--vc:{vc}">{arrow} {'Above' if direction=='up' else 'Below' if direction=='down' else 'Stable'}</div>
        </div>"""

    # Asymmetry chip
    asym_html = ""
    if c["asym_val"] is not None:
        av = c["asym_val"]
        ac = "#E74C3C" if abs(av) >= 15 else "#F39C12" if abs(av) >= 10 else "#2ECC71"
        asym_html = f'<div class="asym-chip" style="--ac:{ac}">L|R {av:+.1f}%</div>'

    # First name only for display
    first_name = c["name"].split()[0].title()
    last_name  = " ".join(c["name"].split()[1:]).title()

    st.markdown(f"""
    <div class="card" style="--border:{border}">
      <div class="card-top">
        <div class="card-name">{first_name} <span style="font-weight:400;color:#888">{last_name}</span></div>
        <div class="card-badge" style="--border:{badge_bg}">{c['status']}</div>
      </div>
      <div style="font-size:11px;color:#999;margin-bottom:6px">{c['flag_str']}</div>
      <div class="metrics-row">{metrics_html}</div>
      {asym_html}
      <div class="rec-box">
        <div class="rec-label">What this means for today</div>
        {c['rec']}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Jump height bar chart ──────────────────────────────────────────────────────
if any(c["jh"][0] is not None for c in cards):
    st.markdown('<div class="section-label">Jump Height — Squad Overview</div>',
                unsafe_allow_html=True)
    names  = [c["name"].split()[0].title() for c in cards if c["jh"][0] is not None]
    values = [c["jh"][0] for c in cards if c["jh"][0] is not None]
    colors = [c["color"] for c in cards if c["jh"][0] is not None]

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
        font=dict(family="Inter"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-size:10px;color:#ccc;margin-top:28px;padding-top:16px;
border-top:1px solid #EEEEEE">
N1 Performance Lab · Hawkin Dynamics CMJ · SWC Method (Hopkins)<br>
🔴 High Load = 2+ flags or asymmetry &gt;15% &nbsp;·&nbsp;
🟡 Monitor = 1 flag or asymmetry 10–15% &nbsp;·&nbsp;
🟢 Ready = clear
</div>
""", unsafe_allow_html=True)
