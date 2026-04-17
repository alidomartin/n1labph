"""
N1 + Predictive Engine Integration
====================================
ENGINE 1 — N1 V10 Biomechanical Forensics
  Detects mechanical compensation from force-time curve signatures.
  Output: STABLE | COMPENSATING

ENGINE 2 — Load-Probability (Zone7-style)
  Estimates systemic burnout probability from SWC flag accumulation.
  Output: LOW RISK | MODERATE RISK | HIGH RISK

ENGINE 3 — HARMONIZER
  Combines both engines into a single operational classification.
  Output: READY | MAINTENANCE REQUIRED | TECHNICAL INTERVENTION | CRITICAL: MANDATORY REST
"""

from dataclasses import dataclass, field
from typing import Optional
import re


# ── Parsed metric container ────────────────────────────────────────────────────

@dataclass
class Metric:
    name:      str
    value:     Optional[float]
    direction: str   # "up" | "down" | "neutral"


def parse_cell(cell) -> tuple[Optional[float], str]:
    """Extract (value, direction) from Hawkin summary export cell."""
    if cell is None or str(cell).strip() == "":
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


# ── Athlete profile ────────────────────────────────────────────────────────────

@dataclass
class AthleteProfile:
    name: str
    # Output metrics
    jump_height:      Metric = None
    flight_time:      Metric = None
    mrsi:             Metric = None
    rsi:              Metric = None
    takeoff_velocity: Metric = None
    # Braking / eccentric
    braking_rfd:      Metric = None
    braking_net_imp:  Metric = None
    time_to_takeoff:  Metric = None
    braking_impulse:  Metric = None
    # Propulsive / concentric
    prop_net_imp:     Metric = None
    impulse_ratio:    Metric = None
    p1p2_ratio:       Metric = None
    # Asymmetry
    lr_braking_imp:   Metric = None
    lr_prop_imp:      Metric = None
    lr_landing_imp:   Metric = None


def build_profile(name: str, row: dict) -> AthleteProfile:
    """Build AthleteProfile from a Hawkin summary export row dict."""

    def find(keys: list[str]) -> Optional[Metric]:
        for col, val in row.items():
            cl = col.lower().strip()
            if any(k in cl for k in keys):
                v, d = parse_cell(val)
                return Metric(name=col, value=v, direction=d)
        return None

    p = AthleteProfile(name=name)
    p.jump_height      = find(["jump height"])
    p.flight_time      = find(["flight time"])
    p.mrsi             = find(["mrsi", "m_rsi", "modified rsi"])
    p.rsi              = find(["rsi"]) if not p.mrsi else None
    p.takeoff_velocity = find(["takeoff velocity"])
    p.braking_rfd      = find(["braking rfd"])
    p.braking_net_imp  = find(["braking net impulse"])
    p.time_to_takeoff  = find(["time to takeoff"])
    p.braking_impulse  = find(["braking impulse", "braking imp"])
    p.prop_net_imp     = find(["propulsive net"])
    p.impulse_ratio    = find(["impulse ratio"])
    p.p1p2_ratio       = find(["p1|p2", "p1p2"])
    p.lr_braking_imp   = find(["l|r braking impulse", "braking impulse "])
    p.lr_prop_imp      = find(["l|r propulsive"])
    p.lr_landing_imp   = find(["l|r landing"])
    return p


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 1 — N1 V10 BIOMECHANICAL FORENSICS
# Force-displacement loop signature → mechanical compensation detection
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ForensicsResult:
    status:       str            # "STABLE" | "COMPENSATING"
    severity:     str            # "None" | "Minor" | "Moderate" | "Severe"
    indicators:   list[str] = field(default_factory=list)
    loop_pattern: str = ""       # Description of the F-D loop signature
    score:        int = 0


COMP_THRESHOLDS = {
    "asymmetry_flag":   10.0,   # % L|R imbalance triggers compensation flag
    "asymmetry_severe": 15.0,   # % L|R imbalance triggers severe
}


def run_forensics(p: AthleteProfile) -> ForensicsResult:
    """
    Identify mechanical compensation from CMJ force-time signatures.

    Compensation patterns detected:
    1. P1 Phase Dominance   — P1|P2 ▲ with output ▼ → front-loading to compensate late-phase deficit
    2. Extended Brake Phase — TTT ▲ + braking RFD ▼ → slower, less forceful eccentric loading
    3. Over-Brake Pattern   — braking impulse ▲ + propulsive net ▼ → energy lost in braking
    4. Bilateral Asymmetry  — L|R > threshold → unilateral load redistribution
    5. SSC Efficiency Loss  — mRSI ▼ + output maintained → reduced elastic energy return
    """
    indicators = []
    score = 0

    def is_up(m):   return m is not None and m.direction == "up"
    def is_down(m): return m is not None and m.direction == "down"
    def asym_val(m):
        if m is None or m.value is None: return 0.0
        return abs(m.value)

    # ── Pattern 1: P1 Phase Dominance ─────────────────────────────────────────
    if is_up(p.p1p2_ratio) and is_down(p.prop_net_imp):
        indicators.append(
            "P1 Phase Dominance: Early propulsive phase is disproportionately elevated "
            "while net propulsive impulse is below baseline — athlete front-loading takeoff "
            "effort to compensate for late-phase force production deficit."
        )
        score += 2

    # ── Pattern 2: Extended Brake Phase ───────────────────────────────────────
    if is_up(p.time_to_takeoff) and is_down(p.braking_rfd):
        indicators.append(
            "Extended Brake Phase: Time to takeoff is elevated while braking RFD is "
            "suppressed — countermovement is slower and less forceful than baseline. "
            "Athlete spending more time in the eccentric phase without generating "
            "greater peak eccentric force."
        )
        score += 2

    # ── Pattern 3: Over-Brake Pattern ─────────────────────────────────────────
    if is_up(p.braking_impulse) and is_down(p.prop_net_imp):
        indicators.append(
            "Over-Brake Pattern: Braking impulse is above baseline while propulsive "
            "net impulse is suppressed — excess energy absorbed in the braking phase "
            "is not being recycled into propulsive output. Indicative of reduced "
            "stretch-shortening cycle transfer efficiency."
        )
        score += 2

    # ── Pattern 4: Bilateral Asymmetry Compensation ───────────────────────────
    av_brake = asym_val(p.lr_braking_imp)
    av_prop  = asym_val(p.lr_prop_imp)
    av_land  = asym_val(p.lr_landing_imp)

    if av_brake >= COMP_THRESHOLDS["asymmetry_severe"]:
        indicators.append(
            f"Critical Braking Asymmetry: L|R Braking Impulse at {p.lr_braking_imp.value:+.1f}% "
            f"exceeds the 15% clinical threshold — one limb is absorbing disproportionate "
            f"braking load, indicating unilateral mechanical compensation under fatigue."
        )
        score += 3
    elif av_brake >= COMP_THRESHOLDS["asymmetry_flag"]:
        indicators.append(
            f"Braking Asymmetry Flag: L|R Braking Impulse at {p.lr_braking_imp.value:+.1f}% "
            f"is at the clinical monitoring threshold — emerging unilateral load redistribution."
        )
        score += 1

    if av_prop >= COMP_THRESHOLDS["asymmetry_severe"]:
        indicators.append(
            f"Critical Propulsive Asymmetry: L|R Propulsive Impulse at {p.lr_prop_imp.value:+.1f}% "
            f"exceeds threshold — propulsive phase is driven asymmetrically."
        )
        score += 2
    elif av_prop >= COMP_THRESHOLDS["asymmetry_flag"]:
        indicators.append(
            f"Propulsive Asymmetry Flag: L|R Propulsive at {p.lr_prop_imp.value:+.1f}%."
        )
        score += 1

    if av_land >= COMP_THRESHOLDS["asymmetry_flag"]:
        indicators.append(
            f"Landing Asymmetry: L|R Landing Impulse at {p.lr_landing_imp.value:+.1f}% "
            f"— asymmetric ground contact on landing."
        )
        score += 1

    # ── Pattern 5: SSC Efficiency Loss ────────────────────────────────────────
    reactive = p.mrsi or p.rsi
    if is_down(reactive) and not is_down(p.jump_height):
        indicators.append(
            "SSC Efficiency Loss: Reactive strength index is below baseline while jump "
            "height is maintained — athlete is generating similar output through a slower, "
            "more effort-intensive strategy rather than elastic energy return. "
            "Compensation is masking the underlying neuromuscular deficit."
        )
        score += 1

    # ── Classify ───────────────────────────────────────────────────────────────
    if score == 0:
        status   = "STABLE"
        severity = "None"
        loop_pattern = (
            "Force-displacement loop signature is within normal variation. "
            "Braking and propulsive phase mechanics are consistent with baseline."
        )
    elif score <= 2:
        status   = "COMPENSATING"
        severity = "Minor"
        loop_pattern = (
            "Minor loop distortion detected. One compensation pattern present — "
            "athlete is beginning to alter movement strategy to maintain output."
        )
    elif score <= 4:
        status   = "COMPENSATING"
        severity = "Moderate"
        loop_pattern = (
            "Moderate loop distortion. Multiple compensation patterns active — "
            "movement strategy has shifted materially from baseline. "
            "Output may appear maintained but at a higher mechanical cost."
        )
    else:
        status   = "COMPENSATING"
        severity = "Severe"
        loop_pattern = (
            "Severe loop distortion. Force-displacement profile shows significant "
            "deviation from baseline across multiple phases. Injury risk is elevated "
            "under continued high-load exposure."
        )

    return ForensicsResult(
        status=status, severity=severity,
        indicators=indicators, loop_pattern=loop_pattern, score=score
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 2 — LOAD-PROBABILITY (Zone7-style)
# Systemic burnout probability from SWC accumulation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PredictiveResult:
    risk_level:  str          # "LOW RISK" | "MODERATE RISK" | "HIGH RISK"
    probability: int          # 0–100 burnout probability estimate
    score:       int
    drivers:     list[str] = field(default_factory=list)
    narrative:   str = ""


def run_predictive(p: AthleteProfile) -> PredictiveResult:
    """
    Zone7-style load-probability check.
    Weights metrics by their neuromuscular fatigue sensitivity.

    Weights (evidence-based):
      mRSI:             3  — most sensitive to neuromuscular fatigue (Gathercole et al.)
      Jump Height:      2  — primary output marker
      Takeoff Velocity: 2  — direct force-velocity output
      RSI:              2  — reactive strength capacity
      Braking RFD:      2  — eccentric force production rate
      Prop. Net Imp:    1  — concentric phase output
      Impulse Ratio:    1  — phase balance
      Flight Time:      1  — secondary output
    """
    score = 0
    drivers = []

    def check(m, label, weight, higher_is_better=True):
        nonlocal score
        if m is None: return
        bad_dir = "down" if higher_is_better else "up"
        if m.direction == bad_dir:
            score += weight
            drivers.append(f"{label} ({'▼' if bad_dir == 'down' else '▲'}  −{weight}pt)")

    check(p.mrsi,             "mRSI",             3)
    check(p.jump_height,      "Jump Height",       2)
    check(p.takeoff_velocity, "Takeoff Velocity",  2)
    check(p.rsi,              "RSI",               2)
    check(p.braking_rfd,      "Braking RFD",       2)
    check(p.prop_net_imp,     "Propulsive Net Imp",1)
    check(p.impulse_ratio,    "Impulse Ratio",     1)
    check(p.flight_time,      "Flight Time",       1)

    # Max possible score = 14 (all weighted metrics below SWC)
    probability = min(int((score / 14) * 100), 100)

    if score <= 3:
        risk_level = "LOW RISK"
        narrative = (
            "Systemic load indicators are within acceptable range. "
            "No significant burnout signal detected from neuromuscular output metrics."
        )
    elif score <= 7:
        risk_level = "MODERATE RISK"
        narrative = (
            "Multiple neuromuscular output metrics below baseline. "
            "Systemic fatigue is accumulating. Monitor training load closely "
            "and avoid adding high-intensity exposure."
        )
    else:
        risk_level = "HIGH RISK"
        narrative = (
            "High systemic burnout probability. Output and driver metrics are "
            "broadly suppressed — this is a whole-system fatigue response, "
            "not an isolated deficit. Continued high-intensity training "
            "at this state significantly elevates injury probability."
        )

    return PredictiveResult(
        risk_level=risk_level, probability=probability,
        score=score, drivers=drivers, narrative=narrative
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 3 — THE HARMONIZER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HarmonizedResult:
    classification: str       # Final operational label
    priority:       int       # 0=Ready, 1=Technical, 2=Maintenance, 3=Critical
    color:          str       # Hex colour for UI
    icon:           str
    action:         str       # One-line coaching action
    rationale:      str       # Why this classification


_CLASSIFICATIONS = {
    "READY": dict(
        priority=0, color="#2ECC71", icon="✅",
        action="Clear for full training. Standard session load applies.",
        rationale="Both biomechanical and systemic indicators are within normal range."
    ),
    "MAINTENANCE REQUIRED": dict(
        priority=2, color="#3498DB", icon="🔧",
        action="Reduce volume and intensity. Prioritise recovery modalities.",
        rationale=(
            "High systemic fatigue detected, but movement mechanics are sound. "
            "The athlete is tired, not broken — recovery is the intervention, "
            "not a technique correction."
        )
    ),
    "TECHNICAL INTERVENTION": dict(
        priority=1, color="#F39C12", icon="⚙️",
        action="Reduce load and address movement pattern with targeted technical work.",
        rationale=(
            "Systemic load is manageable, but compensation patterns in the "
            "force-displacement signature indicate a faulty or degraded movement strategy. "
            "This is a skill problem, not a fatigue problem. "
            "Fresh athletes with compensation patterns can entrench poor mechanics "
            "under continued high-load exposure."
        )
    ),
    "CRITICAL: MANDATORY REST": dict(
        priority=3, color="#E74C3C", icon="🚨",
        action="MANDATORY REST. No high-intensity training. Medical/recovery staff to be notified.",
        rationale=(
            "Both systemic fatigue AND mechanical compensation are present simultaneously. "
            "The athlete is fatigued AND has altered their movement strategy to compensate — "
            "the highest-risk combination for acute injury under continued load."
        )
    ),
}


def run_harmonizer(
    forensics: ForensicsResult,
    predictive: PredictiveResult
) -> HarmonizedResult:
    """
    THE HARMONIZER
    Applies the N1 + Predictive Engine decision matrix.

    Matrix:
      N1=STABLE    + Predictive=LOW RISK      → READY
      N1=STABLE    + Predictive=MODERATE RISK → MAINTENANCE REQUIRED
      N1=STABLE    + Predictive=HIGH RISK     → MAINTENANCE REQUIRED
      N1=COMP(any) + Predictive=LOW RISK      → TECHNICAL INTERVENTION
      N1=COMP(any) + Predictive=MODERATE RISK → CRITICAL: MANDATORY REST
      N1=COMP(any) + Predictive=HIGH RISK     → CRITICAL: MANDATORY REST
    """
    n1_comp  = forensics.status == "COMPENSATING"
    pred_low = predictive.risk_level == "LOW RISK"
    pred_high = predictive.risk_level == "HIGH RISK"
    pred_mod  = predictive.risk_level == "MODERATE RISK"

    if not n1_comp and pred_low:
        label = "READY"
    elif not n1_comp and (pred_mod or pred_high):
        label = "MAINTENANCE REQUIRED"
    elif n1_comp and pred_low:
        label = "TECHNICAL INTERVENTION"
    else:  # n1_comp AND (mod or high predictive risk)
        label = "CRITICAL: MANDATORY REST"

    cfg = _CLASSIFICATIONS[label]
    return HarmonizedResult(
        classification=label,
        priority=cfg["priority"],
        color=cfg["color"],
        icon=cfg["icon"],
        action=cfg["action"],
        rationale=cfg["rationale"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE — single entry point
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EngineOutput:
    name:       str
    profile:    AthleteProfile
    forensics:  ForensicsResult
    predictive: PredictiveResult
    harmonized: HarmonizedResult


def run_pipeline(name: str, row: dict) -> EngineOutput:
    """Run all three engines for one athlete row."""
    profile    = build_profile(name, row)
    forensics  = run_forensics(profile)
    predictive = run_predictive(profile)
    harmonized = run_harmonizer(forensics, predictive)
    return EngineOutput(
        name=name,
        profile=profile,
        forensics=forensics,
        predictive=predictive,
        harmonized=harmonized,
    )
