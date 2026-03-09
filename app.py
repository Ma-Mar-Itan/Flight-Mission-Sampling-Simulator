"""
Flight Mission & Sampling Simulator — Streamlit Edition
=========================================================
Realistic drone survey simulation demonstrating how mission parameters
affect coverage quality, reconstruction fidelity, and statistical bias
in agricultural remote sensing.

Run:
    streamlit run app.py

Architecture
------------
Section 1 — Synthetic Field Generator
Section 2 — Mission Geometry  (lawnmower path, footprint, GSD)
Section 3 — Drone Motion Engine (interpolation, capture scheduling)
Section 4 — Battery Model (time/turn/wind cost, cutoff logic)
Section 5 — Reconstruction Engine (footprint masking, overlap degradation)
Section 6 — Statistics Engine (zonal stats, sampling-bias comparison)
Section 7 — Plotly Figure Builders
Section 8 — Streamlit UI
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter

# =============================================================================
# CONSTANTS — physically interpretable, no magic numbers
# =============================================================================

FIELD_ACRES      = 4.0
FIELD_SIDE_M     = math.sqrt(FIELD_ACRES * 4046.86)   # ≈ 127.6 m per side
RASTER_ROWS      = 128
RASTER_COLS      = 128
CELL_SIZE_M      = FIELD_SIDE_M / RASTER_COLS          # ≈ 0.997 m / pixel

# Camera model — Sony RX1R II equivalent (35 mm fixed lens)
SENSOR_WIDTH_MM  = 35.9
FOCAL_LENGTH_MM  = 35.0
IMAGE_WIDTH_PX   = 8000
HFOV_DEG         = 2 * math.degrees(math.atan(SENSOR_WIDTH_MM / (2 * FOCAL_LENGTH_MM)))

# Battery power model
POWER_PER_SECOND  = 0.055   # % per second nominal hover + cruise
TURN_PENALTY_PCT  = 0.8     # % per lane reversal
WIND_FACTOR_SCALE = 0.03    # extra % per second per wind unit

# Overlap quality threshold
OVERLAP_IDEAL = 0.80        # below this, reconstruction gaps appear

# Quadrant zone labels
ZONE_NAMES = {0: "NW", 1: "NE", 2: "SW", 3: "SE"}


# =============================================================================
# SECTION 1 — SYNTHETIC FIELD GENERATOR
# =============================================================================

def _smooth_noise(rows: int, cols: int, scale: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Gaussian-smoothed random field normalised to [0, 1]."""
    raw      = rng.random((rows, cols)).astype(np.float32)
    smoothed = gaussian_filter(raw, sigma=scale)
    lo, hi   = smoothed.min(), smoothed.max()
    return (smoothed - lo) / (hi - lo + 1e-9)


@st.cache_data
def generate_field_raster(seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate synthetic 4-acre agricultural field rasters.

    Cached by Streamlit so the field is only computed once per session.

    Returns dict with keys:
      vari  — Visible Atmospherically Resistant Index  [float32, -0.3…0.6]
      ndvi  — Normalised Difference Vegetation Index   [float32, -1…1]
      dem   — Digital Elevation Model, relative (m)    [float32, 0…8]
      rgb   — Natural-colour orthomosaic               [uint8, H×W×3]
      zones — Quadrant labels 0-3                      [int32]
    """
    rng = np.random.default_rng(seed)

    # DEM: rolling terrain with east–west slope
    base_terrain = _smooth_noise(RASTER_ROWS, RASTER_COLS, scale=20.0, rng=rng)
    ew_slope     = np.linspace(0, 1, RASTER_COLS)[np.newaxis, :]
    dem          = (0.6 * base_terrain + 0.4 * ew_slope) * 8.0

    # Vegetation index (VARI proxy)
    crop_base  = _smooth_noise(RASTER_ROWS, RASTER_COLS, scale=12.0, rng=rng)
    micro_var  = _smooth_noise(RASTER_ROWS, RASTER_COLS, scale=6.0,  rng=rng)

    # Stressed patch — simulates disease or waterlogging
    cx, cy     = int(RASTER_ROWS * 0.35), int(RASTER_COLS * 0.60)
    rr, cc     = np.ogrid[:RASTER_ROWS, :RASTER_COLS]
    stress_map = np.exp(-((rr - cx)**2 + (cc - cy)**2) / (2 * 15**2))
    vari_raw   = crop_base - 0.5 * micro_var - 0.6 * stress_map
    vari       = np.clip(vari_raw / (vari_raw.max() + 1e-9),
                         -0.3, 0.6).astype(np.float32)

    # NDVI — correlated with VARI but independently noisy
    ndvi = np.clip(vari * 1.2 + 0.1 + rng.normal(0, 0.03, vari.shape),
                   -1.0, 1.0).astype(np.float32)

    # RGB — approximate natural colour derived from VARI
    vari_01 = (vari - vari.min()) / (vari.max() - vari.min() + 1e-9)
    r_ch    = np.clip(0.35 + 0.25 * (1 - vari_01) + 0.04 * rng.random(vari.shape), 0, 1)
    g_ch    = np.clip(0.28 + 0.40 * vari_01        + 0.04 * rng.random(vari.shape), 0, 1)
    b_ch    = np.clip(0.10 + 0.10 * vari_01        + 0.04 * rng.random(vari.shape), 0, 1)
    rgb     = (np.stack([r_ch, g_ch, b_ch], axis=-1) * 255).astype(np.uint8)

    # Zone labels (NW=0, NE=1, SW=2, SE=3)
    zones           = np.zeros((RASTER_ROWS, RASTER_COLS), dtype=np.int32)
    h2, w2          = RASTER_ROWS // 2, RASTER_COLS // 2
    zones[:h2, w2:] = 1
    zones[h2:, :w2] = 2
    zones[h2:, w2:] = 3

    return {
        "vari": vari, "ndvi": ndvi,
        "dem":  dem.astype(np.float32),
        "rgb":  rgb, "zones": zones,
    }


# =============================================================================
# SECTION 2 — MISSION GEOMETRY
# =============================================================================

def compute_footprint_radius_px(altitude_m: float) -> float:
    """
    Ground footprint radius in raster pixels.
    footprint_half_width = altitude × tan(HFOV/2)
    Higher altitude → larger radius → coarser coverage per photo.
    """
    half_width_m = altitude_m * math.tan(math.radians(HFOV_DEG / 2.0))
    return half_width_m / CELL_SIZE_M


def compute_gsd_cm(altitude_m: float) -> float:
    """
    Ground Sampling Distance in cm/pixel.
    GSD = (sensor_mm × altitude_m × 100) / (focal_mm × image_px)
    Higher altitude → larger GSD → coarser spatial detail.
    """
    return (SENSOR_WIDTH_MM * altitude_m * 100.0) / (FOCAL_LENGTH_MM * IMAGE_WIDTH_PX)


def compute_line_spacing_px(altitude_m: float, overlap_frac: float) -> float:
    """
    Inter-strip spacing in pixels.
    spacing = footprint_diameter × (1 − overlap_frac)
    Lower overlap → wider spacing → more gaps between strips.
    """
    diameter_px = 2.0 * compute_footprint_radius_px(altitude_m)
    return max(2.0, diameter_px * (1.0 - overlap_frac))


def generate_lawnmower_path(
    altitude_m: float,
    overlap_frac: float,
    margin_px: float = 4.0,
) -> List[Tuple[float, float]]:
    """
    Generate a boustrophedon (lawnmower / serpentine) survey path.

    Runs north–south strips, alternating direction on each pass.
    Returns list of (col, row) waypoints in raster pixel coordinates.
    """
    spacing = compute_line_spacing_px(altitude_m, overlap_frac)
    col_min = margin_px
    col_max = RASTER_COLS - margin_px
    row_min = margin_px
    row_max = RASTER_ROWS - margin_px

    waypoints: List[Tuple[float, float]] = []
    col      = col_min
    go_south = True

    while col <= col_max + spacing * 0.5:
        col_c = min(col, col_max)
        if go_south:
            waypoints.append((col_c, row_min))
            waypoints.append((col_c, row_max))
        else:
            waypoints.append((col_c, row_max))
            waypoints.append((col_c, row_min))
        go_south = not go_south
        col      += spacing

    return waypoints


def count_turns(waypoints: List[Tuple[float, float]]) -> int:
    """Count lane-reversal turn events (end of each north–south strip)."""
    turns = 0
    for i in range(1, len(waypoints) - 1):
        dy_prev = waypoints[i][1]   - waypoints[i - 1][1]
        dy_next = waypoints[i + 1][1] - waypoints[i][1]
        if abs(dy_prev) > 0.1 and abs(dy_next) < 0.1:
            turns += 1
    return max(turns, 0)


def compute_estimated_overlap(fp_radius_px: float,
                               line_spacing_px: float) -> float:
    """Actual overlap fraction from footprint size and line spacing."""
    diameter = 2.0 * fp_radius_px
    return float(np.clip(1.0 - line_spacing_px / max(diameter, 1e-9), 0.0, 1.0))


# =============================================================================
# SECTION 3 — DRONE MOTION ENGINE
# =============================================================================

def compute_segment_lengths(
    waypoints: List[Tuple[float, float]]
) -> List[float]:
    """Euclidean pixel-length of each consecutive waypoint segment."""
    return [
        math.hypot(waypoints[i + 1][0] - waypoints[i][0],
                   waypoints[i + 1][1] - waypoints[i][1])
        for i in range(len(waypoints) - 1)
    ]


def total_path_length_m(waypoints: List[Tuple[float, float]]) -> float:
    """Total planned path length in metres."""
    return sum(compute_segment_lengths(waypoints)) * CELL_SIZE_M


def mission_duration_s(waypoints: List[Tuple[float, float]],
                       speed_ms: float) -> float:
    """Estimated full mission duration in seconds."""
    return total_path_length_m(waypoints) / max(speed_ms, 0.1)


def interpolate_position(
    waypoints: List[Tuple[float, float]],
    seg_lengths: List[float],
    progress: float,
) -> Tuple[float, float]:
    """
    Linear interpolation of drone position along the waypoint path.

    progress : float [0, 1] — fraction of total path length completed.
    Returns (col, row) in raster pixel coordinates.
    """
    if not waypoints:
        return (0.0, 0.0)
    total = sum(seg_lengths)
    if total == 0:
        return waypoints[0]

    target = progress * total
    accum  = 0.0

    for i, seg_len in enumerate(seg_lengths):
        if accum + seg_len >= target or i == len(seg_lengths) - 1:
            t   = (target - accum) / max(seg_len, 1e-9)
            t   = min(max(t, 0.0), 1.0)
            col = waypoints[i][0] + t * (waypoints[i + 1][0] - waypoints[i][0])
            row = waypoints[i][1] + t * (waypoints[i + 1][1] - waypoints[i][1])
            return (col, row)
        accum += seg_len

    return waypoints[-1]


def schedule_capture_events(
    waypoints: List[Tuple[float, float]],
    seg_lengths: List[float],
    speed_ms: float,
    interval_s: float,
) -> List[Dict[str, float]]:
    """
    Schedule photo capture events at fixed time intervals along the path.

    Returns list of dicts: {t, col, row, progress}
    """
    px_per_s   = speed_ms / CELL_SIZE_M
    total_px   = sum(seg_lengths)
    total_time = total_px / max(px_per_s, 1e-9)

    events: List[Dict] = []
    t = 0.0
    while t <= total_time + 1e-6:
        progress = min((t * px_per_s) / max(total_px, 1e-9), 1.0)
        col, row = interpolate_position(waypoints, seg_lengths, progress)
        events.append({"t": round(t, 3), "col": col, "row": row,
                        "progress": progress})
        t += interval_s

    return events


# =============================================================================
# SECTION 4 — BATTERY MODEL
# =============================================================================

def compute_battery_budget(
    mission_time_s: float,
    n_turns: int,
    wind_factor: float,
    battery_capacity_pct: float = 100.0,
) -> Dict[str, Any]:
    """
    Estimate battery consumption and mission feasibility.

    Formula:
      base_cost = mission_time_s × POWER_PER_SECOND
      turn_cost = n_turns        × TURN_PENALTY_PCT
      wind_cost = mission_time_s × WIND_FACTOR_SCALE × wind_factor
      used_pct  = base_cost + turn_cost + wind_cost

    If used_pct > battery_capacity_pct, fraction_possible is back-solved
    and the mission is truncated proportionally.
    """
    per_sec   = POWER_PER_SECOND + WIND_FACTOR_SCALE * wind_factor
    base_cost = mission_time_s * POWER_PER_SECOND
    turn_cost = n_turns        * TURN_PENALTY_PCT
    wind_cost = mission_time_s * WIND_FACTOR_SCALE * wind_factor
    used_pct  = base_cost + turn_cost + wind_cost
    feasible  = used_pct <= battery_capacity_pct

    if feasible:
        fraction_possible = 1.0
    else:
        available = max(0.0, battery_capacity_pct - turn_cost)
        max_time  = available / max(per_sec, 1e-9)
        fraction_possible = min(1.0, max_time / max(mission_time_s, 1e-9))

    return {
        "used_pct":          round(used_pct, 2),
        "available_pct":     battery_capacity_pct,
        "mission_feasible":  feasible,
        "fraction_possible": round(fraction_possible, 4),
        "base_cost_pct":     round(base_cost, 2),
        "turn_cost_pct":     round(turn_cost, 2),
        "wind_cost_pct":     round(wind_cost, 2),
    }


def apply_battery_cutoff(
    capture_events: List[Dict],
    fraction_possible: float,
) -> List[Dict]:
    """Discard capture events beyond the battery-limited path fraction."""
    return [ev for ev in capture_events
            if ev["progress"] <= fraction_possible + 1e-6]


# =============================================================================
# SECTION 5 — RECONSTRUCTION ENGINE
# =============================================================================

def build_coverage_mask(
    capture_events: List[Dict],
    footprint_radius_px: float,
    overlap_frac: float,
    rng_seed: int = 0,
) -> np.ndarray:
    """
    Build a boolean coverage mask from completed capture events.

    Each capture paints a soft circular footprint (weight 1→0 centre→edge).
    The accumulated weight map is thresholded to binary.

    When overlap_frac < OVERLAP_IDEAL, stochastic column-aligned gaps simulate
    photogrammetric reconstruction failure from insufficient image overlap.

    Returns bool array shape (RASTER_ROWS × RASTER_COLS).
    """
    weight = np.zeros((RASTER_ROWS, RASTER_COLS), dtype=np.float32)
    r      = max(footprint_radius_px, 1.0)
    rr, cc = np.ogrid[:RASTER_ROWS, :RASTER_COLS]

    for ev in capture_events:
        dist2 = (rr - ev["row"])**2 + (cc - ev["col"])**2
        fp_w  = np.clip(1.0 - dist2 / (r**2), 0.0, 1.0)
        weight = np.maximum(weight, fp_w)

    mask = weight > 0.05

    # Overlap-gap degradation
    if overlap_frac < OVERLAP_IDEAL and mask.any():
        rng          = np.random.default_rng(rng_seed + 7)
        gap_severity = (OVERLAP_IDEAL - overlap_frac) / OVERLAP_IDEAL
        gap_noise    = rng.random((RASTER_ROWS, RASTER_COLS)).astype(np.float32)
        col_band     = (
            np.abs(np.sin(np.linspace(0, math.pi * 10, RASTER_COLS)))
            * gap_severity
        )[np.newaxis, :]
        gap_holes = gap_noise < (col_band * 0.75)
        mask      = mask & ~gap_holes

    return mask


def compute_coverage_fraction(mask: np.ndarray) -> float:
    """Fraction of field pixels with at least one capture footprint."""
    return float(mask.sum()) / float(mask.size)


# =============================================================================
# SECTION 6 — STATISTICS ENGINE
# =============================================================================

def compute_zonal_stats(
    vari: np.ndarray,
    zones: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Per-quadrant zonal statistics for VARI.

    mask=None  → truth mode (all pixels used)
    mask=array → sampled mode (captured pixels only)
    """
    results = []
    for zone_id, zone_name in ZONE_NAMES.items():
        zone_px = zones == zone_id
        active  = zone_px & mask if mask is not None else zone_px
        vals    = vari[active]
        if len(vals) == 0:
            results.append({"Zone": zone_name, "Mean": "N/A", "Std": "N/A",
                             "Min": "N/A", "Max": "N/A", "Pixels": 0})
        else:
            results.append({
                "Zone":   zone_name,
                "Mean":   round(float(vals.mean()), 4),
                "Std":    round(float(vals.std()),  4),
                "Min":    round(float(vals.min()),  4),
                "Max":    round(float(vals.max()),  4),
                "Pixels": int(len(vals)),
            })
    return results


def compute_sampling_bias(
    vari: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Compare true field mean VARI vs sampled (captured-only) mean.
    Quantifies statistical bias introduced by incomplete coverage.
    """
    true_vals    = vari.ravel()
    sampled_vals = vari[mask].ravel() if mask.any() else np.array([])

    true_mean    = float(true_vals.mean())
    sampled_mean = (float(sampled_vals.mean())
                    if len(sampled_vals) > 0 else None)

    if sampled_mean is None:
        return {
            "true_mean": round(true_mean, 4), "sampled_mean": None,
            "absolute_bias": None, "relative_bias": None,
            "n_sampled_px": 0, "n_total_px": int(vari.size),
        }

    bias = sampled_mean - true_mean
    return {
        "true_mean":     round(true_mean,    4),
        "sampled_mean":  round(sampled_mean, 4),
        "absolute_bias": round(bias,         4),
        "relative_bias": round(bias / (abs(true_mean) + 1e-9) * 100, 2),
        "n_sampled_px":  int(mask.sum()),
        "n_total_px":    int(vari.size),
    }


def compute_efficiency_score(coverage_fraction: float,
                              battery_used_pct: float) -> float:
    """
    Efficiency Score = (coverage % / battery used %).
    Higher = more field covered per unit battery consumed.
    """
    if battery_used_pct <= 0:
        return 0.0
    return round((coverage_fraction * 100.0) / battery_used_pct, 3)




def run_simulation(
    altitude_m: float,
    overlap_frac: float,
    speed_ms: float,
    interval_s: float,
    battery_cap: float,
    wind_factor: float,
    slider_progress: float,
) -> Dict:
    """Execute the full simulation pipeline and return all results."""
    field = generate_field_raster(42)

    waypoints   = generate_lawnmower_path(altitude_m, overlap_frac)
    seg_lengths = compute_segment_lengths(waypoints)
    n_turns     = count_turns(waypoints)
    fp_radius   = compute_footprint_radius_px(altitude_m)
    gsd_cm      = compute_gsd_cm(altitude_m)
    spacing_px  = compute_line_spacing_px(altitude_m, overlap_frac)
    est_overlap = compute_estimated_overlap(fp_radius, spacing_px)
    path_len_m  = total_path_length_m(waypoints)
    full_dur_s  = mission_duration_s(waypoints, speed_ms)

    all_captures = schedule_capture_events(waypoints, seg_lengths, speed_ms, interval_s)

    batt        = compute_battery_budget(full_dur_s, n_turns, wind_factor, battery_cap)
    frac_ok     = batt["fraction_possible"]

    completed    = apply_battery_cutoff(all_captures, frac_ok)
    slider_frac  = slider_progress * frac_ok
    visible_caps = [ev for ev in completed if ev["progress"] <= slider_frac + 1e-6]
    current_pos  = (interpolate_position(waypoints, seg_lengths, slider_frac)
                    if waypoints else None)

    cov_mask  = build_coverage_mask(visible_caps, fp_radius, overlap_frac)
    cov_frac  = compute_coverage_fraction(cov_mask)

    true_zonal   = compute_zonal_stats(field["vari"], field["zones"])
    samp_zonal   = compute_zonal_stats(field["vari"], field["zones"], mask=cov_mask)
    bias         = compute_sampling_bias(field["vari"], cov_mask)
    batt_now_pct = round(min(batt["used_pct"], battery_cap) * slider_progress, 2)
    eff_score    = compute_efficiency_score(cov_frac, max(batt_now_pct, 0.1))

    return dict(
        field=field, waypoints=waypoints, seg_lengths=seg_lengths,
        all_captures=all_captures, visible_caps=visible_caps,
        current_pos=current_pos, batt=batt, batt_now_pct=batt_now_pct,
        cov_mask=cov_mask, cov_frac=cov_frac, est_overlap=est_overlap,
        fp_radius=fp_radius, gsd_cm=gsd_cm, path_len_m=path_len_m,
        full_dur_s=full_dur_s, slider_frac=slider_frac, n_turns=n_turns,
        true_zonal=true_zonal, samp_zonal=samp_zonal, bias=bias,
        eff_score=eff_score, spacing_px=spacing_px,
    )

# =============================================================================
# SECTION 7 — PLOTLY FIGURE BUILDERS
# =============================================================================

import plotly.graph_objects as go
import pandas as pd

_DARK_BG   = "#f8fafc"
_PAPER_BG  = "#ffffff"
_GRID_COL  = "#e2e8f0"
_AXIS_COL  = "#94a3b8"
_TEXT_COL  = "#0f172a"
_MUTED_COL = "#64748b"

_COLORBAR = dict(
    thickness=14,
    len=0.75,
    tickfont=dict(color=_MUTED_COL, size=10),
    title=dict(text="VARI", font=dict(color=_MUTED_COL, size=11)),
    bgcolor="rgba(255,255,255,0)",
    borderwidth=0,
)


def _dark_layout(title: str, height: int = 420, margin=None) -> dict:
    m = margin or dict(l=48, r=16, t=48, b=40)
    return dict(
        title=dict(text=title, font=dict(color=_TEXT_COL, size=13, family="monospace"),
                   x=0.0, xanchor="left"),
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_MUTED_COL, size=11),
        margin=m,
        height=height,
    )


def fig_mission_map(waypoints, all_captures, visible_caps, current_pos, vari, batt):
    fig = go.Figure()

    # VARI heatmap background
    fig.add_trace(go.Heatmap(
        z=vari.astype(float),
        colorscale="RdYlGn",
        zmin=-0.3, zmax=0.6,
        showscale=True,
        colorbar=dict(**_COLORBAR),
        opacity=0.65,
        name="VARI",
        hovertemplate="VARI: %{z:.3f}<extra></extra>",
    ))

    # Planned path
    if waypoints:
        fig.add_trace(go.Scatter(
            x=[w[0] for w in waypoints],
            y=[w[1] for w in waypoints],
            mode="lines",
            line=dict(color="rgba(160,170,180,0.35)", width=1, dash="dot"),
            name="Planned path",
            hoverinfo="skip",
        ))

    # Scheduled (hollow)
    if all_captures:
        fig.add_trace(go.Scatter(
            x=[e["col"] for e in all_captures],
            y=[e["row"] for e in all_captures],
            mode="markers",
            marker=dict(symbol="circle-open", size=4, color="rgba(80,180,255,0.28)"),
            name="Scheduled",
            customdata=[e["t"] for e in all_captures],
            hovertemplate="t=%{customdata:.1f}s<extra></extra>",
        ))

    # Completed captures
    if visible_caps:
        fig.add_trace(go.Scatter(
            x=[e["col"] for e in visible_caps],
            y=[e["row"] for e in visible_caps],
            mode="markers",
            marker=dict(symbol="circle", size=7, color="#38bdf8",
                        line=dict(width=1, color="rgba(255,255,255,0.6)")),
            name=f"Captured ({len(visible_caps)})",
            customdata=[e["t"] for e in visible_caps],
            hovertemplate="t=%{customdata:.1f}s<extra></extra>",
        ))

    # Drone marker
    if current_pos:
        fig.add_trace(go.Scatter(
            x=[current_pos[0]], y=[current_pos[1]],
            mode="markers",
            marker=dict(symbol="star", size=20, color="#f97316",
                        line=dict(width=2, color="white")),
            name="Drone",
            hovertemplate="Drone<extra></extra>",
        ))

    # Battery cutoff
    if not batt.get("mission_feasible", True) and visible_caps:
        last = visible_caps[-1]
        fig.add_trace(go.Scatter(
            x=[last["col"]], y=[last["row"]],
            mode="markers+text",
            marker=dict(symbol="x", size=22, color="#ef4444",
                        line=dict(width=3, color="white")),
            text=["⚡ CUTOFF"], textposition="top center",
            textfont=dict(color="#ef4444", size=11),
            name="Cutoff",
            hovertemplate="Battery depleted<extra></extra>",
        ))

    fig.update_layout(
        **_dark_layout("LIVE MISSION VIEW", height=480),
        xaxis=dict(range=[0, 128], showgrid=False, zeroline=False,
                   color=_AXIS_COL, title=dict(text="col (px)", font=dict(size=10))),
        yaxis=dict(range=[128, 0], showgrid=False, zeroline=False,
                   color=_AXIS_COL, scaleanchor="x",
                   title=dict(text="row (px)", font=dict(size=10))),
        legend=dict(bgcolor="rgba(248,250,252,0.92)", font=dict(size=10),
                    bordercolor=_GRID_COL, borderwidth=1,
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
    )
    return fig


def fig_true_map(vari):
    fig = go.Figure(go.Heatmap(
        z=vari.astype(float),
        colorscale="RdYlGn", zmin=-0.3, zmax=0.6,
        colorbar=dict(**_COLORBAR),
        hovertemplate="VARI: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **_dark_layout("TRUE FIELD — full VARI", height=360),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   autorange="reversed", scaleanchor="x"),
    )
    return fig


def fig_recon_map(vari, mask):
    display = vari.astype(float).copy()
    display[~mask] = float("nan")

    fig = go.Figure(go.Heatmap(
        z=display,
        colorscale="RdYlGn", zmin=-0.3, zmax=0.6,
        colorbar=dict(**_COLORBAR),
        hovertemplate="VARI: %{z:.3f}<extra></extra>",
    ))

    if not mask.all():
        import numpy as np
        unobs = np.where(mask, float("nan"), 0.0)
        fig.add_trace(go.Heatmap(
            z=unobs,
            colorscale=[[0, "rgba(220,228,240,0.88)"], [1, "rgba(220,228,240,0.88)"]],
            showscale=False,
            hovertemplate="Not observed<extra></extra>",
        ))

    fig.update_layout(
        **_dark_layout("RECONSTRUCTED — sampled pixels only", height=360),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   autorange="reversed", scaleanchor="x"),
    )
    return fig


def fig_terrain(dem, vari, mask=None):
    import numpy as np
    vari_disp = vari.copy()
    if mask is not None:
        vari_disp = np.where(mask, vari_disp, float("nan"))

    step = 2
    surf = go.Surface(
        z=dem[::step, ::step],
        surfacecolor=vari_disp[::step, ::step],
        colorscale="RdYlGn",
        cmin=-0.3, cmax=0.6,
        colorbar=dict(**_COLORBAR),
        lighting=dict(ambient=0.7, diffuse=0.5, roughness=0.5),
        hovertemplate="Elev: %{z:.2f}m  VARI: %{surfacecolor:.3f}<extra></extra>",
    )
    fig = go.Figure(surf)

    # Note: 3D scene layout does NOT use paper_bgcolor / plot_bgcolor keys
    fig.update_layout(
        title=dict(text="3D TERRAIN — DEM + sampled VARI",
                   font=dict(color=_TEXT_COL, size=13, family="monospace"),
                   x=0.0, xanchor="left"),
        paper_bgcolor=_PAPER_BG,
        font=dict(color=_MUTED_COL, size=11),
        height=500,
        margin=dict(l=0, r=0, t=48, b=0),
        scene=dict(
            bgcolor=_PAPER_BG,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       backgroundcolor=_DARK_BG, gridcolor=_GRID_COL),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       backgroundcolor=_DARK_BG, gridcolor=_GRID_COL),
            zaxis=dict(showgrid=True, zeroline=False, color=_AXIS_COL,
                       backgroundcolor=_DARK_BG, gridcolor=_GRID_COL,
                       title=dict(text="Elevation (m)")),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.1)),
        ),
    )
    return fig


def fig_battery_breakdown(batt):
    categories = ["Base Flight", "Turn Penalty", "Wind"]
    values     = [batt["base_cost_pct"], batt["turn_cost_pct"], batt["wind_cost_pct"]]
    colors     = ["#38bdf8", "#a78bfa", "#fb923c"]

    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=colors,
        marker_line=dict(width=0),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=_TEXT_COL, size=12),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))

    cap = batt["available_pct"]
    fig.add_hline(
        y=cap, line_color="#ef4444", line_dash="dash", line_width=1.5,
        annotation_text=f"Capacity {cap:.0f}%",
        annotation_font=dict(color="#ef4444", size=11),
        annotation_position="top right",
    )

    fig.update_layout(
        **_dark_layout("BATTERY COST BREAKDOWN", height=280,
                       margin=dict(l=48, r=16, t=48, b=36)),
        xaxis=dict(showgrid=False, zeroline=False, color=_AXIS_COL),
        yaxis=dict(showgrid=True, gridcolor=_GRID_COL, zeroline=False,
                   color=_AXIS_COL, title=dict(text="Battery %", font=dict(size=10)),
                   range=[0, max(max(values), cap) * 1.45]),
        bargap=0.35,
    )
    return fig


# =============================================================================
# SECTION 8 — STREAMLIT UI
# =============================================================================

import streamlit as st

st.set_page_config(
    page_title="Flight Mission Simulator",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #f8fafc !important;
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] * { font-family: 'IBM Plex Sans', sans-serif; }

/* Main content bg */
section.main > div { background: #f8fafc; }

/* Headings */
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #0f172a !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 12px 16px !important;
    transition: border-color .2s, box-shadow .2s;
}
[data-testid="stMetric"]:hover { border-color: #cbd5e1; box-shadow: 0 2px 8px rgba(0,0,0,.06); }
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: .08em;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 22px !important;
}

/* Tabs */
[data-testid="stTabs"] { border-bottom: 1px solid #e2e8f0; }
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: #94a3b8 !important;
    padding: 10px 18px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0284c7 !important;
    border-bottom: 2px solid #0284c7 !important;
}

/* Sliders */
[data-testid="stSlider"] label {
    font-size: 11px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #0284c7 !important; }

/* Run button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase;
    padding: 12px !important;
    cursor: pointer;
    transition: opacity .15s;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px;
}

/* Divider */
hr { border-color: #e2e8f0 !important; margin: 20px 0 !important; }

/* Caption text */
[data-testid="stCaptionContainer"] { color: #94a3b8 !important; font-size: 11px !important; }

/* Alerts / banners */
.banner {
    border-radius: 6px;
    padding: 11px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .04em;
    margin-bottom: 16px;
    border-left: 3px solid;
}
.banner-ok   { background: rgba(34,197,94,.08);  border-color: #16a34a; color: #15803d; }
.banner-warn { background: rgba(234,88,12,.08);  border-color: #ea580c; color: #c2410c; }
.banner-err  { background: rgba(220,38,38,.08);  border-color: #dc2626; color: #b91c1c; }

/* KPI section label */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #e2e8f0;
}

/* Score badge */
.score-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 48px;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -.02em;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "ran_once" not in st.session_state:
    st.session_state.ran_once = False


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:IBM Plex Mono,monospace;font-size:18px;'
        'font-weight:700;color:#0f172a;margin-bottom:4px;">🛰 MISSION SIM</div>'
        '<div style="font-family:IBM Plex Mono,monospace;font-size:9px;'
        'color:#484f58;letter-spacing:.12em;margin-bottom:20px;">'
        'DRONE · AGRICULTURE · 4 ACRES</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">✈ Flight</div>', unsafe_allow_html=True)
    altitude_m  = st.slider("Altitude (m)",          10,  120,  30,  5)
    speed_ms    = st.slider("Speed (m/s)",             1,   20,   5,  1)
    interval_s  = st.slider("Capture interval (s)",  0.5, 10.0, 2.0, 0.5)
    overlap_pct = st.slider("Sidelap overlap (%)",    20,   95,  75,  5)

    st.markdown('<div class="section-label" style="margin-top:18px;">⚡ Battery</div>',
                unsafe_allow_html=True)
    battery_cap = st.slider("Battery capacity (%)",  10,  100, 100,  5)
    wind_factor = st.slider("Wind resistance (×)",  0.0,  5.0, 1.0, 0.5)

    st.markdown('<div class="section-label" style="margin-top:18px;">▶ Playback</div>',
                unsafe_allow_html=True)
    progress_pct = st.slider("Mission progress (%)",  0,  100, 100,  1)

    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("⚡  RUN SIMULATION")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;'
        f'color:#484f58;line-height:1.8;">'
        f'CAMERA · Sony RX1R II equiv<br>'
        f'HFOV · {HFOV_DEG:.1f}°<br>'
        f'FIELD · {FIELD_SIDE_M:.0f} × {FIELD_SIDE_M:.0f} m<br>'
        f'RASTER · 128 × 128 px</div>',
        unsafe_allow_html=True,
    )


# ── Run simulation on button click or first load ───────────────────────────────
if run_clicked or not st.session_state.ran_once:
    with st.spinner("Running simulation…"):
        st.session_state.results = run_simulation(
            altitude_m      = float(altitude_m),
            overlap_frac    = overlap_pct / 100.0,
            speed_ms        = float(speed_ms),
            interval_s      = float(interval_s),
            battery_cap     = float(battery_cap),
            wind_factor     = float(wind_factor),
            slider_progress = progress_pct / 100.0,
        )
        st.session_state.ran_once = True

# Progress slider always refreshes without re-simulating (cheap)
elif st.session_state.results is not None:
    R_old = st.session_state.results
    # Re-apply progress slider only — reuse existing captures/mask/stats
    new_slider_frac = (progress_pct / 100.0) * R_old["batt"]["fraction_possible"]
    vis = [ev for ev in R_old["all_captures"]
           if ev["progress"] <= new_slider_frac + 1e-6]
    cur = (interpolate_position(R_old["waypoints"],
                                compute_segment_lengths(R_old["waypoints"]),
                                new_slider_frac)
           if R_old["waypoints"] else None)
    cov_mask2 = build_coverage_mask(vis, R_old["fp_radius"], R_old["est_overlap"])
    cov_frac2 = compute_coverage_fraction(cov_mask2)
    batt_now2 = round(min(R_old["batt"]["used_pct"], battery_cap) * (progress_pct/100), 2)
    eff2      = compute_efficiency_score(cov_frac2, max(batt_now2, 0.1))
    bias2     = compute_sampling_bias(R_old["field"]["vari"], cov_mask2)
    samp_z2   = compute_zonal_stats(R_old["field"]["vari"], R_old["field"]["zones"], mask=cov_mask2)
    # Patch into results dict
    R_patched = dict(R_old)
    R_patched.update(
        visible_caps=vis, current_pos=cur, cov_mask=cov_mask2,
        cov_frac=cov_frac2, batt_now_pct=batt_now2, eff_score=eff2,
        bias=bias2, samp_zonal=samp_z2, slider_frac=new_slider_frac,
    )
    st.session_state.results = R_patched

R = st.session_state.results

if R is None:
    st.markdown(
        '<div style="text-align:center;padding:80px;font-family:IBM Plex Mono,monospace;'
        'color:#484f58;">Press ⚡ RUN SIMULATION to begin</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# Unpack
field       = R["field"]
batt        = R["batt"]
bias        = R["bias"]
cov_frac    = R["cov_frac"]
est_overlap = R["est_overlap"]


# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="font-size:28px;letter-spacing:-.01em;margin-bottom:2px;">'
    '🛰 Flight Mission & Sampling Simulator</h1>'
    '<p style="color:#7d8590;font-size:13px;margin-top:0;margin-bottom:20px;">'
    'Agricultural drone survey simulation · coverage · reconstruction · VARI bias</p>',
    unsafe_allow_html=True,
)


# ── Status banner ──────────────────────────────────────────────────────────────
if not batt["mission_feasible"]:
    st.markdown(
        f'<div class="banner banner-err">⚡ BATTERY CUTOFF — '
        f'Mission terminated at {batt["fraction_possible"]*100:.1f}% of planned path. '
        f'Reduce wind, lower altitude, or increase battery capacity.</div>',
        unsafe_allow_html=True,
    )
elif est_overlap < 0.70:
    st.markdown(
        '<div class="banner banner-warn">⚠ LOW SIDELAP — '
        'Overlap below 70% causes photogrammetric reconstruction gaps. '
        'Increase overlap or lower altitude.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="banner banner-ok">✓ MISSION FEASIBLE — '
        'Battery sufficient · Overlap adequate for full reconstruction.</div>',
        unsafe_allow_html=True,
    )


# ── KPI row ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Mission metrics</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Coverage",    f"{cov_frac*100:.1f}%")
with c2: st.metric("Captures",    f"{len(R['visible_caps'])} / {len(R['all_captures'])}")
with c3: st.metric("GSD",         f"{R['gsd_cm']:.2f} cm/px")
with c4: st.metric("Overlap",     f"{est_overlap*100:.0f}%")

c5, c6, c7, c8 = st.columns(4)
with c5: st.metric("Path length",  f"{R['path_len_m']:.0f} m")
with c6: st.metric("Turns",        str(R["n_turns"]))
with c7: st.metric("Elapsed",      f"{R['full_dur_s'] * R['slider_frac'] / 60:.1f} min")
with c8: st.metric("Efficiency",   f"{R['eff_score']:.2f}")

st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_live, tab_maps, tab_terrain, tab_analytics = st.tabs([
    "  🛰  LIVE MISSION  ",
    "  🗺  TRUE vs RECON  ",
    "  🏔  3D TERRAIN  ",
    "  📊  ANALYTICS  ",
])


# ── Tab 1: Live Mission ────────────────────────────────────────────────────────
with tab_live:
    st.plotly_chart(
        fig_mission_map(R["waypoints"], R["all_captures"], R["visible_caps"],
                        R["current_pos"], field["vari"], batt),
        use_container_width=True,
    )

    st.markdown('<div class="section-label" style="margin-top:4px;">Efficiency score</div>',
                unsafe_allow_html=True)

    eff       = R["eff_score"]
    eff_color = "#22c55e" if eff > 1.5 else "#fb923c" if eff > 0.8 else "#ef4444"
    eff_label = "EXCELLENT" if eff > 1.5 else "MODERATE" if eff > 0.8 else "POOR"

    col_score, col_desc = st.columns([1, 3])
    with col_score:
        st.markdown(
            f'<div style="padding:24px 0 8px 0;">'
            f'<div class="score-badge" style="color:{eff_color};">{eff:.2f}</div>'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;'
            f'color:{eff_color};letter-spacing:.12em;margin-top:4px;">{eff_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_desc:
        st.markdown(
            f"**Score = Coverage% ÷ Battery%** &nbsp;·&nbsp; higher is better\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Coverage | **{cov_frac*100:.1f}%** of 4-acre field |\n"
            f"| Battery used | **{R['batt_now_pct']:.1f}%** of capacity |\n"
            f"| Footprint radius | **{R['fp_radius'] * CELL_SIZE_M:.1f} m** |\n"
            f"| Strips flown | **{R['n_turns'] + 1}** lanes |"
        )


# ── Tab 2: True vs Reconstructed ──────────────────────────────────────────────
with tab_maps:
    col_t, col_r = st.columns(2)
    with col_t:
        st.plotly_chart(fig_true_map(field["vari"]), use_container_width=True)
    with col_r:
        st.plotly_chart(fig_recon_map(field["vari"], R["cov_mask"]),
                        use_container_width=True)

    st.divider()
    st.markdown('<div class="section-label">VARI Sampling Bias</div>',
                unsafe_allow_html=True)

    true_m = bias["true_mean"]
    samp_m = bias["sampled_mean"]
    abs_b  = bias["absolute_bias"]
    rel_b  = bias["relative_bias"]

    b1, b2, b3, b4 = st.columns(4)
    with b1: st.metric("True Mean VARI",   f"{true_m:.4f}" if true_m is not None else "—")
    with b2: st.metric("Sampled Mean",     f"{samp_m:.4f}" if samp_m is not None else "—")
    with b3: st.metric("Absolute Bias",    f"{abs_b:+.4f}" if abs_b is not None else "—")
    with b4: st.metric("Relative Bias",    f"{rel_b:+.2f}%" if rel_b is not None else "—")

    if abs_b is not None and abs(abs_b) > 0.01:
        st.warning(
            f"Sampling bias of **{abs_b:+.4f} VARI** ({rel_b:+.2f}%) detected — "
            "captured pixels don't evenly represent the full field. "
            "Increase coverage to reduce bias."
        )


# ── Tab 3: 3D Terrain ─────────────────────────────────────────────────────────
with tab_terrain:
    mask_arg = R["cov_mask"] if R["cov_mask"].any() else None
    st.plotly_chart(
        fig_terrain(field["dem"], field["vari"], mask=mask_arg),
        use_container_width=True,
    )
    st.caption(
        "DEM surface (0–8 m relative relief) coloured by sampled VARI. "
        "Dark = unobserved. Drag to rotate · scroll to zoom."
    )


# ── Tab 4: Analytics ──────────────────────────────────────────────────────────
with tab_analytics:

    st.markdown('<div class="section-label">Zonal Statistics — True vs Sampled</div>',
                unsafe_allow_html=True)

    z1, z2 = st.columns(2)
    with z1:
        st.caption("True field (all pixels)")
        st.dataframe(pd.DataFrame(R["true_zonal"]),
                     use_container_width=True, hide_index=True)
    with z2:
        st.caption("Sampled only (captured pixels)")
        st.dataframe(pd.DataFrame(R["samp_zonal"]),
                     use_container_width=True, hide_index=True)

    st.divider()
    st.plotly_chart(fig_battery_breakdown(batt), use_container_width=True)

    st.divider()
    st.markdown('<div class="section-label">Full Mission Summary</div>',
                unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Field | {FIELD_ACRES} ac · {FIELD_SIDE_M:.0f}×{FIELD_SIDE_M:.0f} m |
| Altitude | {altitude_m} m |
| GSD | {R['gsd_cm']:.2f} cm/px |
| Footprint radius | {R['fp_radius'] * CELL_SIZE_M:.1f} m |
| Flight speed | {speed_ms} m/s |
| Capture interval | {interval_s} s |
| Path length | {R['path_len_m']:.0f} m |
| Full duration | {R['full_dur_s']/60:.1f} min |
""")
    with s2:
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Sidelap setting | {overlap_pct}% |
| Actual overlap | {est_overlap*100:.1f}% |
| Waypoints | {len(R['waypoints'])} |
| Lane turns | {R['n_turns']} |
| Scheduled captures | {len(R['all_captures'])} |
| Completed captures | {len(R['visible_caps'])} |
| Coverage | {cov_frac*100:.2f}% |
| Battery required | {batt['used_pct']:.1f}% / {batt['available_pct']:.0f}% cap |
""")