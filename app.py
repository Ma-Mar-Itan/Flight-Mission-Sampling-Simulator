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


# =============================================================================
# SECTION 7 — PLOTLY FIGURE BUILDERS
# =============================================================================

_DARK_BG   = "#0d1117"
_DARK_SURF = "#161b22"
_AXIS_COL  = "#555"


def _base_layout(title: str, height: int = 420) -> dict:
    return dict(
        title=dict(text=title, font=dict(color="#e0e0e0", size=14)),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        margin=dict(l=40, r=20, t=44, b=40),
        height=height,
        font=dict(color="#aaa"),
    )


def fig_mission_map(
    waypoints: List[Tuple[float, float]],
    all_captures: List[Dict],
    visible_caps: List[Dict],
    current_pos: Optional[Tuple[float, float]],
    vari: np.ndarray,
    batt: Dict,
) -> go.Figure:
    """
    Live Mission View: VARI heatmap background + flight path + captures + drone.
    """
    # Mask NaN-safe display copy
    display_vari = vari.astype(float)

    fig = go.Figure()

    # 1. VARI background heatmap
    fig.add_trace(go.Heatmap(
        z=display_vari,
        colorscale="RdYlGn",
        zmin=-0.3, zmax=0.6,
        showscale=False,
        opacity=0.72,
        name="VARI (ground truth)",
        hovertemplate="VARI: %{z:.3f}<extra></extra>",
    ))

    # 2. Full planned path (grey dotted)
    if waypoints:
        fig.add_trace(go.Scatter(
            x=[w[0] for w in waypoints],
            y=[w[1] for w in waypoints],
            mode="lines",
            line=dict(color="rgba(180,180,180,0.40)", width=1.3, dash="dot"),
            name="Planned path",
            hoverinfo="skip",
        ))

    # 3. All scheduled capture positions (hollow)
    if all_captures:
        fig.add_trace(go.Scatter(
            x=[e["col"] for e in all_captures],
            y=[e["row"] for e in all_captures],
            mode="markers",
            marker=dict(symbol="circle-open", size=4,
                        color="rgba(100,200,255,0.30)"),
            name="Scheduled captures",
            customdata=[e["t"] for e in all_captures],
            hovertemplate="t=%{customdata:.1f}s<extra></extra>",
        ))

    # 4. Completed captures (filled cyan)
    if visible_caps:
        fig.add_trace(go.Scatter(
            x=[e["col"] for e in visible_caps],
            y=[e["row"] for e in visible_caps],
            mode="markers",
            marker=dict(symbol="circle", size=6, color="#00b4d8",
                        line=dict(width=1, color="white")),
            name=f"Captured ({len(visible_caps)})",
            customdata=[e["t"] for e in visible_caps],
            hovertemplate="t=%{customdata:.1f}s  (%{x:.1f}, %{y:.1f})<extra></extra>",
        ))

    # 5. Current drone position (star)
    if current_pos:
        fig.add_trace(go.Scatter(
            x=[current_pos[0]], y=[current_pos[1]],
            mode="markers",
            marker=dict(symbol="star", size=18, color="#ff6b35",
                        line=dict(width=2, color="white")),
            name="Drone",
            hovertemplate="Drone @ (%{x:.1f}, %{y:.1f})<extra></extra>",
        ))

    # 6. Battery cutoff marker (red ×)
    if not batt.get("mission_feasible", True) and visible_caps:
        last = visible_caps[-1]
        fig.add_trace(go.Scatter(
            x=[last["col"]], y=[last["row"]],
            mode="markers+text",
            marker=dict(symbol="x", size=20, color="red",
                        line=dict(width=3, color="white")),
            text=["⚡ CUTOFF"],
            textposition="top center",
            textfont=dict(color="red", size=11),
            name="Battery cutoff",
            hovertemplate="Battery depleted here<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout("🛰  Live Mission View", height=460),
        xaxis=dict(range=[0, RASTER_COLS], showgrid=False, zeroline=False,
                   color=_AXIS_COL, title="Column (px)"),
        yaxis=dict(range=[RASTER_ROWS, 0], showgrid=False, zeroline=False,
                   color=_AXIS_COL, title="Row (px)", scaleanchor="x"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)",
                    font=dict(color="#ccc", size=10), x=0.01, y=0.99),
    )
    return fig


def fig_true_map(vari: np.ndarray) -> go.Figure:
    """Full ground-truth VARI heatmap."""
    fig = go.Figure(go.Heatmap(
        z=vari.astype(float),
        colorscale="RdYlGn",
        zmin=-0.3, zmax=0.6,
        colorbar=dict(title="VARI", thickness=12, len=0.7,
                      tickfont=dict(color="#aaa"),
                      titlefont=dict(color="#aaa")),
        hovertemplate="VARI: %{z:.3f}<br>row=%{y}  col=%{x}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("✅  True Farm — full VARI", height=380),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   autorange="reversed", scaleanchor="x"),
    )
    return fig


def fig_recon_map(vari: np.ndarray, mask: np.ndarray) -> go.Figure:
    """Reconstructed VARI — only captured pixels shown; gaps are dark."""
    display = vari.astype(float).copy()
    display[~mask] = float("nan")

    fig = go.Figure(go.Heatmap(
        z=display,
        colorscale="RdYlGn",
        zmin=-0.3, zmax=0.6,
        colorbar=dict(title="VARI", thickness=12, len=0.7,
                      tickfont=dict(color="#aaa"),
                      titlefont=dict(color="#aaa")),
        hovertemplate="VARI: %{z:.3f}<br>row=%{y}  col=%{x}<extra></extra>",
    ))

    # Dark overlay for unobserved pixels
    if not mask.all():
        unobs = np.where(mask, float("nan"), 0.0)
        fig.add_trace(go.Heatmap(
            z=unobs,
            colorscale=[[0, "rgba(8,8,12,0.94)"], [1, "rgba(8,8,12,0.94)"]],
            showscale=False,
            hovertemplate="Not observed<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout("🛰  Reconstructed Map — sampled pixels only", height=380),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   autorange="reversed", scaleanchor="x"),
    )
    return fig


def fig_terrain(dem: np.ndarray, vari: np.ndarray,
                mask: Optional[np.ndarray] = None) -> go.Figure:
    """3D DEM surface coloured by sampled VARI."""
    vari_disp = vari.copy()
    if mask is not None:
        vari_disp = np.where(mask, vari_disp, float("nan"))

    step = 2   # downsample for 3D performance
    fig = go.Figure(go.Surface(
        z=dem[::step, ::step],
        surfacecolor=vari_disp[::step, ::step],
        colorscale="RdYlGn",
        cmin=-0.3, cmax=0.6,
        colorbar=dict(title="VARI", thickness=12,
                      tickfont=dict(color="#aaa"),
                      titlefont=dict(color="#aaa"))),
    )
    fig.update_layout(
        **_base_layout("🏔  3D Terrain — DEM + sampled VARI", height=480),
        scene=dict(
            bgcolor=_DARK_BG,
            xaxis=dict(showgrid=False, color="#444", title=""),
            yaxis=dict(showgrid=False, color="#444", title=""),
            zaxis=dict(showgrid=False, color="#444", title="Elev (m)"),
            camera=dict(eye=dict(x=1.4, y=-1.4, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=44, b=0),
    )
    return fig


def fig_battery_breakdown(batt: Dict) -> go.Figure:
    """Bar chart of battery cost components vs capacity."""
    categories = ["Base Flight", "Turn Penalty", "Wind Resistance"]
    values     = [batt["base_cost_pct"], batt["turn_cost_pct"], batt["wind_cost_pct"]]
    colors     = ["#00e5ff", "#7c3aed", "#f97316"]

    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#c9d1d9", size=12),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))

    cap = batt["available_pct"]
    fig.add_hline(y=cap, line_color="#ef4444", line_dash="dash", line_width=2,
                  annotation_text=f"Capacity ({cap:.0f}%)",
                  annotation_font_color="#ef4444",
                  annotation_position="top right")

    fig.update_layout(
        **_base_layout("⚡  Battery Cost Breakdown", height=260),
        xaxis=dict(showgrid=False, zeroline=False, color=_AXIS_COL),
        yaxis=dict(showgrid=True, gridcolor="#21262d", zeroline=False,
                   color=_AXIS_COL, title="Battery %",
                   range=[0, max(max(values), cap) * 1.35]),
    )
    return fig


# =============================================================================
# SECTION 8 — STREAMLIT UI
# =============================================================================

def _color_metric(value: float, thresholds: Tuple[float, float],
                  fmt: str = "{:.1f}") -> str:
    """Return a coloured markdown string based on value vs thresholds."""
    lo, hi = thresholds
    if value >= hi:
        color = "#22c55e"   # green
    elif value >= lo:
        color = "#f97316"   # orange
    else:
        color = "#ef4444"   # red
    return f'<span style="color:{color};font-weight:700;">{fmt.format(value)}</span>'


def run_simulation(
    altitude_m: float,
    overlap_frac: float,
    speed_ms: float,
    interval_s: float,
    battery_cap: float,
    wind_factor: float,
    slider_progress: float,
) -> Dict:
    """
    Execute the full simulation pipeline and return all results.
    Called on every Streamlit interaction.
    """
    field = generate_field_raster(42)

    # Section 2: geometry
    waypoints   = generate_lawnmower_path(altitude_m, overlap_frac)
    seg_lengths = compute_segment_lengths(waypoints)
    n_turns     = count_turns(waypoints)
    fp_radius   = compute_footprint_radius_px(altitude_m)
    gsd_cm      = compute_gsd_cm(altitude_m)
    spacing_px  = compute_line_spacing_px(altitude_m, overlap_frac)
    est_overlap = compute_estimated_overlap(fp_radius, spacing_px)
    path_len_m  = total_path_length_m(waypoints)
    full_dur_s  = mission_duration_s(waypoints, speed_ms)

    # Section 3: capture schedule
    all_captures = schedule_capture_events(
        waypoints, seg_lengths, speed_ms, interval_s)

    # Section 4: battery
    batt        = compute_battery_budget(
        full_dur_s, n_turns, wind_factor, battery_cap)
    frac_ok     = batt["fraction_possible"]

    # Apply battery cutoff then progress slider
    completed    = apply_battery_cutoff(all_captures, frac_ok)
    slider_frac  = slider_progress * frac_ok
    visible_caps = [ev for ev in completed
                    if ev["progress"] <= slider_frac + 1e-6]
    current_pos  = (interpolate_position(waypoints, seg_lengths, slider_frac)
                    if waypoints else None)

    # Section 5: reconstruction
    cov_mask  = build_coverage_mask(visible_caps, fp_radius, overlap_frac)
    cov_frac  = compute_coverage_fraction(cov_mask)

    # Section 6: statistics
    true_zonal   = compute_zonal_stats(field["vari"], field["zones"])
    samp_zonal   = compute_zonal_stats(field["vari"], field["zones"],
                                       mask=cov_mask)
    bias         = compute_sampling_bias(field["vari"], cov_mask)
    batt_now_pct = round(
        min(batt["used_pct"], battery_cap) * slider_progress, 2)
    eff_score    = compute_efficiency_score(cov_frac, max(batt_now_pct, 0.1))

    return dict(
        field=field,
        waypoints=waypoints,
        all_captures=all_captures,
        visible_caps=visible_caps,
        current_pos=current_pos,
        batt=batt,
        batt_now_pct=batt_now_pct,
        cov_mask=cov_mask,
        cov_frac=cov_frac,
        est_overlap=est_overlap,
        fp_radius=fp_radius,
        gsd_cm=gsd_cm,
        path_len_m=path_len_m,
        full_dur_s=full_dur_s,
        slider_frac=slider_frac,
        n_turns=n_turns,
        true_zonal=true_zonal,
        samp_zonal=samp_zonal,
        bias=bias,
        eff_score=eff_score,
        spacing_px=spacing_px,
    )


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Mission & Sampling Simulator",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark-theme CSS injection ───────────────────────────────────────────
st.markdown("""
<style>
/* Dark background everywhere */
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    background-color: #0d1117;
}
[data-testid="stSidebar"] {
    border-right: 1px solid #30363d;
}
/* Metric cards */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 14px !important;
}
[data-testid="stMetricLabel"] { color: #6e7681 !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #c9d1d9 !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }
/* Tab styling */
[data-testid="stTabs"] button { color: #6e7681 !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom-color: #00e5ff !important;
}
/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
/* Section headers */
.sim-section {
    font-size: 10px; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #6e7681;
    border-bottom: 1px solid #30363d;
    padding-bottom: 4px; margin-bottom: 10px; margin-top: 14px;
}
/* Alert boxes */
.alert-danger {
    background: rgba(239,68,68,.10); border: 1px solid rgba(239,68,68,.4);
    border-radius: 8px; padding: 10px 14px; color: #fca5a5;
    font-size: 12px; font-weight: 600; margin: 6px 0;
}
.alert-warn {
    background: rgba(249,115,22,.10); border: 1px solid rgba(249,115,22,.4);
    border-radius: 8px; padding: 10px 14px; color: #fdba74;
    font-size: 12px; font-weight: 600; margin: 6px 0;
}
.alert-ok {
    background: rgba(34,197,94,.10); border: 1px solid rgba(34,197,94,.4);
    border-radius: 8px; padding: 10px 14px; color: #86efac;
    font-size: 12px; font-weight: 600; margin: 6px 0;
}
/* Efficiency score */
.eff-block {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px 18px; display: flex; align-items: center; gap: 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰 Flight Mission Simulator")
    st.caption("4-acre agricultural survey · Sony RX1R II camera model")
    st.divider()

    st.markdown('<div class="sim-section">✈ Flight Parameters</div>',
                unsafe_allow_html=True)
    altitude_m   = st.slider("Altitude (m)",       10,  120, 30,   5)
    speed_ms     = st.slider("Flight Speed (m/s)",  1,   20,  5,   1)
    interval_s   = st.slider("Capture Interval (s)", 0.5, 10.0, 2.0, 0.5)
    overlap_pct  = st.slider("Overlap (%)",         20,   95, 75,   5)

    st.markdown('<div class="sim-section">⚡ Battery</div>',
                unsafe_allow_html=True)
    battery_cap  = st.slider("Battery Capacity (%)", 10, 100, 100, 5)
    wind_factor  = st.slider("Wind Resistance (×)",  0.0, 5.0, 1.0, 0.5)

    st.markdown('<div class="sim-section">▶ Mission Progress</div>',
                unsafe_allow_html=True)
    progress_pct = st.slider("Mission Progress (%)", 0, 100, 100, 1)

    st.divider()
    st.caption(
        "**Camera:** Sony RX1R II equivalent  \n"
        f"HFOV: {HFOV_DEG:.1f}°  ·  "
        f"Field: {FIELD_SIDE_M:.1f} m × {FIELD_SIDE_M:.1f} m"
    )


# ── Run simulation ────────────────────────────────────────────────────────────
R = run_simulation(
    altitude_m    = float(altitude_m),
    overlap_frac  = overlap_pct / 100.0,
    speed_ms      = float(speed_ms),
    interval_s    = float(interval_s),
    battery_cap   = float(battery_cap),
    wind_factor   = float(wind_factor),
    slider_progress = progress_pct / 100.0,
)

field        = R["field"]
batt         = R["batt"]
bias         = R["bias"]
cov_frac     = R["cov_frac"]
est_overlap  = R["est_overlap"]


# ── Page title ────────────────────────────────────────────────────────────────
st.markdown("# 🛰 Flight Mission & Sampling Simulator")
st.markdown(
    "Simulate how a drone acquires agricultural imagery over a 4-acre field. "
    "Adjust parameters to see how mission design affects coverage, reconstruction "
    "quality, and statistical bias in VARI estimates."
)

# ── Status banners ─────────────────────────────────────────────────────────────
if not batt["mission_feasible"]:
    st.markdown(
        '<div class="alert-danger">⚡ <b>BATTERY CUTOFF</b> — '
        'Mission terminated early. Reconstructed map shows partial coverage only. '
        f'Only {batt["fraction_possible"]*100:.1f}% of path completed.</div>',
        unsafe_allow_html=True,
    )
elif est_overlap < 0.70:
    st.markdown(
        '<div class="alert-warn">⚠ <b>LOW OVERLAP</b> — '
        'Reconstruction gaps are visible below 70% sidelap. '
        'Increase overlap for reliable photogrammetric mapping.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="alert-ok">✅ <b>MISSION FEASIBLE</b> — '
        'Battery sufficient · Overlap adequate for reconstruction.</div>',
        unsafe_allow_html=True,
    )


# ── Top KPI row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
with c1:
    st.metric("Coverage", f"{cov_frac*100:.1f}%")
with c2:
    st.metric("Captures", f"{len(R['visible_caps'])}/{len(R['all_captures'])}")
with c3:
    st.metric("GSD", f"{R['gsd_cm']:.2f} cm/px")
with c4:
    st.metric("Overlap est.", f"{est_overlap*100:.0f}%")
with c5:
    st.metric("Waypoints", R["waypoints"].__len__())
with c6:
    st.metric("Turns", R["n_turns"])
with c7:
    st.metric("Elapsed", f"{R['full_dur_s'] * R['slider_frac'] / 60:.1f} min")
with c8:
    st.metric("Path", f"{R['path_len_m']:.0f} m")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_mission, tab_maps, tab_terrain, tab_analytics = st.tabs([
    "🛰 Live Mission",
    "🗺 True vs Reconstructed",
    "🏔 3D Terrain",
    "📈 Analytics",
])


# ── Tab 1: Live Mission ────────────────────────────────────────────────────────
with tab_mission:
    st.plotly_chart(
        fig_mission_map(
            R["waypoints"], R["all_captures"], R["visible_caps"],
            R["current_pos"], field["vari"], batt,
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### ⚡ Mission Efficiency")

    eff = R["eff_score"]
    eff_color = "#22c55e" if eff > 1.5 else "#f97316" if eff > 0.8 else "#ef4444"
    eff_label = "Excellent" if eff > 1.5 else "Moderate" if eff > 0.8 else "Poor"

    col_eff, col_eff_desc = st.columns([1, 3])
    with col_eff:
        st.markdown(
            f'<div style="text-align:center;padding:20px;">'
            f'<div style="font-size:42px;font-weight:900;'
            f'color:{eff_color};font-family:monospace;">{eff:.2f}</div>'
            f'<div style="font-size:12px;color:{eff_color};">{eff_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_eff_desc:
        st.markdown(
            f"""
**Efficiency Score = Coverage % / Battery Used %**

Higher is better — it means more field was covered per unit of battery spent.

| Metric | Value |
|---|---|
| Coverage | **{cov_frac*100:.1f}%** of field observed |
| Battery used | **{R['batt_now_pct']:.1f}%** of capacity |
| Footprint radius | **{R['fp_radius'] * CELL_SIZE_M:.1f} m** at {altitude_m} m altitude |
| Estimated overlap | **{est_overlap*100:.0f}%** sidelap |
"""
        )


# ── Tab 2: True vs Reconstructed ──────────────────────────────────────────────
with tab_maps:
    col_true, col_recon = st.columns(2)
    with col_true:
        st.plotly_chart(
            fig_true_map(field["vari"]),
            use_container_width=True,
        )
    with col_recon:
        st.plotly_chart(
            fig_recon_map(field["vari"], R["cov_mask"]),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### 🌿 VARI Sampling Bias")
    st.markdown(
        "The drone's incomplete coverage introduces **statistical bias** — "
        "the sampled field mean differs from the true field mean because "
        "the captured pixels are not a perfectly representative sample."
    )

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    true_m = bias["true_mean"]
    samp_m = bias["sampled_mean"]
    abs_b  = bias["absolute_bias"]
    rel_b  = bias["relative_bias"]

    with b_col1:
        st.metric("True Field Mean VARI",    f"{true_m:.4f}" if true_m is not None else "—")
    with b_col2:
        st.metric("Sampled Mean VARI",       f"{samp_m:.4f}" if samp_m is not None else "—")
    with b_col3:
        delta_str = (f"{abs_b:+.4f}" if abs_b is not None else "—")
        st.metric("Absolute Bias",           delta_str)
    with b_col4:
        rel_str = (f"{rel_b:+.2f}%" if rel_b is not None else "—")
        st.metric("Relative Bias",           rel_str)

    if abs_b is not None and abs(abs_b) > 0.01:
        st.warning(
            f"⚠ Sampling bias of **{abs_b:+.4f} VARI** ({rel_b:+.2f}%) detected. "
            "This occurs because the captured pixels over-represent or under-represent "
            "certain vegetation conditions. Increase coverage to reduce bias."
        )


# ── Tab 3: 3D Terrain ─────────────────────────────────────────────────────────
with tab_terrain:
    st.plotly_chart(
        fig_terrain(
            field["dem"], field["vari"],
            mask=R["cov_mask"] if R["cov_mask"].any() else None,
        ),
        use_container_width=True,
    )
    st.caption(
        "Surface elevation from the synthetic DEM (0–8 m relative relief). "
        "Colour shows VARI from sampled pixels only — dark areas are unobserved. "
        "Rotate with mouse drag; zoom with scroll."
    )


# ── Tab 4: Analytics ──────────────────────────────────────────────────────────
with tab_analytics:

    # Zonal statistics side by side
    st.markdown("#### 📊 Zonal Statistics — True vs Sampled")
    st.markdown(
        "Per-quadrant VARI statistics. "
        "Differences between True and Sampled columns reveal where "
        "coverage gaps introduce the most bias."
    )

    import pandas as pd

    z_col1, z_col2 = st.columns(2)
    with z_col1:
        st.markdown("**True Field (all pixels)**")
        df_true = pd.DataFrame(R["true_zonal"])
        st.dataframe(df_true, use_container_width=True, hide_index=True)

    with z_col2:
        st.markdown("**Sampled Only (captured pixels)**")
        df_samp = pd.DataFrame(R["samp_zonal"])
        st.dataframe(df_samp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### ⚡ Battery Cost Breakdown")
    st.plotly_chart(
        fig_battery_breakdown(batt),
        use_container_width=True,
    )

    # Full mission parameter summary
    st.markdown("---")
    st.markdown("#### 📋 Full Mission Summary")

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Field area | {FIELD_ACRES} acres ({FIELD_SIDE_M:.1f} × {FIELD_SIDE_M:.1f} m) |
| Altitude | {altitude_m} m |
| GSD | {R['gsd_cm']:.2f} cm/px |
| Footprint radius | {R['fp_radius'] * CELL_SIZE_M:.1f} m |
| Flight speed | {speed_ms} m/s |
| Capture interval | {interval_s} s |
| Total path length | {R['path_len_m']:.0f} m |
| Full mission duration | {R['full_dur_s']/60:.2f} min |
""")

    with summary_col2:
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Overlap setting | {overlap_pct}% |
| Estimated actual overlap | {est_overlap*100:.1f}% |
| Total waypoints | {len(R['waypoints'])} |
| Lane-reversal turns | {R['n_turns']} |
| Scheduled captures | {len(R['all_captures'])} |
| Completed captures | {len(R['visible_caps'])} |
| Area covered | {cov_frac*100:.2f}% |
| Mission feasible | {'✅ Yes' if batt['mission_feasible'] else '❌ No (battery cutoff)'} |
""")

    # Battery detail
    st.markdown(f"""
**Battery breakdown:**
- Base flight cost: **{batt['base_cost_pct']:.1f}%**
- Turn penalty ({R['n_turns']} turns × 0.8%): **{batt['turn_cost_pct']:.1f}%**
- Wind resistance cost: **{batt['wind_cost_pct']:.1f}%**
- **Total required: {batt['used_pct']:.1f}% / Available: {batt['available_pct']:.0f}%**
- Battery used at current progress: **{R['batt_now_pct']:.1f}%**
- Efficiency score: **{R['eff_score']:.3f}**
""")