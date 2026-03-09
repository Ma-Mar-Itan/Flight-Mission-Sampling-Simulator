"""
Flight Mission & Sampling Simulator
====================================
Realistic drone survey simulation demonstrating how mission parameters
affect coverage quality, reconstruction fidelity, and statistical bias
in agricultural remote sensing.

Architecture
------------
Section 1 — Synthetic Field Generator
Section 2 — Mission Geometry  (lawnmower path, footprint, GSD)
Section 3 — Drone Motion Engine (interpolation, capture scheduling)
Section 4 — Battery Model (time/turn/wind cost, cutoff logic)
Section 5 — Reconstruction Engine (footprint masking, overlap degradation)
Section 6 — Statistics Engine (zonal stats, sampling-bias comparison)
Section 7 — Plotting Helpers (returns Plotly JSON-serialisable figure dicts)
Section 8 — Flask API Routes + single-file HTML frontend

Model Assumptions
-----------------
* Camera: Sony RX1R II equivalent (35.9 mm sensor, 35 mm focal length, 8 Mpx wide).
* Footprint radius = altitude * tan(HFOV/2), expressed in raster pixels.
* GSD (cm/px) = (sensor_mm * altitude_m * 100) / (focal_mm * image_width_px).
* Battery: base_cost = time * POWER_PER_SECOND; turn_cost = turns * 0.8 %;
           wind_cost = time * 0.03 * wind_factor.
* Overlap < 70 % introduces stochastic strip-aligned reconstruction gaps.
* All analytics have two modes: truth (full raster) and sampled (captured only).
"""

import math
import json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from flask import Flask, request, jsonify, render_template_string


# ---------------------------------------------------------------------------
# JSON SAFETY HELPER
# ---------------------------------------------------------------------------

def _sanitise(obj):
    """
    Recursively replace float NaN / Inf with None so the response is
    valid JSON (Python's json module emits bare NaN which browsers reject).
    Also converts numpy scalars to plain Python types.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    return obj

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------------------------

FIELD_ACRES      = 4.0
FIELD_SIDE_M     = math.sqrt(FIELD_ACRES * 4046.86)   # ~127.6 m per side
RASTER_ROWS      = 128
RASTER_COLS      = 128
CELL_SIZE_M      = FIELD_SIDE_M / RASTER_COLS          # ~0.997 m / pixel

# Camera model (Sony RX1R II, 35 mm fixed lens)
SENSOR_WIDTH_MM  = 35.9
FOCAL_LENGTH_MM  = 35.0
IMAGE_WIDTH_PX   = 8000
HFOV_DEG         = 2 * math.degrees(math.atan(SENSOR_WIDTH_MM / (2 * FOCAL_LENGTH_MM)))

# Battery power model constants
POWER_PER_SECOND  = 0.055   # % per second nominal
TURN_PENALTY_PCT  = 0.8     # % per lane reversal
WIND_FACTOR_SCALE = 0.03    # extra % per second per wind unit

# Overlap thresholds
OVERLAP_IDEAL = 0.80   # below this, reconstruction gaps appear

# Quadrant zone labels
ZONE_NAMES = {0: "NW", 1: "NE", 2: "SW", 3: "SE"}

app = Flask(__name__)


# =============================================================================
# SECTION 1 — SYNTHETIC FIELD GENERATOR
# =============================================================================

def _smooth_noise(rows: int, cols: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian-smoothed random field normalised to [0, 1]."""
    raw      = rng.random((rows, cols)).astype(np.float32)
    smoothed = gaussian_filter(raw, sigma=scale)
    lo, hi   = smoothed.min(), smoothed.max()
    return (smoothed - lo) / (hi - lo + 1e-9)


def generate_field_raster(seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate synthetic 4-acre agricultural field rasters.

    Returns dict:
      vari  — Visible Atmospherically Resistant Index  [float32, -0.3..0.6]
      ndvi  — Normalised Difference Vegetation Index   [float32, -1..1]
      dem   — Digital Elevation Model relative (m)     [float32]
      rgb   — Natural-colour orthomosaic               [uint8, H x W x 3]
      zones — Quadrant labels 0-3                      [int32]
    """
    rng = np.random.default_rng(seed)

    # DEM: rolling terrain with east-west slope
    base_terrain = _smooth_noise(RASTER_ROWS, RASTER_COLS, scale=20.0, rng=rng)
    ew_slope     = np.linspace(0, 1, RASTER_COLS)[np.newaxis, :]
    dem          = (0.6 * base_terrain + 0.4 * ew_slope) * 8.0   # 0-8 m

    # Crop health / VARI
    crop_base  = _smooth_noise(RASTER_ROWS, RASTER_COLS, scale=12.0, rng=rng)
    micro_var  = _smooth_noise(RASTER_ROWS, RASTER_COLS, scale=6.0,  rng=rng)

    # Stressed patch (disease / waterlogging)
    cx, cy     = int(RASTER_ROWS * 0.35), int(RASTER_COLS * 0.60)
    rr, cc     = np.ogrid[:RASTER_ROWS, :RASTER_COLS]
    stress_map = np.exp(-((rr - cx)**2 + (cc - cy)**2) / (2 * 15**2))
    vari_raw   = crop_base - 0.5 * micro_var - 0.6 * stress_map
    vari       = np.clip(vari_raw / (vari_raw.max() + 1e-9), -0.3, 0.6).astype(np.float32)

    # NDVI (correlated but independently noisy)
    ndvi = np.clip(vari * 1.2 + 0.1 + rng.normal(0, 0.03, vari.shape), -1.0, 1.0).astype(np.float32)

    # RGB from VARI
    vari_01 = (vari - vari.min()) / (vari.max() - vari.min() + 1e-9)
    r_ch    = np.clip(0.35 + 0.25 * (1 - vari_01) + 0.04 * rng.random(vari.shape), 0, 1)
    g_ch    = np.clip(0.28 + 0.40 * vari_01        + 0.04 * rng.random(vari.shape), 0, 1)
    b_ch    = np.clip(0.10 + 0.10 * vari_01        + 0.04 * rng.random(vari.shape), 0, 1)
    rgb     = (np.stack([r_ch, g_ch, b_ch], axis=-1) * 255).astype(np.uint8)

    # Zone labels (NW=0, NE=1, SW=2, SE=3)
    zones = np.zeros((RASTER_ROWS, RASTER_COLS), dtype=np.int32)
    h2, w2          = RASTER_ROWS // 2, RASTER_COLS // 2
    zones[:h2, w2:] = 1
    zones[h2:, :w2] = 2
    zones[h2:, w2:] = 3

    return {"vari": vari, "ndvi": ndvi, "dem": dem.astype(np.float32), "rgb": rgb, "zones": zones}


# =============================================================================
# SECTION 2 — MISSION GEOMETRY
# =============================================================================

def compute_footprint_radius_px(altitude_m: float) -> float:
    """
    Ground footprint radius in raster pixels.
    footprint_half_width = altitude * tan(HFOV/2)
    Higher altitude -> larger radius -> coarser coverage per photo.
    """
    half_width_m = altitude_m * math.tan(math.radians(HFOV_DEG / 2.0))
    return half_width_m / CELL_SIZE_M


def compute_gsd_cm(altitude_m: float) -> float:
    """
    Ground Sampling Distance in cm/pixel.
    GSD = (sensor_mm * altitude_m * 100) / (focal_mm * image_px)
    Higher altitude -> larger GSD -> coarser detail.
    """
    return (SENSOR_WIDTH_MM * altitude_m * 100.0) / (FOCAL_LENGTH_MM * IMAGE_WIDTH_PX)


def compute_line_spacing_px(altitude_m: float, overlap_frac: float) -> float:
    """
    Inter-strip spacing in pixels.
    spacing = footprint_diameter * (1 - sidelap_fraction)
    Lower overlap -> wider spacing -> more gaps.
    """
    diameter_px = 2.0 * compute_footprint_radius_px(altitude_m)
    return max(2.0, diameter_px * (1.0 - overlap_frac))


def generate_lawnmower_path(
    altitude_m: float,
    overlap_frac: float,
    margin_px: float = 4.0,
) -> List[Tuple[float, float]]:
    """
    Generate a boustrophedon (lawnmower) survey path.

    Runs north-south strips alternating direction.
    Returns list of (col, row) waypoints in pixel coordinates.
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
    """Count lane-reversal turn events (end of each strip)."""
    turns = 0
    for i in range(1, len(waypoints) - 1):
        dy_prev = waypoints[i][1]   - waypoints[i-1][1]
        dy_next = waypoints[i+1][1] - waypoints[i][1]
        if abs(dy_prev) > 0.1 and abs(dy_next) < 0.1:
            turns += 1
    return max(turns, 0)


# =============================================================================
# SECTION 3 — DRONE MOTION ENGINE
# =============================================================================

def compute_segment_lengths(waypoints: List[Tuple[float, float]]) -> List[float]:
    """Euclidean pixel-length of each consecutive waypoint segment."""
    return [
        math.hypot(waypoints[i+1][0] - waypoints[i][0],
                   waypoints[i+1][1] - waypoints[i][1])
        for i in range(len(waypoints) - 1)
    ]


def total_path_length_m(waypoints: List[Tuple[float, float]]) -> float:
    """Total path length in metres."""
    return sum(compute_segment_lengths(waypoints)) * CELL_SIZE_M


def mission_duration_s(waypoints: List[Tuple[float, float]], speed_ms: float) -> float:
    """Estimated mission duration in seconds."""
    return total_path_length_m(waypoints) / max(speed_ms, 0.1)


def interpolate_position(
    waypoints: List[Tuple[float, float]],
    seg_lengths: List[float],
    progress: float,
) -> Tuple[float, float]:
    """
    Linear interpolation of drone position.

    progress : float [0, 1] — fraction of total path completed.
    Returns (col, row) in pixel coordinates.
    """
    if not waypoints:
        return (0.0, 0.0)
    total = sum(seg_lengths)
    if total == 0:
        return waypoints[0]

    target  = progress * total
    accum   = 0.0
    for i, seg_len in enumerate(seg_lengths):
        next_accum = accum + seg_len
        if next_accum >= target or i == len(seg_lengths) - 1:
            t   = (target - accum) / max(seg_len, 1e-9)
            t   = min(max(t, 0.0), 1.0)
            col = waypoints[i][0] + t * (waypoints[i+1][0] - waypoints[i][0])
            row = waypoints[i][1] + t * (waypoints[i+1][1] - waypoints[i][1])
            return (col, row)
        accum = next_accum

    return waypoints[-1]


def schedule_capture_events(
    waypoints: List[Tuple[float, float]],
    seg_lengths: List[float],
    speed_ms: float,
    interval_s: float,
) -> List[Dict[str, float]]:
    """
    Generate photo capture events spaced interval_s apart in time.

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
        events.append({"t": round(t, 3), "col": col, "row": row, "progress": progress})
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
      base_cost = mission_time_s * POWER_PER_SECOND
      turn_cost = n_turns * TURN_PENALTY_PCT
      wind_cost = mission_time_s * WIND_FACTOR_SCALE * wind_factor
      used_pct  = base_cost + turn_cost + wind_cost

    If used_pct > battery_capacity_pct, the mission is cut short.
    fraction_possible = max completable fraction of the path [0, 1].
    """
    per_sec = POWER_PER_SECOND + WIND_FACTOR_SCALE * wind_factor
    base_cost = mission_time_s * POWER_PER_SECOND
    turn_cost = n_turns        * TURN_PENALTY_PCT
    wind_cost = mission_time_s * WIND_FACTOR_SCALE * wind_factor
    used_pct  = base_cost + turn_cost + wind_cost
    feasible  = used_pct <= battery_capacity_pct

    if feasible:
        fraction_possible = 1.0
    else:
        available_for_cruise = max(0.0, battery_capacity_pct - turn_cost)
        max_time_s           = available_for_cruise / max(per_sec, 1e-9)
        fraction_possible    = min(1.0, max_time_s / max(mission_time_s, 1e-9))

    return {
        "used_pct":               round(used_pct, 2),
        "available_pct":          battery_capacity_pct,
        "mission_feasible":       feasible,
        "fraction_possible":      round(fraction_possible, 4),
        "base_cost_pct":          round(base_cost, 2),
        "turn_cost_pct":          round(turn_cost, 2),
        "wind_cost_pct":          round(wind_cost, 2),
    }


def apply_battery_cutoff(
    capture_events: List[Dict],
    fraction_possible: float,
) -> List[Dict]:
    """Return only events within the battery-achievable path fraction."""
    return [ev for ev in capture_events if ev["progress"] <= fraction_possible + 1e-6]


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

    Each capture paints a soft circular footprint (weight 1 at centre, 0 at edge).
    The accumulated weight map is thresholded to binary.

    When overlap_frac < OVERLAP_IDEAL, stochastic column-aligned gaps are introduced
    to simulate photogrammetric reconstruction failure from insufficient image overlap.

    Returns bool array (RASTER_ROWS x RASTER_COLS).
    """
    weight = np.zeros((RASTER_ROWS, RASTER_COLS), dtype=np.float32)
    r      = max(footprint_radius_px, 1.0)
    rr, cc = np.ogrid[:RASTER_ROWS, :RASTER_COLS]

    for ev in capture_events:
        dist2    = (rr - ev["row"])**2 + (cc - ev["col"])**2
        fp_w     = np.clip(1.0 - dist2 / (r**2), 0.0, 1.0)
        weight   = np.maximum(weight, fp_w)

    mask = weight > 0.05

    # Overlap-gap degradation: punch inter-strip holes when overlap is low
    if overlap_frac < OVERLAP_IDEAL and mask.any():
        rng          = np.random.default_rng(rng_seed + 7)
        gap_severity = (OVERLAP_IDEAL - overlap_frac) / OVERLAP_IDEAL   # [0, 1]
        gap_noise    = rng.random((RASTER_ROWS, RASTER_COLS)).astype(np.float32)
        # Column-banding mimics inter-strip reconstruction gaps
        col_band     = (np.abs(np.sin(np.linspace(0, math.pi * 10, RASTER_COLS))) * gap_severity)[np.newaxis, :]
        gap_holes    = gap_noise < (col_band * 0.75)
        mask         = mask & ~gap_holes

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

    mask=None  -> truth mode (all pixels)
    mask=array -> sampled mode (captured pixels only)
    """
    results = []
    for zone_id, zone_name in ZONE_NAMES.items():
        zone_px = zones == zone_id
        active  = zone_px & mask if mask is not None else zone_px
        vals    = vari[active]
        if len(vals) == 0:
            results.append({"zone": zone_name, "mean": None, "std": None,
                             "min": None, "max": None, "n_pixels": 0})
        else:
            results.append({
                "zone":     zone_name,
                "mean":     round(float(vals.mean()), 4),
                "std":      round(float(vals.std()),  4),
                "min":      round(float(vals.min()),  4),
                "max":      round(float(vals.max()),  4),
                "n_pixels": int(len(vals)),
            })
    return results


def compute_sampling_bias(
    vari: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Compare true field mean VARI vs sampled (captured-only) mean.
    Quantifies statistical bias from incomplete coverage.
    """
    true_vals    = vari.ravel()
    sampled_vals = vari[mask].ravel() if mask.any() else np.array([])

    true_mean    = float(true_vals.mean())
    sampled_mean = float(sampled_vals.mean()) if len(sampled_vals) > 0 else float("nan")

    if math.isnan(sampled_mean):
        return {"true_mean": round(true_mean, 4), "sampled_mean": None,
                "absolute_bias": None, "relative_bias": None,
                "n_sampled_px": 0, "n_total_px": int(vari.size)}

    bias = sampled_mean - true_mean
    return {
        "true_mean":     round(true_mean,    4),
        "sampled_mean":  round(sampled_mean, 4),
        "absolute_bias": round(bias,         4),
        "relative_bias": round(bias / (abs(true_mean) + 1e-9) * 100, 2),
        "n_sampled_px":  int(mask.sum()),
        "n_total_px":    int(vari.size),
    }


def compute_efficiency_score(coverage_fraction: float, battery_used_pct: float) -> float:
    """
    Efficiency Score = (coverage % / battery used %).
    Higher = more field covered per unit battery.
    """
    if battery_used_pct <= 0:
        return 0.0
    return round((coverage_fraction * 100.0) / battery_used_pct, 4)


def compute_estimated_overlap(fp_radius_px: float, line_spacing_px: float) -> float:
    """Actual overlap fraction from footprint and line spacing."""
    diameter = 2.0 * fp_radius_px
    return float(np.clip(1.0 - line_spacing_px / max(diameter, 1e-9), 0.0, 1.0))


# =============================================================================
# SECTION 7 — PLOTTING HELPERS
# =============================================================================

def _heatmap_trace(
    z: np.ndarray,
    colorscale: str = "RdYlGn",
    zmin: float = -0.3,
    zmax: float = 0.6,
    mask: Optional[np.ndarray] = None,
    name: str = "VARI",
    show_scale: bool = True,
) -> dict:
    display = z.astype(float).copy()
    if mask is not None:
        display[~mask] = float("nan")
    return {
        "type":           "heatmap",
        "z":              display.tolist(),
        "colorscale":     colorscale,
        "zmin":           zmin,
        "zmax":           zmax,
        "showscale":      show_scale,
        "colorbar":       {"title": name, "thickness": 10, "len": 0.55,
                           "tickfont": {"color": "#aaa"}, "titlefont": {"color": "#aaa"}},
        "hovertemplate":  f"{name}: %{{z:.3f}}<extra></extra>",
    }


def build_mission_map_figure(
    waypoints: List[Tuple[float, float]],
    all_captures: List[Dict],
    visible_caps: List[Dict],
    current_pos: Optional[Tuple[float, float]],
    vari: np.ndarray,
    batt: Dict,
) -> dict:
    """
    Live Mission View: VARI background + flight path + captures + drone position.
    """
    traces = []

    # 1. Background VARI heatmap (semi-transparent reference)
    bg = _heatmap_trace(vari, show_scale=False, name="VARI")
    bg["opacity"] = 0.70
    bg["name"]    = "VARI (ground truth)"
    traces.append(bg)

    # 2. Full planned path (grey dotted)
    if waypoints:
        traces.append({
            "type": "scatter",
            "x":    [w[0] for w in waypoints],
            "y":    [w[1] for w in waypoints],
            "mode": "lines",
            "line": {"color": "rgba(180,180,180,0.4)", "width": 1.2, "dash": "dot"},
            "name": "Planned path",
            "hoverinfo": "skip",
        })

    # 3. All scheduled capture points (hollow)
    if all_captures:
        traces.append({
            "type":         "scatter",
            "x":            [e["col"] for e in all_captures],
            "y":            [e["row"] for e in all_captures],
            "mode":         "markers",
            "marker":       {"symbol": "circle-open", "size": 4,
                             "color": "rgba(100,200,255,0.30)"},
            "name":         "Scheduled captures",
            "hovertemplate":"t=%{customdata:.1f}s<extra></extra>",
            "customdata":   [e["t"] for e in all_captures],
        })

    # 4. Completed captures (filled)
    if visible_caps:
        traces.append({
            "type":         "scatter",
            "x":            [e["col"] for e in visible_caps],
            "y":            [e["row"] for e in visible_caps],
            "mode":         "markers",
            "marker":       {"symbol": "circle", "size": 6, "color": "#00b4d8",
                             "line": {"width": 1, "color": "white"}},
            "name":         f"Captured ({len(visible_caps)})",
            "hovertemplate":"t=%{customdata:.1f}s  (%{x:.1f},%{y:.1f})<extra></extra>",
            "customdata":   [e["t"] for e in visible_caps],
        })

    # 5. Current drone position
    if current_pos:
        traces.append({
            "type":         "scatter",
            "x":            [current_pos[0]],
            "y":            [current_pos[1]],
            "mode":         "markers",
            "marker":       {"symbol": "star", "size": 18, "color": "#ff6b35",
                             "line": {"width": 2, "color": "white"}},
            "name":         "Drone",
            "hovertemplate":"Drone @ (%{x:.1f}, %{y:.1f})<extra></extra>",
        })

    # 6. Battery cutoff marker
    if not batt.get("mission_feasible", True) and visible_caps:
        last = visible_caps[-1]
        traces.append({
            "type":         "scatter",
            "x":            [last["col"]],
            "y":            [last["row"]],
            "mode":         "markers+text",
            "marker":       {"symbol": "x", "size": 20, "color": "red",
                             "line": {"width": 3, "color": "white"}},
            "text":         ["⚡"],
            "textposition": "top center",
            "textfont":     {"color": "red", "size": 16},
            "name":         "Battery cutoff",
            "hovertemplate":"Battery depleted here<extra></extra>",
        })

    layout = {
        "title":        {"text": "Live Mission View", "font": {"color": "#e0e0e0", "size": 14}},
        "paper_bgcolor":"#0d1117",
        "plot_bgcolor": "#0d1117",
        "xaxis":        {"range": [0, RASTER_COLS], "showgrid": False,
                         "zeroline": False, "color": "#555", "title": "Column (px)"},
        "yaxis":        {"range": [RASTER_ROWS, 0], "showgrid": False,
                         "zeroline": False, "color": "#555", "title": "Row (px)",
                         "scaleanchor": "x"},
        "legend":       {"bgcolor": "rgba(0,0,0,0.5)", "font": {"color": "#ccc", "size": 10},
                         "x": 0.01, "y": 0.99},
        "margin":       {"l": 50, "r": 10, "t": 40, "b": 40},
        "height":       450,
    }
    return {"data": traces, "layout": layout}


def build_comparison_figures(
    vari: np.ndarray,
    mask: np.ndarray,
) -> Tuple[dict, dict]:
    """Return (true_fig, reconstructed_fig) for the side-by-side comparison."""

    def _fig(z, title, apply_mask=None):
        trace  = _heatmap_trace(z, mask=apply_mask, show_scale=True)
        layout = {
            "title":         {"text": title, "font": {"color": "#e0e0e0", "size": 13}},
            "paper_bgcolor": "#0d1117",
            "plot_bgcolor":  "#0d1117",
            "xaxis":         {"showgrid": False, "zeroline": False,
                              "showticklabels": False, "color": "#555"},
            "yaxis":         {"showgrid": False, "zeroline": False,
                              "showticklabels": False, "autorange": "reversed",
                              "scaleanchor": "x", "color": "#555"},
            "margin":        {"l": 10, "r": 10, "t": 40, "b": 10},
            "height":        380,
        }
        return {"data": [trace], "layout": layout}

    true_fig  = _fig(vari, "True Farm (full VARI)", apply_mask=None)
    recon_fig = _fig(vari, "Reconstructed Map (sampled only)", apply_mask=mask)

    # Dark overlay for unobserved pixels
    if not mask.all():
        unobs = np.where(mask, float("nan"), 0.0).tolist()
        recon_fig["data"].append({
            "type":       "heatmap",
            "z":          unobs,
            "colorscale": [[0, "rgba(8,8,12,0.93)"], [1, "rgba(8,8,12,0.93)"]],
            "showscale":  False,
            "hovertemplate": "Not observed<extra></extra>",
        })

    return true_fig, recon_fig


def build_terrain_figure(
    dem: np.ndarray,
    vari: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """3D surface DEM coloured by VARI (sampled pixels only if mask given)."""
    vari_display = vari.copy()
    if mask is not None:
        vari_display = np.where(mask, vari_display, float("nan"))

    step    = 2   # downsample for 3D performance
    z_vals  = dem[::step, ::step].tolist()
    col_val = vari_display[::step, ::step].tolist()

    trace = {
        "type":         "surface",
        "z":            z_vals,
        "surfacecolor": col_val,
        "colorscale":   "RdYlGn",
        "cmin":         -0.3,
        "cmax":          0.6,
        "colorbar":     {"title": "VARI", "thickness": 10,
                         "tickfont": {"color": "#aaa"}, "titlefont": {"color": "#aaa"}},
        "hovertemplate":"Elev: %{z:.2f} m<br>VARI: %{surfacecolor:.3f}<extra></extra>",
        "lighting":     {"ambient": 0.7, "diffuse": 0.5, "roughness": 0.5},
    }

    layout = {
        "title":         {"text": "3D Terrain — DEM + VARI (sampled)",
                          "font": {"color": "#e0e0e0", "size": 14}},
        "paper_bgcolor": "#0d1117",
        "scene":         {
            "bgcolor": "#0d1117",
            "xaxis":   {"showgrid": False, "color": "#444", "title": ""},
            "yaxis":   {"showgrid": False, "color": "#444", "title": ""},
            "zaxis":   {"showgrid": False, "color": "#444", "title": "Elev (m)"},
            "camera":  {"eye": {"x": 1.4, "y": -1.4, "z": 1.1}},
        },
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
        "height": 450,
    }
    return {"data": [trace], "layout": layout}


# =============================================================================
# SECTION 8 — FLASK APP
# =============================================================================

_FIELD = generate_field_raster(seed=42)   # pre-generated once at startup


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Main simulation endpoint.

    Input JSON:
      altitude_m, overlap_frac, speed_ms, interval_s,
      battery_cap, wind_factor, slider_progress [0-1], seed

    Returns:
      figures  (mission, true_map, recon_map, terrain)
      mission_stats, battery, efficiency, vari_bias,
      true_zonal, sampled_zonal
    """
    body = request.get_json(force=True)

    altitude_m      = max(10.0, min(120.0, float(body.get("altitude_m",      30.0))))
    overlap_frac    = max(0.20, min(0.95,  float(body.get("overlap_frac",     0.75))))
    speed_ms        = max(1.0,  min(20.0,  float(body.get("speed_ms",          5.0))))
    interval_s      = max(0.5,  min(10.0,  float(body.get("interval_s",        2.0))))
    battery_cap     = max(10.0, min(100.0, float(body.get("battery_cap",     100.0))))
    wind_factor     = max(0.0,  min(5.0,   float(body.get("wind_factor",       1.0))))
    slider_progress = max(0.0,  min(1.0,   float(body.get("slider_progress",   1.0))))
    seed            = int(body.get("seed", 42))

    field = _FIELD

    # ── Section 2: Geometry ──────────────────────────────────────
    waypoints   = generate_lawnmower_path(altitude_m, overlap_frac)
    seg_lengths = compute_segment_lengths(waypoints)
    n_turns     = count_turns(waypoints)
    fp_radius   = compute_footprint_radius_px(altitude_m)
    gsd_cm      = compute_gsd_cm(altitude_m)
    spacing_px  = compute_line_spacing_px(altitude_m, overlap_frac)
    est_overlap = compute_estimated_overlap(fp_radius, spacing_px)
    path_len_m  = total_path_length_m(waypoints)
    full_dur_s  = mission_duration_s(waypoints, speed_ms)

    # ── Section 3: Capture schedule ──────────────────────────────
    all_captures = schedule_capture_events(waypoints, seg_lengths, speed_ms, interval_s)

    # ── Section 4: Battery ────────────────────────────────────────
    batt         = compute_battery_budget(full_dur_s, n_turns, wind_factor, battery_cap)
    frac_ok      = batt["fraction_possible"]

    # Battery cutoff: only keep events within the battery budget
    completed    = apply_battery_cutoff(all_captures, frac_ok)

    # Progress slider restricts visible events further (within battery-ok range)
    slider_frac  = slider_progress * frac_ok
    visible_caps = [ev for ev in completed if ev["progress"] <= slider_frac + 1e-6]

    # Drone current position at slider_frac
    current_pos  = interpolate_position(waypoints, seg_lengths, slider_frac) if waypoints else None

    # ── Section 5: Reconstruction ─────────────────────────────────
    cov_mask     = build_coverage_mask(visible_caps, fp_radius, overlap_frac, rng_seed=seed)
    cov_frac     = compute_coverage_fraction(cov_mask)

    # ── Section 6: Statistics ──────────────────────────────────────
    true_zonal   = compute_zonal_stats(field["vari"], field["zones"], mask=None)
    samp_zonal   = compute_zonal_stats(field["vari"], field["zones"], mask=cov_mask)
    bias_stats   = compute_sampling_bias(field["vari"], cov_mask)
    batt_used_now = round(min(batt["used_pct"], battery_cap) * slider_progress, 2)
    eff_score    = compute_efficiency_score(cov_frac, max(batt_used_now, 0.1))

    # ── Section 7: Figures ─────────────────────────────────────────
    mission_fig           = build_mission_map_figure(
        waypoints, all_captures, visible_caps, current_pos, field["vari"], batt)
    true_fig, recon_fig   = build_comparison_figures(field["vari"], cov_mask)
    terrain_fig           = build_terrain_figure(
        field["dem"], field["vari"], mask=cov_mask if cov_mask.any() else None)

    payload = {
        "figures": {
            "mission":   mission_fig,
            "true_map":  true_fig,
            "recon_map": recon_fig,
            "terrain":   terrain_fig,
        },
        "mission_stats": {
            "n_waypoints":        len(waypoints),
            "n_captures_total":   len(all_captures),
            "n_captures_done":    len(visible_caps),
            "path_length_m":      round(path_len_m, 1),
            "full_duration_min":  round(full_dur_s / 60, 2),
            "elapsed_time_min":   round(full_dur_s * slider_frac / 60, 2),
            "n_turns":            n_turns,
            "altitude_m":         altitude_m,
            "gsd_cm":             round(gsd_cm, 2),
            "footprint_radius_m": round(fp_radius * CELL_SIZE_M, 1),
            "est_overlap_pct":    round(est_overlap * 100, 1),
            "coverage_pct":       round(cov_frac * 100, 2),
            "speed_ms":           speed_ms,
            "interval_s":         interval_s,
        },
        "battery": {
            **batt,
            "battery_used_now_pct": batt_used_now,
        },
        "efficiency": {
            "efficiency_score": eff_score,
            "coverage_pct":     round(cov_frac * 100, 2),
            "battery_used_pct": batt_used_now,
        },
        "vari_bias":     bias_stats,
        "true_zonal":    true_zonal,
        "sampled_zonal": samp_zonal,
    }

    # Sanitise before serialising: replace NaN/Inf with null (valid JSON)
    clean = _sanitise(payload)
    return app.response_class(
        response=json.dumps(clean),
        status=200,
        mimetype="application/json",
    )


# =============================================================================
# HTML FRONTEND — single file, inline CSS + JS
# =============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Flight Mission &amp; Sampling Simulator</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
:root {
  --bg:      #0d1117; --surface: #161b22; --border: #30363d;
  --accent:  #00e5ff; --accent2: #7c3aed; --warn: #f97316;
  --danger:  #ef4444; --success: #22c55e; --text: #c9d1d9;
  --muted:   #6e7681; --mono: 'Courier New', monospace;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
     font-size:13px;display:flex;height:100vh;overflow:hidden;}

/* ── Sidebar ─────────────────────────────────────────────────── */
#sidebar{width:272px;min-width:272px;background:var(--surface);
  border-right:1px solid var(--border);display:flex;flex-direction:column;
  overflow-y:auto;padding:14px 12px 20px;gap:10px;}
.logo{font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);padding-bottom:8px;border-bottom:1px solid var(--border);}
.logo span{color:var(--muted);font-weight:400;}
.sec-title{font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);margin-top:2px;}
.ctrl-group{display:flex;flex-direction:column;gap:7px;}
.ctrl{display:flex;flex-direction:column;gap:2px;}
.ctrl label{font-size:11px;color:var(--muted);display:flex;justify-content:space-between;}
.ctrl label span{color:var(--accent);font-family:var(--mono);}
input[type=range]{-webkit-appearance:none;width:100%;height:4px;
  background:var(--border);border-radius:2px;outline:none;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;
  border-radius:50%;background:var(--accent);cursor:pointer;border:2px solid var(--bg);}

#btn-run{background:var(--accent);color:var(--bg);border:none;padding:9px 0;
  border-radius:6px;font-weight:700;font-size:12px;letter-spacing:.06em;
  cursor:pointer;width:100%;transition:opacity .15s;}
#btn-run:hover{opacity:.85;}#btn-run:disabled{opacity:.4;cursor:not-allowed;}

/* ── Battery bar ─────────────────────────────────────────────── */
.batt-wrap{background:var(--bg);border:1px solid var(--border);border-radius:8px;
  padding:9px 11px;display:flex;flex-direction:column;gap:4px;}
.batt-lbl{font-size:10px;color:var(--muted);display:flex;justify-content:space-between;}
.batt-bar{height:7px;border-radius:4px;background:var(--border);overflow:hidden;}
.batt-fill{height:100%;border-radius:4px;background:var(--success);transition:width .3s,background .3s;}
.batt-fill.warn{background:var(--warn);}.batt-fill.danger{background:var(--danger);}

/* ── Progress slider ─────────────────────────────────────────── */
.prog-wrap{background:var(--bg);border:1px solid var(--border);border-radius:8px;
  padding:9px 11px;display:flex;flex-direction:column;gap:5px;}
.prog-lbl{font-size:10px;color:var(--muted);display:flex;justify-content:space-between;}

/* ── Stats grid ──────────────────────────────────────────────── */
.stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;}
.stat-cell{background:var(--bg);border:1px solid var(--border);border-radius:6px;
  padding:6px 8px;display:flex;flex-direction:column;gap:2px;}
.stat-cell .lbl{font-size:9px;text-transform:uppercase;color:var(--muted);letter-spacing:.08em;}
.stat-cell .val{font-size:14px;font-weight:700;color:var(--text);font-family:var(--mono);}
.val.accent{color:var(--accent);}.val.success{color:var(--success);}
.val.warn{color:var(--warn);}.val.danger{color:var(--danger);}

/* ── VARI bias ───────────────────────────────────────────────── */
.bias-block{background:var(--bg);border:1px solid var(--border);border-radius:8px;
  padding:9px 11px;display:flex;flex-direction:column;gap:4px;}
.bias-row{display:flex;justify-content:space-between;align-items:center;}
.bias-key{font-size:10px;color:var(--muted);}
.bias-val{font-size:12px;font-weight:700;font-family:var(--mono);}
.bias-d{font-size:11px;font-weight:700;font-family:var(--mono);}
.bias-d.pos{color:var(--success);}.bias-d.neg{color:var(--danger);}

/* ── Main ────────────────────────────────────────────────────── */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden;}
#tabs{display:flex;gap:2px;padding:10px 14px 0;background:var(--surface);
  border-bottom:1px solid var(--border);}
.tab{padding:7px 15px;border-radius:6px 6px 0 0;cursor:pointer;font-size:12px;
  font-weight:600;color:var(--muted);border:1px solid transparent;border-bottom:none;
  background:transparent;transition:color .15s;}
.tab:hover{color:var(--text);}
.tab.active{color:var(--accent);background:var(--bg);border-color:var(--border);}
#content{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px;}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden;}
.panel-hdr{padding:7px 13px;font-size:10px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;background:rgba(255,255,255,.02);}
.badge{font-size:9px;padding:2px 8px;border-radius:10px;font-weight:700;}
.badge-ok{background:rgba(34,197,94,.15);color:var(--success);}
.badge-w{background:rgba(249,115,22,.15);color:var(--warn);}
.badge-d{background:rgba(239,68,68,.15);color:var(--danger);}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:10px;}

/* ── Efficiency gauge ───────────────────────────────────────── */
.eff-row{display:flex;align-items:center;gap:12px;padding:12px 14px;}
.eff-circle{width:54px;height:54px;border-radius:50%;border:4px solid var(--border);
  display:flex;align-items:center;justify-content:center;font-size:11px;
  font-weight:700;font-family:var(--mono);flex-shrink:0;}
.eff-desc{font-size:11px;color:var(--muted);line-height:1.6;}
.eff-desc strong{color:var(--text);}

/* ── Zone table ─────────────────────────────────────────────── */
.zt{width:100%;border-collapse:collapse;font-size:10px;}
.zt th,.zt td{padding:4px 7px;border-bottom:1px solid var(--border);text-align:right;}
.zt th{color:var(--muted);text-align:center;}
.zt td:first-child{text-align:left;color:var(--accent);font-weight:700;}

/* ── Alerts ──────────────────────────────────────────────────── */
.alert{display:none;padding:7px 13px;font-size:11px;font-weight:600;border-radius:6px;margin:0 12px;}
.alert.batt-alert{display:block;background:rgba(239,68,68,.10);
  border:1px solid rgba(239,68,68,.4);color:#fca5a5;}
.alert.ovlp-alert{display:block;background:rgba(249,115,22,.10);
  border:1px solid rgba(249,115,22,.35);color:#fdba74;}

/* ── Tab panels ─────────────────────────────────────────────── */
.tab-panel{display:none;flex-direction:column;gap:10px;}
.tab-panel.active{display:flex;}

/* ── Loading ─────────────────────────────────────────────────── */
#loading{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);
  backdrop-filter:blur(4px);z-index:999;align-items:center;justify-content:center;
  flex-direction:column;gap:12px;font-size:13px;color:var(--accent);font-weight:700;letter-spacing:.06em;}
#loading.show{display:flex;}
.spin{width:30px;height:30px;border:3px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;animation:spin .7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
</style>
</head>
<body>

<div id="loading"><div class="spin"></div><span>SIMULATING MISSION…</span></div>

<!-- ══════════════ SIDEBAR ═══════════════════════════════════════ -->
<div id="sidebar">
  <div class="logo">Flight Mission Simulator<br><span>4-acre agricultural survey</span></div>

  <div>
    <div class="sec-title">✈ Flight Parameters</div>
    <div class="ctrl-group" style="margin-top:6px;">
      <div class="ctrl">
        <label>Altitude <span id="v-alt">30 m</span></label>
        <input type="range" id="r-alt" min="10" max="120" step="5" value="30">
      </div>
      <div class="ctrl">
        <label>Flight Speed <span id="v-spd">5 m/s</span></label>
        <input type="range" id="r-spd" min="1" max="20" step="1" value="5">
      </div>
      <div class="ctrl">
        <label>Capture Interval <span id="v-int">2 s</span></label>
        <input type="range" id="r-int" min="0.5" max="10" step="0.5" value="2">
      </div>
      <div class="ctrl">
        <label>Overlap <span id="v-ovlp">75 %</span></label>
        <input type="range" id="r-ovlp" min="20" max="95" step="5" value="75">
      </div>
    </div>
  </div>

  <div>
    <div class="sec-title">⚡ Battery</div>
    <div class="ctrl-group" style="margin-top:6px;">
      <div class="ctrl">
        <label>Battery Capacity <span id="v-batt">100 %</span></label>
        <input type="range" id="r-batt" min="10" max="100" step="5" value="100">
      </div>
      <div class="ctrl">
        <label>Wind Resistance <span id="v-wind">1.0×</span></label>
        <input type="range" id="r-wind" min="0" max="5" step="0.5" value="1">
      </div>
    </div>
  </div>

  <div class="batt-wrap">
    <div class="batt-lbl">Battery Used
      <span id="batt-pct-lbl" style="color:var(--success);font-family:var(--mono)">0 %</span>
    </div>
    <div class="batt-bar"><div class="batt-fill" id="batt-fill" style="width:0%"></div></div>
  </div>

  <div class="prog-wrap">
    <div class="prog-lbl">Mission Progress <span id="v-prog">100 %</span></div>
    <input type="range" id="r-prog" min="0" max="100" step="1" value="100">
  </div>

  <button id="btn-run">▶ RUN SIMULATION</button>

  <div>
    <div class="sec-title">📊 Mission Stats</div>
    <div class="stats-grid" style="margin-top:5px;">
      <div class="stat-cell"><span class="lbl">Coverage</span><span class="val accent" id="s-cov">—</span></div>
      <div class="stat-cell"><span class="lbl">Captures</span><span class="val" id="s-cap">—</span></div>
      <div class="stat-cell"><span class="lbl">GSD</span><span class="val" id="s-gsd">—</span></div>
      <div class="stat-cell"><span class="lbl">Overlap est.</span><span class="val" id="s-ovlp">—</span></div>
      <div class="stat-cell"><span class="lbl">Waypoints</span><span class="val" id="s-wpts">—</span></div>
      <div class="stat-cell"><span class="lbl">Turns</span><span class="val" id="s-turns">—</span></div>
      <div class="stat-cell"><span class="lbl">Elapsed</span><span class="val" id="s-time">—</span></div>
      <div class="stat-cell"><span class="lbl">Path length</span><span class="val" id="s-path">—</span></div>
    </div>
  </div>

  <div>
    <div class="sec-title">🌿 VARI Sampling Bias</div>
    <div class="bias-block" style="margin-top:5px;">
      <div class="bias-row"><span class="bias-key">True field mean</span><span class="bias-val" id="b-true">—</span></div>
      <div class="bias-row"><span class="bias-key">Sampled mean</span><span class="bias-val" id="b-samp">—</span></div>
      <div class="bias-row"><span class="bias-key">Absolute bias</span><span class="bias-d" id="b-abs">—</span></div>
      <div class="bias-row"><span class="bias-key">Relative bias</span><span class="bias-d" id="b-rel">—</span></div>
    </div>
  </div>
</div>

<!-- ══════════════ MAIN ══════════════════════════════════════════ -->
<div id="main">
  <div class="alert" id="alert-batt">⚡ BATTERY CUTOFF — Mission terminated early. Map shows partial coverage only.</div>
  <div class="alert" id="alert-ovlp">⚠ LOW OVERLAP — Reconstruction gaps visible. Increase overlap above 70 % for reliable mapping.</div>

  <div id="tabs">
    <div class="tab active" data-tab="mission">🛰 Live Mission</div>
    <div class="tab" data-tab="maps">🗺 True vs Reconstructed</div>
    <div class="tab" data-tab="terrain">🏔 3D Terrain</div>
    <div class="tab" data-tab="analytics">📈 Analytics</div>
  </div>

  <div id="content">

    <!-- Live Mission tab -->
    <div class="tab-panel active" id="tab-mission">
      <div class="panel">
        <div class="panel-hdr">Live Mission View — flight path &amp; capture events
          <span class="badge" id="m-badge">READY</span>
        </div>
        <div id="plt-mission" style="padding:4px;"></div>
      </div>
      <div class="panel">
        <div class="panel-hdr">Mission Efficiency</div>
        <div class="eff-row">
          <div class="eff-circle" id="eff-circle">—</div>
          <div class="eff-desc">
            <strong>Efficiency = Coverage % / Battery Used %</strong><br>
            Higher = more field covered per unit battery.<br>
            <span id="eff-detail" style="color:var(--muted)">Run simulation to compute.</span>
          </div>
        </div>
      </div>
    </div>

    <!-- True vs Reconstructed tab -->
    <div class="tab-panel" id="tab-maps">
      <div class="two-col">
        <div class="panel">
          <div class="panel-hdr">✅ True Farm (full VARI)</div>
          <div id="plt-true" style="padding:4px;"></div>
        </div>
        <div class="panel">
          <div class="panel-hdr">🛰 Reconstructed Map (captured only)
            <span class="badge" id="r-badge">—</span>
          </div>
          <div id="plt-recon" style="padding:4px;"></div>
        </div>
      </div>
    </div>

    <!-- 3D Terrain tab -->
    <div class="tab-panel" id="tab-terrain">
      <div class="panel">
        <div class="panel-hdr">3D Terrain — DEM coloured by sampled VARI</div>
        <div id="plt-terrain" style="padding:4px;"></div>
      </div>
    </div>

    <!-- Analytics tab -->
    <div class="tab-panel" id="tab-analytics">
      <div class="two-col">
        <div class="panel">
          <div class="panel-hdr">Zonal Stats — True Field</div>
          <div style="padding:12px;">
            <table class="zt" id="t-true">
              <thead><tr><th>Zone</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Px</th></tr></thead>
              <tbody><tr><td colspan="6" style="color:var(--muted);text-align:center;padding:12px;">Run simulation</td></tr></tbody>
            </table>
          </div>
        </div>
        <div class="panel">
          <div class="panel-hdr">Zonal Stats — Sampled Only</div>
          <div style="padding:12px;">
            <table class="zt" id="t-samp">
              <thead><tr><th>Zone</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Px</th></tr></thead>
              <tbody><tr><td colspan="6" style="color:var(--muted);text-align:center;padding:12px;">Run simulation</td></tr></tbody>
            </table>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-hdr">Battery Cost Breakdown</div>
        <div id="plt-batt" style="padding:4px;"></div>
      </div>
    </div>

  </div><!-- /content -->
</div><!-- /main -->

<script>
const CFG = {responsive:true, displayModeBar:false};
let lastData = null;

// ── Slider live labels ────────────────────────────────────────────
function wire(id,lblId,fmt){
  const el=document.getElementById(id),lb=document.getElementById(lblId);
  const upd=()=>{lb.textContent=fmt(parseFloat(el.value));};
  el.addEventListener('input',upd); upd();
}
wire('r-alt','v-alt', v=>v+' m');
wire('r-spd','v-spd', v=>v+' m/s');
wire('r-int','v-int', v=>v+' s');
wire('r-ovlp','v-ovlp', v=>v+' %');
wire('r-batt','v-batt', v=>v+' %');
wire('r-wind','v-wind', v=>v.toFixed(1)+'×');
wire('r-prog','v-prog', v=>v+' %');

// Progress slider → re-simulate
document.getElementById('r-prog').addEventListener('change', ()=>{if(lastData) runSim();});

// ── Tabs ──────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-'+t.dataset.tab).classList.add('active');
  });
});

// ── Simulate ──────────────────────────────────────────────────────
document.getElementById('btn-run').addEventListener('click', runSim);

async function runSim(){
  const btn=document.getElementById('btn-run');
  btn.disabled=true;
  document.getElementById('loading').classList.add('show');
  const p={
    altitude_m:     +document.getElementById('r-alt').value,
    overlap_frac:   +document.getElementById('r-ovlp').value/100,
    speed_ms:       +document.getElementById('r-spd').value,
    interval_s:     +document.getElementById('r-int').value,
    battery_cap:    +document.getElementById('r-batt').value,
    wind_factor:    +document.getElementById('r-wind').value,
    slider_progress:+document.getElementById('r-prog').value/100,
  };
  try{
    const r=await fetch('/api/simulate',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
    const d=await r.json();
    lastData=d;
    render(d);
  }catch(e){alert('Error: '+e.message);}
  finally{btn.disabled=false;document.getElementById('loading').classList.remove('show');}
}

// ── Render ─────────────────────────────────────────────────────────
function render(d){
  const {figures:f,mission_stats:ms,battery:b,efficiency:e,vari_bias:v,
         true_zonal:tz,sampled_zonal:sz}=d;

  Plotly.react('plt-mission', f.mission.data,  f.mission.layout,  CFG);
  Plotly.react('plt-true',    f.true_map.data,  f.true_map.layout,  CFG);
  Plotly.react('plt-recon',   f.recon_map.data, f.recon_map.layout, CFG);
  Plotly.react('plt-terrain', f.terrain.data,   f.terrain.layout,   CFG);

  // Sidebar stats
  set('s-cov',  ms.coverage_pct.toFixed(1)+'%', cvClass(ms.coverage_pct));
  set('s-cap',  ms.n_captures_done+'/'+ms.n_captures_total);
  set('s-gsd',  ms.gsd_cm.toFixed(1)+' cm');
  set('s-ovlp', ms.est_overlap_pct.toFixed(0)+'%', ovClass(ms.est_overlap_pct));
  set('s-wpts', ms.n_waypoints);
  set('s-turns',ms.n_turns);
  set('s-time', ms.elapsed_time_min.toFixed(1)+' min');
  set('s-path', ms.path_length_m.toFixed(0)+' m');

  // Battery bar
  const used=b.battery_used_now_pct, avail=b.available_pct;
  const pct=Math.min(used/avail*100,100);
  const fill=document.getElementById('batt-fill');
  fill.style.width=pct+'%';
  fill.className='batt-fill'+(pct>90?' danger':pct>70?' warn':'');
  document.getElementById('batt-pct-lbl').textContent=used.toFixed(1)+' %';

  // VARI bias
  document.getElementById('b-true').textContent=v.true_mean!=null?v.true_mean.toFixed(4):'—';
  document.getElementById('b-samp').textContent=v.sampled_mean!=null?v.sampled_mean.toFixed(4):'—';
  setDelta('b-abs', v.absolute_bias, x=>x.toFixed(4));
  setDelta('b-rel', v.relative_bias, x=>x.toFixed(2)+'%');

  // Efficiency
  const eff=e.efficiency_score;
  const ec=document.getElementById('eff-circle');
  ec.textContent=eff.toFixed(2);
  const ec_col=eff>1.5?'var(--success)':eff>0.8?'var(--warn)':'var(--danger)';
  ec.style.borderColor=ec_col; ec.style.color=ec_col;
  document.getElementById('eff-detail').innerHTML=
    'Coverage: <strong>'+e.coverage_pct.toFixed(1)+'%</strong> &nbsp;/&nbsp; '+
    'Battery: <strong>'+e.battery_used_pct.toFixed(1)+'%</strong>';

  // Badges & alerts
  const mb=document.getElementById('m-badge');
  if(!b.mission_feasible){
    mb.textContent='⚡ BATTERY CUTOFF'; mb.className='badge badge-d';
    document.getElementById('alert-batt').classList.add('batt-alert');
  } else {
    mb.textContent='✅ FEASIBLE'; mb.className='badge badge-ok';
    document.getElementById('alert-batt').className='alert';
  }
  const rb=document.getElementById('r-badge');
  if(ms.est_overlap_pct<70){
    rb.textContent='⚠ LOW OVERLAP'; rb.className='badge badge-w';
    document.getElementById('alert-ovlp').classList.add('ovlp-alert');
  } else {
    rb.textContent=ms.coverage_pct.toFixed(1)+'% covered'; rb.className='badge badge-ok';
    document.getElementById('alert-ovlp').className='alert';
  }

  // Zone tables
  fillZone('t-true', tz);
  fillZone('t-samp', sz);

  // Battery breakdown chart
  const cats=['Base Flight','Turn Penalty','Wind Resistance'];
  const vals=[b.base_cost_pct, b.turn_cost_pct, b.wind_cost_pct];
  const colors=['#00e5ff','#7c3aed','#f97316'];
  Plotly.react('plt-batt',[{
    type:'bar', x:cats, y:vals, marker:{color:colors},
    text:vals.map(v=>v.toFixed(1)+'%'), textposition:'outside',
    textfont:{color:'#c9d1d9',size:12},
    hovertemplate:'%{x}: %{y:.1f}%<extra></extra>',
  }],{
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    font:{color:'#555',size:11},
    xaxis:{showgrid:false,zeroline:false,color:'#555'},
    yaxis:{showgrid:true,gridcolor:'#21262d',zeroline:false,color:'#555',
           title:'Battery %', range:[0,Math.max(...vals)*1.35]},
    margin:{l:50,r:20,t:20,b:60}, height:200,
    shapes:[{type:'line',x0:-0.5,x1:2.5,
      y0:b.available_pct,y1:b.available_pct,
      line:{color:'#ef4444',width:2,dash:'dash'}}],
    annotations:[{x:2.3,y:b.available_pct,text:'Capacity',
      showarrow:false,font:{color:'#ef4444',size:10},yshift:8}],
  }, CFG);
}

function set(id,txt,cls){
  const el=document.getElementById(id);
  el.textContent=txt;
  if(cls) el.className='val '+cls;
}
function setDelta(id,val,fmt){
  const el=document.getElementById(id);
  if(val==null){el.textContent='—'; el.className='bias-d'; return;}
  el.textContent=(val>=0?'+':'')+fmt(val);
  el.className='bias-d '+(val>=0?'pos':'neg');
}
function cvClass(p){return p>85?'success':p>60?'warn':'danger';}
function ovClass(p){return p>70?'success':p>50?'warn':'danger';}

function fillZone(tableId, rows){
  const tb=document.querySelector('#'+tableId+' tbody');
  tb.innerHTML=rows.map(r=>{
    const m=r.mean!=null?r.mean.toFixed(4):'N/A';
    const s=r.std !=null?r.std.toFixed(3):'N/A';
    const mn=r.min!=null?r.min.toFixed(3):'N/A';
    const mx=r.max!=null?r.max.toFixed(3):'N/A';
    const n=r.n_pixels!=null?r.n_pixels:'0';
    return`<tr><td>${r.zone}</td><td>${m}</td><td>${s}</td><td>${mn}</td><td>${mx}</td><td>${n}</td></tr>`;
  }).join('');
}

window.addEventListener('load', runSim);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("=" * 60)
    print("  Flight Mission & Sampling Simulator")
    print(f"  Field : {FIELD_ACRES:.1f} acres  |  {FIELD_SIDE_M:.1f} m × {FIELD_SIDE_M:.1f} m")
    print(f"  Grid  : {RASTER_ROWS} × {RASTER_COLS} px  |  {CELL_SIZE_M:.2f} m/px")
    print(f"  Camera HFOV: {HFOV_DEG:.1f}°")
    print("=" * 60)
    print("  Open: http://localhost:5050")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5050)