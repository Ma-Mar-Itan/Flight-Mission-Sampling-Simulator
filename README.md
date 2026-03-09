# 🛰 Flight Mission & Sampling Simulator

> **Plan your agricultural drone survey before buying a single piece of equipment.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://flight-mission-sampling-simulator-h3witblvxcmugvdqfmzlff.streamlit.app/)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔗 Live App

**[https://flight-mission-sampling-simulator-h3witblvxcmugvdqfmzlff.streamlit.app/](https://flight-mission-sampling-simulator-h3witblvxcmugvdqfmzlff.streamlit.app/)**

No installation required — runs entirely in your browser.

---

## 🌾 What Is This?

The **Flight Mission & Sampling Simulator** is an interactive tool designed for farmers and agronomists who want to **plan a drone survey mission over their agricultural land before investing in any equipment**.

Instead of buying a drone, camera, and software and discovering the hard way that your altitude was too high or your overlap too low, this simulator lets you:

- **Experiment freely** with every mission parameter in real time
- **See exactly** how your choices affect field coverage, image quality, and battery life
- **Understand the trade-offs** between altitude, speed, overlap, and battery before you commit to hardware
- **Detect statistical bias** in crop-health indices that arises from incomplete coverage

The simulator models a **4-acre agricultural field** (≈127 × 127 m) and a **Sony RX1R II camera equivalent** (35 mm fixed lens, 8000 px wide) — a professional drone camera standard — giving physically realistic results you can trust.

---

## ✨ Features

### 🛰 Live Mission View
Watch the drone fly a **boustrophedon (lawnmower) pattern** over your field in real time. See every planned waypoint, each photo capture event, the current drone position, and — if your battery runs out — exactly where the mission gets cut short.

### 🗺 True vs Reconstructed Map
Side-by-side comparison of the **true VARI field** (what actually exists) and the **reconstructed map** (what your drone would actually capture). Grey areas are gaps — uncovered pixels your photogrammetry software cannot reconstruct. Low overlap or battery cutoffs make these gaps larger and more frequent.

### 🏔 3D Terrain View
An interactive **3D elevation model** of the field coloured by vegetation index. Rotate, zoom, and tilt to understand how terrain relief might affect flight planning and shadow patterns.

### 📊 Analytics Dashboard
- **Zonal statistics** — per-quadrant VARI mean, std, min, max for both the true field and your sampled coverage
- **Sampling bias quantification** — how much does your incomplete coverage skew the measured crop-health mean?
- **Battery cost breakdown** — base flight cost vs turn penalty vs wind resistance
- **Full mission summary** table — every parameter in one place

---

## 🎛 Mission Parameters

All parameters are adjustable via the sidebar. Press **⚡ RUN SIMULATION** to apply changes.

| Parameter | Range | What it controls |
|---|---|---|
| **Altitude (m)** | 10 – 120 m | Flying height above ground. Higher = wider footprint, coarser GSD, fewer strips needed, more battery |
| **Flight Speed (m/s)** | 1 – 20 m/s | Cruise speed. Faster = shorter mission time, fewer captures per strip |
| **Capture Interval (s)** | 0.5 – 10 s | Time between photos. Shorter = more images, better coverage, more data |
| **Sidelap Overlap (%)** | 20 – 95% | Strip-to-strip lateral overlap. Below ~70% causes reconstruction gaps |
| **Battery Capacity (%)** | 10 – 100% | Simulates flying with a partially charged or degraded battery |
| **Wind Resistance (×)** | 0 – 5× | Wind multiplier on power consumption. Higher wind drains battery faster |
| **Mission Progress (%)** | 0 – 100% | Scrub through the mission timeline without re-simulating |

---

## 📐 Physical Models

The simulator uses real aerospace and photogrammetry formulas — not toy approximations.

### Camera Model
```
Camera:       Sony RX1R II equivalent
Sensor:       35.9 mm wide
Focal length: 35.0 mm
Image width:  8,000 px
HFOV:         ≈ 54.4°

GSD (cm/px)            = (sensor_mm × altitude_m × 100) / (focal_mm × image_px)
Footprint radius (px)  = altitude × tan(HFOV/2) / cell_size_m
```

### Battery Model
```
base_cost  = mission_time_s × 0.055 %/s
turn_cost  = n_lane_reversals × 0.8 %
wind_cost  = mission_time_s × 0.03 %/s × wind_factor
──────────────────────────────────────────────────────
total_used = base_cost + turn_cost + wind_cost

If total_used > battery_capacity → mission is truncated
fraction_possible = (capacity − turn_cost) / (base + wind rate × time)
```

### Reconstruction Quality
```
Overlap ≥ 80%  →  full photogrammetric reconstruction
Overlap 70–80% →  minor stochastic gaps begin appearing
Overlap < 70%  →  significant column-banded reconstruction failure
```

### Efficiency Score
```
Efficiency = Coverage% / Battery%
> 1.5   →  Excellent (lots of field covered per unit battery)
0.8–1.5 →  Moderate
< 0.8   →  Poor
```

---

## 📊 Output Metrics Explained

| Metric | Meaning |
|---|---|
| **Coverage %** | Fraction of the 4-acre field with at least one photo footprint |
| **GSD (cm/px)** | Ground Sampling Distance — spatial resolution of each pixel in the final orthomosaic |
| **Overlap est.** | Actual sidelap fraction computed from footprint size and line spacing |
| **Captures** | Photos completed / photos scheduled for the full mission |
| **Turns** | Number of lane-reversal manoeuvres — each costs 0.8% battery |
| **Elapsed** | Flight time at the current progress position |
| **Path** | Total planned flight path length in metres |
| **Efficiency** | Coverage% ÷ Battery% — more field per unit of energy |
| **Absolute Bias** | Sampled VARI mean minus true VARI mean — how much incomplete coverage skews the crop-health estimate |
| **Relative Bias** | Absolute bias as a percentage of the true mean |

---

## 🌱 How to Use It for Your Farm

**Step 1 — Set your target altitude.**
Lower altitudes (15–30 m) give finer detail (smaller GSD) but require more strips and more battery. Higher altitudes (60–100 m) cover ground faster but with coarser resolution. Start at 30 m.

**Step 2 — Tune overlap until coverage exceeds 95%.**
Watch the *True vs Reconstructed* tab — gaps appear as grey areas. Aim for the two maps to look nearly identical. 75% overlap is a good starting point.

**Step 3 — Check the battery.**
If you see a red ⚡ BATTERY CUTOFF banner, you need a larger battery, calmer wind conditions, or a lower flight speed. The Analytics tab shows exactly which cost is eating your budget.

**Step 4 — Read the bias number.**
In the *True vs Reconstructed* tab, check the Absolute Bias metric. A value below 0.01 VARI means your coverage is statistically representative. Higher bias means parts of the field are systematically missing from your crop-health estimate.

**Step 5 — Use the Progress slider.**
Scrub through the mission to see how coverage builds over time — useful for planning partial-mission fallback strategies if your drone has to return early.

**Step 6 — Record your optimal settings.**
The altitude, speed, overlap %, and battery capacity that give you > 95% coverage, < 0.01 bias, and a feasible battery budget are your **hardware specification**. Use these numbers when comparing drone models and cameras before purchasing.

---

## 🛠 Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/flight-mission-sampling-simulator.git
cd flight-mission-sampling-simulator

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

The app opens at **http://localhost:8501**

### Requirements

```
streamlit
numpy
scipy
plotly
pandas
```

---

## 🏗 Architecture

The codebase is a single self-contained file (`app.py`, ~1,240 lines) with 8 clearly separated sections:

```
Section 1 — Synthetic Field Generator
            Gaussian-smoothed noise raster · stressed crop patch
            DEM · VARI · NDVI · RGB orthomosaic

Section 2 — Mission Geometry
            Lawnmower path · footprint radius · GSD · line spacing · turn counting

Section 3 — Drone Motion Engine
            Segment interpolation · capture event scheduling · path length

Section 4 — Battery Model
            Time cost · turn penalty · wind resistance · cutoff fraction

Section 5 — Reconstruction Engine
            Circular footprint accumulation · overlap-gap degradation

Section 6 — Statistics Engine
            Zonal stats · sampling bias · efficiency score

Section 7 — Plotly Figure Builders
            Live mission map · true/recon heatmaps · 3D terrain · battery chart

Section 8 — Streamlit UI
            Sidebar controls · session state · KPI row · tabbed layout
```

---

## 🌿 Vegetation Index (VARI)

The simulator uses **VARI** — the Visible Atmospherically Resistant Index — as its primary crop-health indicator:

```
VARI = (Green − Red) / (Green + Red − Blue)

Range: −0.3 (stressed / bare soil) → +0.6 (dense healthy canopy)
```

The synthetic field contains a **deliberately stressed patch** (simulating disease, waterlogging, or nutrient deficiency) visible as the dark red region in the True Map. This makes the bias analysis meaningful — incomplete coverage that misses this patch will produce an overly optimistic VARI estimate.

---

## 📌 Limitations & Assumptions

| Assumption | Reality |
|---|---|
| Flat 4-acre square field | Real fields have irregular boundaries, slopes, obstacles, and no-fly zones |
| Fixed 35 mm lens | Zoom lenses, fisheye, multispectral, and thermal sensors have different HFOV values |
| Linear battery model | Real LiPo curves, temperature effects, and hover vs cruise differences are not modelled |
| Scalar wind factor | Wind direction, gusting, and turbulence are not simulated |
| Heuristic reconstruction gaps | Real SfM photogrammetry depends on feature matching, lighting, forward overlap, and more |
| Single homogeneous field | Real farms have multiple crops, hedgerows, irrigation lines, and access tracks |

---

## 📄 License

MIT License — free to use, modify, and share.

---

## 🤝 Contributing

Issues and pull requests are welcome. If you test the simulator against a real drone mission and find the predictions useful — or wrong — please open an issue with your field specs, drone model, and results.

---

*Built with [Streamlit](https://streamlit.io) · [Plotly](https://plotly.com) · [NumPy](https://numpy.org) · [SciPy](https://scipy.org)*
