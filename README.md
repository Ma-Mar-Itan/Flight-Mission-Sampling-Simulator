# Flight Mission & Sampling Simulator

**Module 2 of the Integrated Smart Farm Project**

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://flight-mission-sampling-simulator-h3witblvxcmugvdqfmzlff.streamlit.app/)
&nbsp;![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
&nbsp;![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit&logoColor=white)

---

## Project Context

This simulator is one component of a broader research initiative to design and deploy a **fully autonomous, data-driven smart farm**. The overarching vision is a vertically integrated agricultural system in which every decision — from irrigation scheduling to harvest timing — is informed by continuous, high-resolution field intelligence gathered and processed without human intervention.

The smart farm system is being developed across several interconnected modules:

| Module | Description | Status |
|---|---|---|
| **1 — Field Sensing** | Soil sensors, weather stations, IoT data pipeline | In progress |
| **2 — Aerial Survey Simulation** | Drone mission planning & VARI sampling ← *this repo* | ✅ Demo live |
| **3 — Photogrammetric Reconstruction** | Orthomosaic generation, canopy height modelling | Planned |
| **4 — Crop Health Analytics** | Vegetation index time-series, anomaly detection | Planned |
| **5 — Autonomous Decision Engine** | ML-driven irrigation, fertilisation, and intervention triggers | Planned |

---

## Module 2 — Aerial Survey Simulation

Before deploying physical drone infrastructure, this module addresses a fundamental research question:

> *What mission parameters — altitude, speed, overlap, battery — are necessary to achieve statistically representative aerial coverage of a given agricultural field?*

Committing to drone hardware without answering this question first leads to systematic errors: incomplete field coverage, photogrammetric reconstruction failure, and biased crop-health estimates that propagate through every downstream analysis module. This simulator makes those failure modes visible and quantifiable prior to any capital expenditure.

### Live Demo

**[https://flight-mission-sampling-simulator-h3witblvxcmugvdqfmzlff.streamlit.app/](https://flight-mission-sampling-simulator-h3witblvxcmugvdqfmzlff.streamlit.app/)**

---

## Scientific Motivation

Vegetation indices such as VARI (Visible Atmospherically Resistant Index) are only meaningful if the sampling strategy that produces them is spatially unbiased. Incomplete drone coverage introduces a **non-random spatial sampling error**: if the uncaptured regions of a field systematically differ in crop health from captured regions — as they do near field margins, in low-overlap strips, or after a battery cutoff — the resulting index estimate is biased regardless of sensor quality.

This module models and quantifies that bias explicitly, producing two measurements that are rarely reported in agricultural remote sensing literature:

- **Absolute Bias** — the difference in mean VARI between the true field and the sampled reconstruction
- **Relative Bias** — the same quantity expressed as a percentage of the true mean

These metrics are used to validate mission parameter choices before physical deployment.

---

## Technical Overview

### Field Model
A synthetic 4-acre (≈127 × 127 m) agricultural raster is generated at 128 × 128 pixel resolution (~1 m/px) using Gaussian-smoothed stochastic noise. A spatially coherent stressed-crop patch (modelling disease or waterlogging) is embedded to create meaningful intra-field heterogeneity.

### Camera Model
The simulator adopts a **Sony RX1R II equivalent** (35.9 mm sensor, 35 mm focal length, 8000 px image width, HFOV ≈ 54.4°) as a representative professional aerial survey camera.

```
GSD (cm/px)           = (sensor_mm × altitude_m × 100) / (focal_mm × image_px)
Footprint radius (px) = altitude_m × tan(HFOV/2) / cell_size_m
```

### Mission Geometry
A **boustrophedon (lawnmower) flight path** is generated from altitude and overlap parameters. Strip spacing is derived directly from footprint geometry:

```
line_spacing = footprint_diameter × (1 − sidelap_fraction)
```

### Battery Model
Power consumption is modelled as a linear function of flight time, lane-reversal count, and wind resistance:

```
total_cost = (time × 0.055) + (turns × 0.8) + (time × 0.03 × wind_factor)   [% of capacity]
```

If total cost exceeds available capacity, the mission is truncated and the achievable path fraction is back-solved analytically.

### Reconstruction Quality
Photogrammetric reconstruction failure at low overlap is modelled via stochastic column-banded gap injection, calibrated to the deficit below the 80% ideal overlap threshold. This mimics the strip-aligned void patterns observed in real Structure-from-Motion failures.

---

## Simulation Parameters

| Parameter | Range | Physical Interpretation |
|---|---|---|
| Altitude | 10 – 120 m | Determines GSD and footprint radius |
| Flight speed | 1 – 20 m/s | Governs mission duration and capture density |
| Capture interval | 0.5 – 10 s | Controls along-track image spacing |
| Sidelap overlap | 20 – 95% | Primary driver of reconstruction quality |
| Battery capacity | 10 – 100% | Models degraded or partial-charge scenarios |
| Wind resistance | 0 – 5× | Scales additional power draw from wind load |
| Mission progress | 0 – 100% | Temporal scrubber for trajectory replay |

---

## Output Metrics

| Metric | Definition |
|---|---|
| Coverage % | Fraction of field pixels with ≥ 1 capture footprint |
| GSD | Ground Sampling Distance in cm/px |
| Efficiency Score | Coverage% / Battery% — spatial yield per unit energy |
| Absolute Bias | Sampled VARI mean − True VARI mean |
| Relative Bias | Absolute bias / True mean × 100% |
| Zonal Statistics | Per-quadrant VARI mean, std, min, max (true vs sampled) |

---

## Stack

```
Python 3.9+
├── streamlit      — interactive web UI
├── plotly         — live mission map, 3D terrain, analytics charts
├── numpy          — raster generation, mask operations, statistics
├── scipy          — Gaussian smoothing for synthetic field generation
└── pandas         — zonal statistics tables
```

---

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/flight-mission-sampling-simulator.git
cd flight-mission-sampling-simulator
pip install -r requirements.txt
streamlit run app.py
```

---

## Relation to the Broader Research Programme

The outputs of this module feed directly into **Module 3 (Photogrammetric Reconstruction)** by establishing the minimum viable mission parameters for gap-free orthomosaic generation. The bias quantification methodology developed here — comparing true-field statistics against sampled-field statistics — will be extended in **Module 4** to evaluate the statistical validity of time-series vegetation index estimates across the full growing season.

The long-term goal is a closed-loop system in which drone survey parameters are dynamically adjusted based on crop-health anomalies detected in the previous flight, removing the need for manual mission planning entirely.

---

## Author

Developed as part of ongoing undergraduate / postgraduate research into autonomous precision agriculture systems.

---

*Supervisor enquiries and collaboration proposals welcome.*
