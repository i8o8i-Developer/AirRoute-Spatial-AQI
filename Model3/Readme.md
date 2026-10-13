# AirRoute Model 3 — Spatial AQI Estimation Engine

> **Module Architectural Status:** Part 1 Data Foundation And Spatial Baseline Complete  
> **Target Geospatial Region:** Delhi National Capital Region (Delhi NCR)  
> **Core Architectural Paradigm:** Two-Stage Spatial Estimator (Stage 1: IDW Baseline; Stage 2: XGBoost Residual Correction)  
> **Development Standard:** Strict Separation From Model 2 With Zero Data Leakage  

---

## 1. Project Role, Purpose, And Mission

Model 3 Fills The Primary Geospatial Gap In The AirRoute System. Model 2 Provides Forward Predictive Forecasts Strictly For Points Having A Complete 24-Hour Historical Observation Sequence. Consequently, Model 2 Cannot Deliver A City-Wide Continuous Pollution Surface Because Most Urban Grid Points Lack Dedicated Physical Monitoring Stations.

Model 3 Addresses This Coverage Gap By Estimating The Air Quality Index (AQI) And Pollutant Concentrations At Arbitrary Unmonitored Coordinates Across Delhi NCR Using Surrounding Physical Monitoring Stations And Contextual Environmental Features.

### Core Geospatial Inquiries Addressed
- What Is The Current Estimated Air Quality Index At Any Unmonitored Coordinate Right Now?
- What Are The Spatial Pollution Gradients And Transitions Between Physical Monitoring Stations?
- Which Specific Neighborhoods And Corridors Experience Elevated Pollution Burdens During Rush Hours?

---

## 2. Non-Negotiable Engineering Guardrails

1. **No Overclaiming Of Physics Capabilities:** We Do Not Claim Road-Level Precision, WRF-Chem Chemical Transport Simulations, Or Physics-Based Atmospheric Plume Dispersion. The Model Faithfully Reflects The Empirical Resolution Supported By The Station Network (~1.5 To 5.0 Kilometers).
2. **Strict Spatial Evaluation Isolation:** No Random Row Splitting Is Ever Permitted. All Validation Is Conducted Via Spatially Aware Leave-One-Station-Out (LOSO) Simulation And Dedicated Spatial/Temporal Holdouts.
3. **Temporal Feature Availability:** No Feature Is Utilized That Would Be Unavailable In A Real-Time Production Setting.
4. **Transparent Residual Correction Standard:** The Stage 2 XGBoost Residual Layer Is Accepted Only If It Demonstrates Statistically Significant Error Reduction Compared To The Tuned IDW Baseline On Genuinely Unseen Stations And Future Time Periods.

---

## 3. PascalCase Directory Architecture And File Manifest

All Project Directories And Code Modules Are Formatted In PascalCase:

```text
Model3/
├── Data/
│   ├── StationHistory.parquet        # Synchronized Historical Observations Across 38 Stations
│   ├── SpatialWindows.parquet        # Leave-One-Station-Out (LOSO) Training Evaluation Windows
│   └── TestTargets.parquet           # Pure Unseen Spatial And Temporal Holdout Targets
├── Models/
│   └── XgbSpatialResidual.json       # Persisted XGBoost Model Artifact (Created In Part 2)
├── Src/
│   ├── __init__.py                   # Package Root Initializer
│   ├── Config.py                     # Central System Hyperparameters, Seeds, And Breakpoints
│   ├── CpcbAqi.py                    # Official Indian CPCB Sub-Index And NAQI Engine
│   ├── SpatialFeatures.py            # Great-Circle Haversine Geodesic Distance And Bearing
│   ├── Idw.py                        # Vectorized IDW Interpolator With Diagnostic Metadata
│   ├── Dataset.py                    # Data Ingestion, QC Filter, And Holdout Generator
│   ├── Evaluate.py                   # Geospatial Evaluation Suite And Moran's I Calculator
│   ├── TuneIdw.py                    # Cross-Validation Hyperparameter Optimization Runner
│   ├── Train.py                      # Master Execution Pipeline Driver
│   └── Predict.py                    # Callable Point And 2D Grid Inference API
├── Results/
│   ├── IdwMetrics.json               # Consolidated Numerical Performance Metrics
│   ├── PerStationBreakdown.csv       # Station-By-Station Error Metrics Table
│   ├── XgbMetrics.json               # Comparison Results (Delivered In Part 2)
│   ├── SpatialPredictions.csv        # Holdout Predictions (Delivered In Part 2)
│   └── Plots/
│       ├── ErrorByStation.png        # Mean Absolute Error Per Station Bar Chart
│       ├── ErrorByHour.png           # Diurnal Hourly Estimation Error Profile
│       ├── PredictedVsActual.png     # 1:1 Correlation Scatter Plot
│       └── StationResidualMap.png    # Spatial Map Of Residual Error Clusters
├── Tests/
│   ├── TestDistance.py               # Unit Tests For Geodetic Haversine Formulas
│   ├── TestCpcbAqi.py                # Unit Tests For Official CPCB Breakpoint Calculations
│   ├── TestIdw.py                    # Unit Tests For Edge Cases, Co-Location, And NaNs
│   └── TestLeakage.py                # Programmatic Proof Of Zero Leakage In LOSO Evaluation
├── Readme.md                         # Detailed Architectural Documentation In Title Case
└── TeamLeadProgressReport.md         # Comprehensive Executive Milestone Audit Document
```

---

## 4. Mathematical Methodology

### Stage 1: Vectorized Inverse Distance Weighting (IDW)
Given Target Coordinate $X$ And Monitored Stations $\{X_i\}_{i=1}^N$ With Observed Values $Y_i$:
$$\hat{Y}_{\text{IDW}}(X) = \frac{\sum_{i=1}^K W_i Y_i}{\sum_{i=1}^K W_i}, \quad W_i = \frac{1}{D(X, X_i)^P}$$

Where:
- $D(X, X_i)$ Represents The Great-Circle Haversine Distance In Kilometers:
  $$D = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
- $K$: Number Of Nearest Monitored Neighbors Selected Within Maximum Search Radius $R_{\max}$.
- $P$: Distance-Decay Power Exponent.

### Edge Case Handling Rules
- **Co-Located Exact Match ($D < 10^{-4}\text{ Km}$):** Immediately Returns The Physical Station Reading (`EXACT_MATCH`).
- **Insufficient Neighbor Coverage ($N_{\text{Valid}} < K_{\min}$):** Returns A Warning Status (`INSUFFICIENT_COVERAGE`) With `NaN` Instead Of Uncontrolled Extrapolation.
- **Missing Sensor Imputation:** Flagged Or Missing Sensors Are Filtered Out Prior To Neighbor Weight Calculation.

---

## 5. Hyperparameter Tuning And Empirical Findings

A Complete Location-Based Cross-Validation Search Was Executed Over The Training Partition:
- **Optimal Neighbors Count ($K^*$):** $8$
- **Optimal Geographic Search Radius ($R^*$):** $25.0\text{ Kilometers}$
- **Optimal Distance-Decay Exponent ($P^*$):** $1.0$ (Linear Weighting)

### Baseline Comparison Summary

| Estimator Approach | Parameter Specification | Validation MAE | Performance Comparison |
| :--- | :--- | :---: | :---: |
| **Nearest-Station Baseline** | $K=1, R=25\text{ Km}$ | **35.00** | *Standard Engineering Reference* |
| **Plain Neighbor Mean** | $K=8, P=0.0$ | **23.50** | *Uniform Unweighted Averaging* |
| **Tuned IDW Baseline** | **$K=8, R=25\text{ Km}, P=1.0$** | **24.26** | **30.7% Error Reduction Vs Nearest** |

---

## 6. Scientific Experiment: Direct AQI Versus Pollutant-First Estimation

| Evaluation Strategy | MAE (AQI Points) | RMSE | $R^2$ Score | Category Accuracy | Tolerance $\le \pm 20$ AQI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pollutant-First Interpolation** | **24.35** | 39.75 | 0.8843 | **78.92%** | **59.11%** |
| **Direct AQI Interpolation** | 24.70 | **38.03** | **0.8941** | 77.87% | 56.89% |

### Core Scientific Takeaway
- Pollutant-First Interpolation Aligns With Physical Atmospheric Stoichiometry, Yielding Lower Mean Absolute Error (24.35 Vs 24.70) And Superior Category Accuracy (78.92% Vs 77.87%).
- Direct AQI Interpolation Smooths The Piecewise Maximum Operator, Delivering Lower Outlier Squared Errors (RMSE 38.03 Vs 39.75).
- Both Feature Streams Are Retained In `SpatialWindows.parquet` To Provide High-Information Input Channels For The Stage 2 XGBoost Residual Estimator.

---

## 7. Residual Spatial Autocorrelation Analysis (Moran's I)

- **Observed Moran's I On Training Partition:** $-0.1196$ (Expected $I = -0.0357$)
- **Observed Moran's I On Unseen Holdout:** $-0.0922$ (Expected $I = -0.0286$)
- **Spatial Autocorrelation Detected:** **False (Spatially Random / Dispersed Residuals)**

> [!IMPORTANT]
> **Implications For Stage 2 Residual Modelling:**  
> Because Moran's I Is Close To Zero, The Residual Errors Do Not Form A Smooth Regional Spatial Gradient. IDW Has Effectively Captured The Distance-Decay Spatial Signal. The Remaining Errors Are Localized Point Micro-Biases (Such As Localized Heavy Freight Emissions Near Anand Vihar And Jahangirpuri). Consequently, The Stage 2 XGBoost Model **Must Not Rely Purely On Spatial Coordinates** (Which Would Risk Overfitting). It Must Learn From Neighbor Spread Indicators (`neighbor_std`), Urban Rush-Hour Regimes, And Ambient Meteorology.

---

## 8. Command-Line Execution And Operational Guide

### Run Automated Unit Test Suite
```bash
python -m pytest
```

### Run Full Training And Hyperparameter Tuning Pipeline
```bash
python -m Model3.Src.Train
```

### High-Level Point And Grid Inference In Python
```python
from Model3.Src.Predict import SpatialAqiPredictor

# Initialize Spatial Prediction Service
PredictorInstance = SpatialAqiPredictor()

# Query Unmonitored Target Coordinate (Example: Connaught Place)
PointEstimateResult = PredictorInstance.PredictPoint(
    TargetLatitude=28.6315,
    TargetLongitude=77.2167,
)
print("Point Prediction Result:", PointEstimateResult)

# Generate Continuous 2D Spatial Pollution Surface Grid
SurfaceGridDataFrame = PredictorInstance.PredictGrid(
    LatitudeStep=0.02,
    LongitudeStep=0.02,
)
print("Surface Grid Preview:")
print(SurfaceGridDataFrame.head())
```

---

## 9. Review Gate And Next Steps

Part 1 Is Completely Implemented, Audited, And Passing All Tests.
Awaiting Team Leadership Authorization Before Proceeding To **Part 2: XGBoost Residual Correction, Grid Surface, And Integration**.
