"""
Central Configuration Module For AirRoute Model 3 Spatial AQI Estimation.
Contains All Tunable Hyperparameters, Geographic Boundaries, Station Metadata,
Official CPCB AQI Breakpoints, File System Paths, And Reproducibility Random Seeds.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Base Directory Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data"
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_DIR = PROJECT_ROOT / "Results"
PLOTS_DIR = RESULTS_DIR / "Plots"

# Ensure Required Subdirectories Exist
for Directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR):
    Directory.mkdir(parents=True, exist_ok=True)

# Parquet Data Files And Performance Result Locations
STATION_HISTORY_PARQUET = DATA_DIR / "StationHistory.parquet"
SPATIAL_WINDOWS_PARQUET = DATA_DIR / "SpatialWindows.parquet"
TEST_TARGETS_PARQUET = DATA_DIR / "TestTargets.parquet"
IDW_METRICS_JSON = RESULTS_DIR / "IdwMetrics.json"
XGB_METRICS_JSON = RESULTS_DIR / "XgbMetrics.json"
SPATIAL_PREDICTIONS_CSV = RESULTS_DIR / "SpatialPredictions.csv"
PER_STATION_BREAKDOWN_CSV = RESULTS_DIR / "PerStationBreakdown.csv"
XGB_MODEL_FILE = MODELS_DIR / "XgbSpatialResidual.json"

# Fixed Global Random Seed For Deterministic Reproducibility
RANDOM_SEED: int = 42

# Delhi NCR Regional Bounding Box Coordinates
REGION_BOUNDS: Dict[str, float] = {
    "lat_min": 28.40,
    "lat_max": 28.90,
    "lon_min": 76.90,
    "lon_max": 77.45,
}

# 2D Spatial Grid Generation Configuration Parameters
GRID_CONFIG: Dict[str, float] = {
    "lat_step": 0.015,  # Approximately 1.6 Kilometers Resolution
    "lon_step": 0.015,
}

# IDW Hyperparameter Search Ranges And Defaults
IDW_K_RANGE: List[int] = [3, 4, 5, 6, 7, 8]
IDW_RADIUS_KM_RANGE: List[float] = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
IDW_POWER_RANGE: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0]

DEFAULT_K: int = 8
DEFAULT_RADIUS_KM: float = 25.0
DEFAULT_POWER: float = 1.0
MIN_VALID_NEIGHBORS: int = 2

# Evaluation Holdout Ratios
SPATIAL_HOLDOUT_RATIO: float = 0.20  # Twenty Percent Stations Reserved Entirely
TEMPORAL_HOLDOUT_RATIO: float = 0.20  # Most Recent Twenty Percent Timestamps Reserved

# Urban Traffic Rush Hour Intervals In Indian Standard Time
MORNING_RUSH_HOURS: Tuple[int, ...] = (8, 9, 10, 11)
EVENING_RUSH_HOURS: Tuple[int, ...] = (17, 18, 19, 20, 21)

# Atmospheric Episode Threshold In Indian AQI Points
EPISODE_THRESHOLD_SEVERE: float = 300.0  # Threshold For Very Poor Or Severe Category

# Physical Quality Control Upper And Lower Acceptance Thresholds
QC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "pm2_5": (0.0, 1000.0),      # Micrograms Per Cubic Meter
    "pm10": (0.0, 2000.0),       # Micrograms Per Cubic Meter
    "no2": (0.0, 1000.0),        # Micrograms Per Cubic Meter
    "so2": (0.0, 1000.0),        # Micrograms Per Cubic Meter
    "co": (0.0, 50.0),           # Milligrams Per Cubic Meter
    "o3": (0.0, 1000.0),         # Micrograms Per Cubic Meter
    "aqi": (0.0, 500.0),         # Standard Indian AQI Numerical Range
    "temperature": (-5.0, 55.0), # Degrees Celsius
    "humidity": (0.0, 100.0),    # Relative Humidity Percentage
    "wind_speed": (0.0, 40.0),   # Meters Per Second
    "wind_direction": (0.0, 360.0), # Compass Degrees
    "pressure": (900.0, 1100.0), # Hectopascals
}

# Maximum Consecutive Invariant Readings Before Sensor Is Flagged As Stuck
STUCK_SENSOR_MAX_HOURS: int = 6

# Official Indian Central Pollution Control Board NAQI Breakpoints
CPCB_BREAKPOINTS = {
    "pm2_5": [
        (0.0, 30.0, 0, 50),
        (30.1, 60.0, 51, 100),
        (60.1, 90.0, 101, 200),
        (90.1, 120.0, 201, 300),
        (120.1, 250.0, 301, 400),
        (250.1, 500.0, 401, 500),
    ],
    "pm10": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 250.0, 101, 200),
        (250.1, 350.0, 201, 300),
        (350.1, 430.0, 301, 400),
        (430.1, 600.0, 401, 500),
    ],
    "no2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 180.0, 101, 200),
        (180.1, 280.0, 201, 300),
        (280.1, 400.0, 301, 400),
        (400.1, 600.0, 401, 500),
    ],
    "so2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 380.0, 101, 200),
        (380.1, 800.0, 201, 300),
        (800.1, 1600.0, 301, 400),
        (1600.1, 2400.0, 401, 500),
    ],
    "co": [
        (0.0, 1.0, 0, 50),
        (1.01, 2.0, 51, 100),
        (2.01, 10.0, 101, 200),
        (10.01, 17.0, 201, 300),
        (17.01, 34.0, 301, 400),
        (34.01, 50.0, 401, 500),
    ],
    "o3": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 168.0, 101, 200),
        (168.1, 208.0, 201, 300),
        (208.1, 748.0, 301, 400),
        (748.1, 1000.0, 401, 500),
    ],
}

# Categorical Health Impact Band Descriptors
AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]

# Official Delhi NCR CAAQMS Stations Coordinates
DELHI_STATIONS: List[Dict[str, object]] = [
    {"station_id": "DEL_001", "name": "Anand Vihar, Delhi - DPCC", "lat": 28.6476, "lon": 77.3158},
    {"station_id": "DEL_002", "name": "R K Puram, Delhi - DPCC", "lat": 28.5632, "lon": 77.1869},
    {"station_id": "DEL_003", "name": "Punjabi Bagh, Delhi - DPCC", "lat": 28.6740, "lon": 77.1310},
    {"station_id": "DEL_004", "name": "Mandir Marg, Delhi - DPCC", "lat": 28.6364, "lon": 77.1996},
    {"station_id": "DEL_005", "name": "IGI Airport (T3), Delhi - IMD", "lat": 28.5562, "lon": 77.0999},
    {"station_id": "DEL_006", "name": "Bawana, Delhi - DPCC", "lat": 28.7762, "lon": 77.0511},
    {"station_id": "DEL_007", "name": "Jahangirpuri, Delhi - DPCC", "lat": 28.7328, "lon": 77.1706},
    {"station_id": "DEL_008", "name": "Wazirpur, Delhi - DPCC", "lat": 28.6998, "lon": 77.1654},
    {"station_id": "DEL_009", "name": "Okhla Phase-2, Delhi - DPCC", "lat": 28.5308, "lon": 77.2712},
    {"station_id": "DEL_010", "name": "Shadipur, Delhi - CPCB", "lat": 28.6515, "lon": 77.1581},
    {"station_id": "DEL_011", "name": "Dwarka-Sector 8, Delhi - DPCC", "lat": 28.5710, "lon": 77.0719},
    {"station_id": "DEL_012", "name": "Lodhi Road, Delhi - IITM", "lat": 28.5918, "lon": 77.2273},
    {"station_id": "DEL_013", "name": "Siri Fort, Delhi - CPCB", "lat": 28.5504, "lon": 77.2159},
    {"station_id": "DEL_014", "name": "Alipur, Delhi - DPCC", "lat": 28.8153, "lon": 77.1530},
    {"station_id": "DEL_015", "name": "DTU, Delhi - CPCB", "lat": 28.7501, "lon": 77.1113},
    {"station_id": "DEL_016", "name": "Rohini, Delhi - DPCC", "lat": 28.7325, "lon": 77.1199},
    {"station_id": "DEL_017", "name": "Vivek Vihar, Delhi - DPCC", "lat": 28.6723, "lon": 77.3153},
    {"station_id": "DEL_018", "name": "Patparganj, Delhi - DPCC", "lat": 28.6238, "lon": 77.2872},
    {"station_id": "DEL_019", "name": "Nehru Nagar, Delhi - DPCC", "lat": 28.5679, "lon": 77.2505},
    {"station_id": "DEL_020", "name": "Ashok Vihar, Delhi - DPCC", "lat": 28.6954, "lon": 77.1817},
    {"station_id": "DEL_021", "name": "Sonia Vihar, Delhi - DPCC", "lat": 28.7105, "lon": 77.2494},
    {"station_id": "DEL_022", "name": "Major Dhyan Chand Stadium, Delhi - DPCC", "lat": 28.6138, "lon": 77.2389},
    {"station_id": "DEL_023", "name": "Jawaharlal Nehru Stadium, Delhi - DPCC", "lat": 28.5802, "lon": 77.2338},
    {"station_id": "DEL_024", "name": "Najafgarh, Delhi - DPCC", "lat": 28.6090, "lon": 76.9798},
    {"station_id": "DEL_025", "name": "Narela, Delhi - DPCC", "lat": 28.8526, "lon": 77.0924},
    {"station_id": "DEL_026", "name": "Mundka, Delhi - DPCC", "lat": 28.6847, "lon": 77.0299},
    {"station_id": "DEL_027", "name": "Pusa, Delhi - DPCC", "lat": 28.6396, "lon": 77.1463},
    {"station_id": "DEL_028", "name": "CRRI Mathura Road, Delhi - IMD", "lat": 28.5512, "lon": 77.2736},
    {"station_id": "DEL_029", "name": "Aya Nagar, Delhi - IMD", "lat": 28.4707, "lon": 77.1099},
    {"station_id": "DEL_030", "name": "Burari Crossing, Delhi - IMD", "lat": 28.7257, "lon": 77.2012},
    {"station_id": "DEL_031", "name": "Dr. Karni Singh Shooting Range, Delhi - DPCC", "lat": 28.4986, "lon": 77.2648},
    {"station_id": "DEL_032", "name": "Sri Aurobindo Marg, Delhi - DPCC", "lat": 28.5313, "lon": 77.1901},
    {"station_id": "DEL_033", "name": "North Campus, DU, Delhi - IMD", "lat": 28.6940, "lon": 77.2104},
    {"station_id": "DEL_034", "name": "ITO, Delhi - CPCB", "lat": 28.6317, "lon": 77.2494},
    {"station_id": "DEL_035", "name": "Sirifort, Delhi - CPCB", "lat": 28.5504, "lon": 77.2159},
    {"station_id": "DEL_036", "name": "Sector 62 Noida, UP - UPPCB", "lat": 28.6245, "lon": 77.3648},
    {"station_id": "DEL_037", "name": "Vikas Sadan Gurugram, Haryana - HSPCB", "lat": 28.4526, "lon": 77.0302},
    {"station_id": "DEL_038", "name": "Sector 51 Gurugram, Haryana - HSPCB", "lat": 28.4230, "lon": 77.0817},
]
