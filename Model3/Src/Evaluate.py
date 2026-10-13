"""
Comprehensive Spatial Model Evaluation And Moran's I Residual Autocorrelation Suite.
Computes MAE, RMSE, R^2, Category Accuracy, Tolerance Accuracies (Within 10, 20, 30 Points),
Global Moran's I Spatial Correlation, And Generates Diagnostic Visualization Plots.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")  # Headless Non-Interactive Backend
import matplotlib.pyplot as plt

from Model3.Src.Config import (
    IDW_METRICS_JSON,
    PER_STATION_BREAKDOWN_CSV,
    PLOTS_DIR,
    RESULTS_DIR,
)
from Model3.Src.CpcbAqi import GetAqiCategory
from Model3.Src.SpatialFeatures import HaversineDistance


def ComputeMetrics(
    ActualValues: np.ndarray,
    PredictedValues: np.ndarray,
    ModelLabel: str = "Model",
) -> Dict[str, Union[float, int, str]]:
    """
    Calculate Comprehensive Geospatial Performance Metrics.
    Computes MAE, RMSE, R^2, Category Accuracy, And Tolerance Accuracies.
    """
    ActualArray = np.asarray(ActualValues, dtype=float)
    PredictedArray = np.asarray(PredictedValues, dtype=float)

    ValidMask = ~np.isnan(ActualArray) & ~np.isnan(PredictedArray)
    TrueValues = ActualArray[ValidMask]
    ForecastValues = PredictedArray[ValidMask]

    SampleCount = len(TrueValues)
    if SampleCount == 0:
        return {"Count": 0, "Status": "No Valid Observation Data"}

    MeanAbsoluteErr = float(mean_absolute_error(TrueValues, ForecastValues))
    RootMeanSquaredErr = float(np.sqrt(mean_squared_error(TrueValues, ForecastValues)))
    RSquaredScore = float(r2_score(TrueValues, ForecastValues))

    # Tolerance Accuracies Within Defined Error Bounds
    AbsoluteErrors = np.abs(TrueValues - ForecastValues)
    WithinTenPointsPct = float(np.mean(AbsoluteErrors <= 10.0) * 100.0)
    WithinTwentyPointsPct = float(np.mean(AbsoluteErrors <= 20.0) * 100.0)
    WithinThirtyPointsPct = float(np.mean(AbsoluteErrors <= 30.0) * 100.0)

    # Health Descriptor Category Agreement
    TrueCategories = [GetAqiCategory(Val) for Val in TrueValues]
    PredictedCategories = [GetAqiCategory(Val) for Val in ForecastValues]
    CategoryAccuracyPct = float(np.mean([T == P for T, P in zip(TrueCategories, PredictedCategories)]) * 100.0)

    return {
        "model_label": ModelLabel,
        "sample_count": SampleCount,
        "mae": round(MeanAbsoluteErr, 2),
        "rmse": round(RootMeanSquaredErr, 2),
        "r2": round(RSquaredScore, 4),
        "category_accuracy_pct": round(CategoryAccuracyPct, 2),
        "tolerance_within_10_pct": round(WithinTenPointsPct, 2),
        "tolerance_within_20_pct": round(WithinTwentyPointsPct, 2),
        "tolerance_within_30_pct": round(WithinThirtyPointsPct, 2),
    }


def ComputeMoransI(
    StationLatitudes: np.ndarray,
    StationLongitudes: np.ndarray,
    ResidualValues: np.ndarray,
    MaxDistanceKm: float = 35.0,
) -> Dict[str, float]:
    """
    Calculate Global Moran's I Spatial Autocorrelation On IDW Residual Errors.
    Values Near Zero Indicate Absence Of Spatially Correlated Error Structure.
    """
    Lats = np.asarray(StationLatitudes, dtype=float)
    Lons = np.asarray(StationLongitudes, dtype=float)
    Res = np.asarray(ResidualValues, dtype=float)

    ValidMask = ~np.isnan(Res) & ~np.isnan(Lats) & ~np.isnan(Lons)
    Lats = Lats[ValidMask]
    Lons = Lons[ValidMask]
    Res = Res[ValidMask]

    PointCount = len(Res)
    if PointCount < 4:
        return {"morans_i": 0.0, "expected_i": 0.0, "spatial_structure_detected": False}

    MeanResidual = np.mean(Res)
    Deviations = Res - MeanResidual
    VarianceDenominator = np.sum(Deviations ** 2)

    if VarianceDenominator == 0:
        return {"morans_i": 0.0, "expected_i": 0.0, "spatial_structure_detected": False}

    # Inverse Distance Spatial Weight Matrix
    WeightMatrix = np.zeros((PointCount, PointCount), dtype=float)
    for i in range(PointCount):
        PairDistances = HaversineDistance(Lats[i], Lons[i], Lats, Lons)
        for j in range(PointCount):
            if i != j and 0 < PairDistances[j] <= MaxDistanceKm:
                WeightMatrix[i, j] = 1.0 / PairDistances[j]

    TotalWeightSum = np.sum(WeightMatrix)
    if TotalWeightSum == 0:
        return {"morans_i": 0.0, "expected_i": 0.0, "spatial_structure_detected": False}

    NumeratorSum = 0.0
    for i in range(PointCount):
        for j in range(PointCount):
            if WeightMatrix[i, j] > 0:
                NumeratorSum += WeightMatrix[i, j] * Deviations[i] * Deviations[j]

    MoransIValue = (PointCount / TotalWeightSum) * (NumeratorSum / VarianceDenominator)
    ExpectedIValue = -1.0 / (PointCount - 1)

    return {
        "morans_i": round(float(MoransIValue), 4),
        "expected_i": round(float(ExpectedIValue), 4),
        "spatial_structure_detected": bool(MoransIValue > ExpectedIValue + 0.10),
    }


def EvaluateLosoDataFrame(
    EvaluationDataFrame: pd.DataFrame,
    PredictionColumn: str = "idw_aqi_direct",
    ActualColumn: str = "actual_aqi",
) -> Tuple[Dict, pd.DataFrame]:
    """
    Run Complete Cross-Validation Evaluation Across Global Regimes And Stratified Slices.
    """
    OverallMetrics = ComputeMetrics(EvaluationDataFrame[ActualColumn], EvaluationDataFrame[PredictionColumn], ModelLabel="IDW_Overall")

    RushHourMask = EvaluationDataFrame["is_rush_hour"] == 1
    RushHourMetrics = ComputeMetrics(EvaluationDataFrame.loc[RushHourMask, ActualColumn], EvaluationDataFrame.loc[RushHourMask, PredictionColumn], ModelLabel="Rush_Hours")
    OffPeakMetrics = ComputeMetrics(EvaluationDataFrame.loc[~RushHourMask, ActualColumn], EvaluationDataFrame.loc[~RushHourMask, PredictionColumn], ModelLabel="Off_Peak_Hours")

    SevereEpisodeMask = EvaluationDataFrame["is_severe_episode"] == 1
    SevereEpisodeMetrics = ComputeMetrics(EvaluationDataFrame.loc[SevereEpisodeMask, ActualColumn], EvaluationDataFrame.loc[SevereEpisodeMask, PredictionColumn], ModelLabel="Severe_Episodes_AQI_gt_300")
    NormalDayMetrics = ComputeMetrics(EvaluationDataFrame.loc[~SevereEpisodeMask, ActualColumn], EvaluationDataFrame.loc[~SevereEpisodeMask, PredictionColumn], ModelLabel="Normal_Days_AQI_le_300")

    StationBreakdownRows = []
    for StationIdKey, StationGroup in EvaluationDataFrame.groupby("target_station_id"):
        StationMetricRow = ComputeMetrics(StationGroup[ActualColumn], StationGroup[PredictionColumn], ModelLabel=StationIdKey)
        StationMetricRow["target_station_id"] = StationIdKey
        StationMetricRow["target_station_name"] = StationGroup["target_station_name"].iloc[0]
        StationMetricRow["lat"] = StationGroup["target_lat"].iloc[0]
        StationMetricRow["lon"] = StationGroup["target_lon"].iloc[0]
        StationMetricRow["mean_residual"] = round(float((StationGroup[ActualColumn] - StationGroup[PredictionColumn]).mean()), 2)
        StationBreakdownRows.append(StationMetricRow)

    StationBreakdownDataFrame = pd.DataFrame(StationBreakdownRows).sort_values("mae", ascending=False).reset_index(drop=True)

    MoranAnalysisResult = ComputeMoransI(
        StationBreakdownDataFrame["lat"].to_numpy(),
        StationBreakdownDataFrame["lon"].to_numpy(),
        StationBreakdownDataFrame["mean_residual"].to_numpy(),
    )

    ConsolidatedReport = {
        "overall": OverallMetrics,
        "rush_hours": RushHourMetrics,
        "off_peak_hours": OffPeakMetrics,
        "severe_episodes": SevereEpisodeMetrics,
        "normal_days": NormalDayMetrics,
        "spatial_autocorrelation": MoranAnalysisResult,
    }

    return ConsolidatedReport, StationBreakdownDataFrame


def GenerateEvaluationPlots(
    SampleDataFrame: pd.DataFrame,
    StationMetricsDataFrame: pd.DataFrame,
    PredictionColumn: str = "idw_aqi_direct",
    ActualColumn: str = "actual_aqi",
) -> List[str]:
    """
    Generate High-Resolution Diagnostic Charts With Title Case Formatting.
    Saves ErrorByStation.png, ErrorByHour.png, PredictedVsActual.png, And StationResidualMap.png.
    """
    GeneratedPlotFilePaths = []

    # 1. Error By Station Horizontal Bar Chart
    plt.figure(figsize=(12, 6))
    OrderedStations = StationMetricsDataFrame.sort_values("mae", ascending=True)
    plt.barh(OrderedStations["target_station_name"].str.split(",").str[0], OrderedStations["mae"], color="#1f77b4")
    plt.xlabel("Mean Absolute Error In AQI Points", fontsize=11)
    plt.title("AirRoute Model 3 — Leave-One-Station-Out Mean Absolute Error By Station", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    PlotFileOne = PLOTS_DIR / "ErrorByStation.png"
    plt.savefig(PlotFileOne, dpi=200)
    plt.close()
    GeneratedPlotFilePaths.append(str(PlotFileOne))

    # 2. Error By Hour Profile Plot
    WorkingDataFrame = SampleDataFrame.copy()
    WorkingDataFrame["absolute_error"] = np.abs(WorkingDataFrame[ActualColumn] - WorkingDataFrame[PredictionColumn])
    HourlyMeanSeries = WorkingDataFrame.groupby("hour")["absolute_error"].mean()

    plt.figure(figsize=(9, 5))
    plt.plot(HourlyMeanSeries.index, HourlyMeanSeries.values, marker="o", color="#d62728", linewidth=2.5)
    plt.axvspan(8, 11, color="orange", alpha=0.2, label="Morning Traffic Peak")
    plt.axvspan(17, 21, color="red", alpha=0.15, label="Evening Traffic Peak")
    plt.xlabel("Hour Of Day In Indian Standard Time", fontsize=11)
    plt.ylabel("Mean Absolute Error In AQI Points", fontsize=11)
    plt.title("AirRoute Model 3 — IDW Estimation Error Profile By Hour Of Day", fontsize=13, fontweight="bold")
    plt.xticks(range(0, 24, 2))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    PlotFileTwo = PLOTS_DIR / "ErrorByHour.png"
    plt.savefig(PlotFileTwo, dpi=200)
    plt.close()
    GeneratedPlotFilePaths.append(str(PlotFileTwo))

    # 3. Predicted Versus Actual 1:1 Scatter Plot
    plt.figure(figsize=(7, 7))
    RandomSample = SampleDataFrame.sample(min(1200, len(SampleDataFrame)), random_state=42)
    plt.scatter(RandomSample[ActualColumn], RandomSample[PredictionColumn], alpha=0.35, edgecolors="none", color="#2ca02c")
    MaxCoordinateBound = max(RandomSample[ActualColumn].max(), RandomSample[PredictionColumn].max()) + 20
    plt.plot([0, MaxCoordinateBound], [0, MaxCoordinateBound], "k--", label="Ideal 1:1 Prediction Line")
    plt.xlabel("Observed Ground Truth AQI Points", fontsize=11)
    plt.ylabel("Estimated Spatial IDW AQI Points", fontsize=11)
    plt.title("Leave-One-Station-Out Validation: Predicted Versus Actual AQI", fontsize=13, fontweight="bold")
    plt.xlim(0, MaxCoordinateBound)
    plt.ylim(0, MaxCoordinateBound)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    PlotFileThree = PLOTS_DIR / "PredictedVsActual.png"
    plt.savefig(PlotFileThree, dpi=200)
    plt.close()
    GeneratedPlotFilePaths.append(str(PlotFileThree))

    # 4. Station Spatial Residual Distribution Map
    plt.figure(figsize=(8, 8))
    ScatterPlot = plt.scatter(
        StationMetricsDataFrame["lon"],
        StationMetricsDataFrame["lat"],
        c=StationMetricsDataFrame["mean_residual"],
        cmap="coolwarm",
        s=120,
        edgecolors="black",
        linewidth=1,
    )
    plt.colorbar(ScatterPlot, label="Mean Residual Error (Observed Minus Estimated AQI)")
    for _, StationRecord in StationMetricsDataFrame.iterrows():
        ShortStationName = StationRecord["target_station_name"].split(",")[0][:12]
        plt.annotate(ShortStationName, (StationRecord["lon"] + 0.005, StationRecord["lat"] + 0.005), fontsize=8, alpha=0.85)

    plt.xlabel("Longitude Degrees East", fontsize=11)
    plt.ylabel("Latitude Degrees North", fontsize=11)
    plt.title("Delhi NCR Stations: Spatial Distribution Of Mean IDW Residuals", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    PlotFileFour = PLOTS_DIR / "StationResidualMap.png"
    plt.savefig(PlotFileFour, dpi=200)
    plt.close()
    GeneratedPlotFilePaths.append(str(PlotFileFour))

    return GeneratedPlotFilePaths


# Backward Compatibility Aliases
compute_metrics = ComputeMetrics
compute_morans_i = ComputeMoransI
evaluate_loso_dataframe = EvaluateLosoDataFrame
generate_evaluation_plots = GenerateEvaluationPlots
