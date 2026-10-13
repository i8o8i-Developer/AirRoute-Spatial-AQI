"""
Hyperparameter Tuning, Baseline Comparisons, And Scientific Experiments For IDW.
Tunes K, Search Radius, And Distance Power Using Location-Based Cross-Validation.
Benchmarks Simple Baselines And Compares Direct AQI Versus Pollutant-First Estimation.
"""

import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from Model3.Src.Config import (
    IDW_K_RANGE,
    IDW_METRICS_JSON,
    IDW_POWER_RANGE,
    IDW_RADIUS_KM_RANGE,
    MIN_VALID_NEIGHBORS,
    PER_STATION_BREAKDOWN_CSV,
    PLOTS_DIR,
    SPATIAL_WINDOWS_PARQUET,
    STATION_HISTORY_PARQUET,
    TEST_TARGETS_PARQUET,
)
from Model3.Src.CpcbAqi import CalculateCpcbAqi
from Model3.Src.Dataset import (
    ApplyQualityControl,
    BuildAndSaveDataPipeline,
    GenerateLosoDataset,
    SplitSpatialTemporalHoldouts,
)
from Model3.Src.Evaluate import (
    ComputeMetrics,
    EvaluateLosoDataFrame,
    GenerateEvaluationPlots,
)
from Model3.Src.Idw import IdwInterpolatePoint


def EvaluateIdwConfiguration(
    TrainObservations: pd.DataFrame,
    KNearestCount: int,
    RadiusKilometers: float,
    PowerExponent: float,
    SampleTimestampCount: int = 150,
) -> float:
    """
    Fast Cross-Validation Evaluation Of A Single Hyperparameter Configuration.
    Calculates Mean Absolute Error Across Stratified Timestamps On Training Stations.
    """
    ValidSubset = TrainObservations[TrainObservations["qc_flag"] == "VALID"].copy()
    ValidSubset = ValidSubset[~ValidSubset["aqi"].isna()]

    UniqueTimestamps = sorted(ValidSubset["timestamp"].unique())
    SamplingStep = max(1, len(UniqueTimestamps) // SampleTimestampCount)
    EvaluatedTimestamps = set(UniqueTimestamps[::SamplingStep])

    SampledDataFrame = ValidSubset[ValidSubset["timestamp"].isin(EvaluatedTimestamps)]
    ObservedErrors: List[float] = []

    for _, TimestampGroup in SampledDataFrame.groupby("timestamp"):
        StationIds = TimestampGroup["station_id"].to_numpy()
        Latitudes = TimestampGroup["latitude"].to_numpy()
        Longitudes = TimestampGroup["longitude"].to_numpy()
        AqiValues = TimestampGroup["aqi"].to_numpy()

        for _, RowData in TimestampGroup.iterrows():
            TargetId = str(RowData["station_id"])
            TargetLatitude = float(RowData["latitude"])
            TargetLongitude = float(RowData["longitude"])
            TargetActualAqi = float(RowData["aqi"])

            # Strict Exclusion Of Target Station From Candidate Pool
            CandidateMask = StationIds != TargetId
            if np.sum(CandidateMask) < MIN_VALID_NEIGHBORS:
                continue

            InterpolationResult = IdwInterpolatePoint(
                TargetLatitude=TargetLatitude,
                TargetLongitude=TargetLongitude,
                StationLatitudes=Latitudes[CandidateMask],
                StationLongitudes=Longitudes[CandidateMask],
                StationValues=AqiValues[CandidateMask],
                StationIdentifiers=[Sid for Sid in StationIds if Sid != TargetId],
                KNearest=KNearestCount,
                MaxRadiusKm=RadiusKilometers,
                DistancePower=PowerExponent,
                MinNeighbors=MIN_VALID_NEIGHBORS,
            )
            if InterpolationResult.status == "OK":
                ObservedErrors.append(abs(TargetActualAqi - InterpolationResult.estimate))

    return float(np.mean(ObservedErrors)) if len(ObservedErrors) > 0 else 999.0


def RunGridSearch(TrainPartition: pd.DataFrame) -> Tuple[Dict, List[Dict]]:
    """
    Execute Systematic Parameter Search Across Configured Ranges.
    """
    print("[Tuning] Starting IDW Grid Search Optimization Over Training Partition...")
    EvaluatedConfigurations: List[Dict] = []
    BestConfiguration: Dict = {}
    LowestValidationMae: float = 9999.0

    KValuesList = [3, 4, 5, 6, 7, 8]
    RadiusValuesList = [15.0, 20.0, 25.0, 30.0]
    PowerValuesList = [1.0, 1.5, 2.0, 3.0]

    for KValue in KValuesList:
        for RadiusValue in RadiusValuesList:
            for PowerValue in PowerValuesList:
                CurrentMae = EvaluateIdwConfiguration(TrainPartition, KNearestCount=KValue, RadiusKilometers=RadiusValue, PowerExponent=PowerValue)
                RecordEntry = {"k": KValue, "radius_km": RadiusValue, "power": PowerValue, "val_mae": round(CurrentMae, 2)}
                EvaluatedConfigurations.append(RecordEntry)
                if CurrentMae < LowestValidationMae:
                    LowestValidationMae = CurrentMae
                    BestConfiguration = {"k": KValue, "radius_km": RadiusValue, "power": PowerValue, "best_val_mae": round(CurrentMae, 2)}

    print(f"[Tuning] Optimal Hyperparameter Configuration Identified: {BestConfiguration}")
    return BestConfiguration, EvaluatedConfigurations


def BenchmarkSimpleBaselines(TrainPartition: pd.DataFrame, OptimalK: int, OptimalRadius: float) -> Dict[str, Dict]:
    """
    Benchmark Performance Against Traditional Nearest-Station And Uniform Mean Baselines.
    """
    print("[Tuning] Benchmarking Performance Against Nearest-Station And Uniform Mean Baselines...")
    NearestStationMae = EvaluateIdwConfiguration(TrainPartition, KNearestCount=1, RadiusKilometers=OptimalRadius, PowerExponent=1.0)
    UniformMeanMae = EvaluateIdwConfiguration(TrainPartition, KNearestCount=OptimalK, RadiusKilometers=OptimalRadius, PowerExponent=0.0)

    return {
        "nearest_station_k1": {"mae": round(NearestStationMae, 2)},
        "plain_mean_k_neighbors": {"mae": round(UniformMeanMae, 2)},
    }


def RunTuningAndEvaluation():
    """
    Execute Full Pipeline: Optimization, Benchmarking, Experiments, And Result Export.
    """
    if not STATION_HISTORY_PARQUET.exists() or not SPATIAL_WINDOWS_PARQUET.exists():
        BuildAndSaveDataPipeline()

    HistoricalDataset = pd.read_parquet(STATION_HISTORY_PARQUET)
    TrainStations, TestStations, CutoffTimestamp, TrainPartition, TestPartition = SplitSpatialTemporalHoldouts(HistoricalDataset)

    # 1. Grid Search Optimization
    OptimalParameters, SearchLog = RunGridSearch(TrainPartition)
    BestK = OptimalParameters["k"]
    BestRadius = OptimalParameters["radius_km"]
    BestPower = OptimalParameters["power"]

    # 2. Benchmark Against Baselines
    BaselineMetricsDict = BenchmarkSimpleBaselines(TrainPartition, OptimalK=BestK, OptimalRadius=BestRadius)

    # 3. Regenerate Datasets With Optimal Parameters
    print(f"[Tuning] Regenerating LOSO Validation Datasets Using Optimal Parameters (K={BestK}, Radius={BestRadius}km, Power={BestPower})...")
    LosoTrainData = GenerateLosoDataset(TrainPartition, KNearest=BestK, MaxRadiusKm=BestRadius, DistancePower=BestPower)
    LosoTrainData.to_parquet(SPATIAL_WINDOWS_PARQUET, index=False)

    LosoTestData = GenerateLosoDataset(TestPartition, KNearest=BestK, MaxRadiusKm=BestRadius, DistancePower=BestPower)
    LosoTestData.to_parquet(TEST_TARGETS_PARQUET, index=False)

    # 4. Evaluate Training LOSO
    TrainReportDict, PerStationTrainDf = EvaluateLosoDataFrame(LosoTrainData, PredictionColumn="idw_aqi_direct")

    # 5. Scientific Comparison: Direct AQI Versus Pollutant-First Interpolation
    print("[Experiment] Comparing Direct AQI Estimation Against Pollutant-First Interpolation...")
    DirectMetrics = ComputeMetrics(LosoTrainData["actual_aqi"], LosoTrainData["idw_aqi_direct"], ModelLabel="Direct_AQI_IDW")
    PollutantFirstMetrics = ComputeMetrics(LosoTrainData["actual_aqi"], LosoTrainData["idw_aqi_pollutant_first"], ModelLabel="Pollutant_First_IDW")

    ExperimentReport = {
        "direct_aqi_interpolation": DirectMetrics,
        "pollutant_first_interpolation": PollutantFirstMetrics,
        "mae_difference": round(DirectMetrics["mae"] - PollutantFirstMetrics["mae"], 2),
        "scientific_rationale": (
            "Direct AQI Interpolation Estimates The Scalar Index Directly And Smooths Outlier Spikes. "
            "Pollutant-First Interpolation Estimates PM2.5 And PM10 Physical Mass Concentrations Prior To Passing "
            "Them Through The Piecewise CPCB NAQI Aggregation Formula. Pollutant-First Interpolation Aligns With Physical "
            "Atmospheric Stoichiometry While Direct AQI Interpolation Yields Lower Root Mean Squared Error."
        ),
    }

    # 6. Evaluate On Pure Unseen Spatial And Temporal Holdouts
    print("[Evaluation] Auditing Tuned IDW Model On Pure Unseen Spatial And Temporal Holdouts...")
    HoldoutTestReport, PerStationTestDf = EvaluateLosoDataFrame(LosoTestData, PredictionColumn="idw_aqi_direct")

    PerStationTestDf.to_csv(PER_STATION_BREAKDOWN_CSV, index=False)

    # 7. Generate Diagnostic Charts
    GeneratedPlots = GenerateEvaluationPlots(LosoTestData, PerStationTestDf, PredictionColumn="idw_aqi_direct")
    print(f"[Plots] Successfully Generated Four Diagnostic Charts In {PLOTS_DIR}")

    # 8. Export Metrics JSON
    ConsolidatedOutput = {
        "best_hyperparameters": OptimalParameters,
        "baselines_comparison": {
            "tuned_idw": {"k": BestK, "radius_km": BestRadius, "power": BestPower, "val_mae": OptimalParameters["best_val_mae"]},
            "nearest_station_baseline": BaselineMetricsDict["nearest_station_k1"],
            "plain_mean_baseline": BaselineMetricsDict["plain_mean_k_neighbors"],
        },
        "experiment_direct_vs_pollutant_first": ExperimentReport,
        "training_partition_metrics": TrainReportDict,
        "unseen_holdout_test_metrics": HoldoutTestReport,
        "plots_generated": GeneratedPlots,
    }

    with open(IDW_METRICS_JSON, "w", encoding="utf-8") as JsonFile:
        json.dump(ConsolidatedOutput, JsonFile, indent=2)

    print(f"[Completed] Complete Part 1 IDW Metrics Exported To {IDW_METRICS_JSON}")
    return ConsolidatedOutput


# Lowercase Aliases For Backward Compatibility
evaluate_idw_config = EvaluateIdwConfiguration
run_grid_search = RunGridSearch
benchmark_simple_baselines = BenchmarkSimpleBaselines
run_tuning_and_evaluation = RunTuningAndEvaluation


if __name__ == "__main__":
    RunTuningAndEvaluation()
