"""
Data Ingestion, Quality Control (QC), And Leak-Free Spatial Dataset Pipeline.
Constructs StationHistory.parquet, SpatialWindows.parquet, And TestTargets.parquet.
Ensures Complete Zero-Leakage Spatial Isolation In Leave-One-Station-Out (LOSO) Validation.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from Model3.Src.Config import (
    DELHI_STATIONS,
    EPISODE_THRESHOLD_SEVERE,
    EVENING_RUSH_HOURS,
    MIN_VALID_NEIGHBORS,
    MORNING_RUSH_HOURS,
    QC_BOUNDS,
    RANDOM_SEED,
    SPATIAL_HOLDOUT_RATIO,
    SPATIAL_WINDOWS_PARQUET,
    STATION_HISTORY_PARQUET,
    STUCK_SENSOR_MAX_HOURS,
    TEMPORAL_HOLDOUT_RATIO,
    TEST_TARGETS_PARQUET,
)
from Model3.Src.CpcbAqi import CalculateCpcbAqi
from Model3.Src.Idw import IdwInterpolatePoint


def GenerateSynchronizedDelhiHistory(
    NumberOfDays: int = 45,
    FrequencyHours: int = 1,
    RandomSeedValue: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate Physically Calibrated Synchronized Historical Observations
    For All Thirty-Eight Official Delhi NCR CAAQMS Monitoring Stations.
    Reflects Regional Meteorological Waves, Traffic Rush Hours, And Sensor Noise.
    """
    RandomState = np.random.RandomState(RandomSeedValue)
    StationsDataFrame = pd.DataFrame(DELHI_STATIONS)
    NumberOfStations = len(StationsDataFrame)

    StartTimestamp = datetime(2025, 11, 1, 0, 0)
    AllTimestamps = [
        StartTimestamp + timedelta(hours=StepIndex * FrequencyHours)
        for StepIndex in range(NumberOfDays * 24 // FrequencyHours)
    ]
    TotalTimestamps = len(AllTimestamps)

    CompiledRecords: List[Dict] = []
    CenterLatitude = StationsDataFrame["lat"].mean()
    CenterLongitude = StationsDataFrame["lon"].mean()

    # Pre-Compute Synoptic Multi-Day Atmospheric Episodes Across Delhi NCR
    TimeIndices = np.arange(TotalTimestamps)
    SynopticWave = (
        160.0
        + 80.0 * np.sin(2 * np.pi * TimeIndices / (7 * 24))
        + 45.0 * np.sin(2 * np.pi * TimeIndices / (18 * 24))
    )

    for StepIndex, CurrentTimestamp in enumerate(AllTimestamps):
        CurrentHour = CurrentTimestamp.hour
        IsRushHourFlag = 1 if (CurrentHour in MORNING_RUSH_HOURS or CurrentHour in EVENING_RUSH_HOURS) else 0

        # Diurnal Ambient Temperature Cycle
        DiurnalTemperature = 22.0 + 7.0 * np.sin(np.pi * (CurrentHour - 8) / 12) + RandomState.normal(0, 1.0)
        CleanTemperature = float(np.clip(DiurnalTemperature, 8.0, 38.0))

        # Diurnal Boundary Layer Atmospheric Inversion (Higher Particulate At Night And Morning)
        BoundaryInversionFactor = 1.0 + 0.38 * np.cos(2 * np.pi * (CurrentHour - 3) / 24)
        RushHourMultiplier = 1.25 if IsRushHourFlag else 1.0

        BasePm25Concentration = SynopticWave[StepIndex] * BoundaryInversionFactor * RushHourMultiplier
        BaseWindSpeed = float(np.clip(2.5 + 1.5 * np.sin(2 * np.pi * StepIndex / 24) + RandomState.normal(0, 0.5), 0.5, 12.0))
        BaseWindDirection = float((290.0 + 40.0 * np.sin(2 * np.pi * StepIndex / (4 * 24)) + RandomState.normal(0, 10)) % 360)

        for _, StationRow in StationsDataFrame.iterrows():
            StationIdentifier = str(StationRow["station_id"])
            StationFullName = str(StationRow["name"])
            StationLatitude = float(StationRow["lat"])
            StationLongitude = float(StationRow["lon"])

            # Localized Emission Source Multiplier
            LocalSourceMultiplier = 1.0
            if "Anand Vihar" in StationFullName or "Jahangirpuri" in StationFullName or "Wazirpur" in StationFullName or "Bawana" in StationFullName:
                LocalSourceMultiplier = 1.35
            elif "Aya Nagar" in StationFullName or "Lodhi Road" in StationFullName or "IGI Airport" in StationFullName:
                LocalSourceMultiplier = 0.82

            RegionalDrift = (StationLatitude - CenterLatitude) * 15.0 - (StationLongitude - CenterLongitude) * 10.0
            SensorStochasticNoise = RandomState.normal(0, 8.0)

            Pm25Value = float(np.clip((BasePm25Concentration * LocalSourceMultiplier + RegionalDrift + SensorStochasticNoise), 15.0, 650.0))
            Pm10Value = float(np.clip(Pm25Value * RandomState.uniform(1.6, 2.1) + RandomState.normal(0, 12.0), 30.0, 950.0))
            No2Value = float(np.clip((35.0 + 25.0 * IsRushHourFlag + 15.0 * (LocalSourceMultiplier - 1.0) * 10) + RandomState.normal(0, 5.0), 5.0, 250.0))
            So2Value = float(np.clip(14.0 + 8.0 * (LocalSourceMultiplier - 0.9) + RandomState.normal(0, 2.0), 2.0, 90.0))
            CoValue = float(np.clip(0.8 + 0.6 * IsRushHourFlag + 0.4 * LocalSourceMultiplier + RandomState.normal(0, 0.1), 0.1, 10.0))
            O3Value = float(np.clip(25.0 + 35.0 * np.sin(np.pi * np.clip(CurrentHour - 6, 0, 12) / 12) + RandomState.normal(0, 4.0), 5.0, 180.0))

            RelativeHumidity = float(np.clip(65.0 - 20.0 * np.sin(np.pi * (CurrentHour - 8) / 12) + RandomState.normal(0, 3.0), 15.0, 95.0))
            AtmosphericPressure = float(np.clip(1012.0 - 3.0 * np.sin(np.pi * CurrentHour / 12) + RandomState.normal(0, 0.5), 990.0, 1030.0))

            CompiledRecords.append({
                "station_id": StationIdentifier,
                "station_name": StationFullName,
                "latitude": StationLatitude,
                "longitude": StationLongitude,
                "timestamp": CurrentTimestamp,
                "pm2_5": round(Pm25Value, 2),
                "pm10": round(Pm10Value, 2),
                "no2": round(No2Value, 2),
                "so2": round(So2Value, 2),
                "co": round(CoValue, 2),
                "o3": round(O3Value, 2),
                "temperature": round(CleanTemperature, 1),
                "humidity": round(RelativeHumidity, 1),
                "wind_speed": round(BaseWindSpeed, 1),
                "wind_direction": round(BaseWindDirection, 1),
                "pressure": round(AtmosphericPressure, 1),
            })

    GeneratedDataFrame = pd.DataFrame(CompiledRecords)

    # Realistic Quality Control Edge Cases (Occasional Sensor Freeze And Missing Records)
    StuckStationId = "DEL_006"
    StuckCondition = (
        (GeneratedDataFrame["station_id"] == StuckStationId)
        & (GeneratedDataFrame["timestamp"] >= AllTimestamps[10])
        & (GeneratedDataFrame["timestamp"] <= AllTimestamps[18])
    )
    if StuckCondition.sum() > 0:
        BaselineValue = GeneratedDataFrame.loc[StuckCondition, "pm2_5"].iloc[0]
        GeneratedDataFrame.loc[StuckCondition, "pm2_5"] = BaselineValue

    MissingSampleIndices = RandomState.choice(GeneratedDataFrame.index, size=int(0.015 * len(GeneratedDataFrame)), replace=False)
    GeneratedDataFrame.loc[MissingSampleIndices, "pm2_5"] = np.nan

    return GeneratedDataFrame


def ApplyQualityControl(InputDataFrame: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Rigorous Quality Control Filters And Calculate Official CPCB AQI Values.
    Marks Rows With Explicit Audit Flags (VALID, STUCK_SENSOR, NEGATIVE_OUTLIER).
    """
    CleanDataFrame = InputDataFrame.copy()
    CleanDataFrame["qc_flag"] = "VALID"

    # Environmental Upper And Lower Plausibility Checks
    for PollutantKey, (LowerBound, UpperBound) in QC_BOUNDS.items():
        if PollutantKey in CleanDataFrame.columns:
            NegativeMask = CleanDataFrame[PollutantKey] < 0
            CleanDataFrame.loc[NegativeMask, "qc_flag"] = "NEGATIVE_OUTLIER"

            OutOfBoundsMask = (CleanDataFrame[PollutantKey] < LowerBound) | (CleanDataFrame[PollutantKey] > UpperBound)
            CleanDataFrame.loc[OutOfBoundsMask, "qc_flag"] = "PHYSICALLY_IMPOSSIBLE"

    # Consecutive Invariant Sensor Reading Check (Stuck Sensor Detection)
    CleanDataFrame = CleanDataFrame.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
    CleanDataFrame["is_stuck"] = False

    for StationIdKey, StationGroup in CleanDataFrame.groupby("station_id"):
        for PollutantName in ["pm2_5", "pm10"]:
            DifferenceSeries = StationGroup[PollutantName].diff().fillna(1.0)
            ZeroDeltaFlag = (DifferenceSeries == 0).astype(int)
            ConsecutiveRunLength = ZeroDeltaFlag.groupby((ZeroDeltaFlag != ZeroDeltaFlag.shift()).cumsum()).cumsum()
            StuckRowIndices = StationGroup.index[ConsecutiveRunLength >= STUCK_SENSOR_MAX_HOURS]
            CleanDataFrame.loc[StuckRowIndices, "qc_flag"] = "STUCK_SENSOR"
            CleanDataFrame.loc[StuckRowIndices, "is_stuck"] = True

    # Compute Ground Truth Official Indian CPCB NAQI
    CalculatedAqiList = []
    ProminentPollutantList = []

    for _, DataRow in CleanDataFrame.iterrows():
        if DataRow["qc_flag"] in ("NEGATIVE_OUTLIER", "PHYSICALLY_IMPOSSIBLE", "STUCK_SENSOR"):
            CalculatedAqiList.append(np.nan)
            ProminentPollutantList.append(None)
            continue

        ObservedPollutants = {
            "pm2_5": DataRow.get("pm2_5", np.nan),
            "pm10": DataRow.get("pm10", np.nan),
            "no2": DataRow.get("no2", np.nan),
            "so2": DataRow.get("so2", np.nan),
            "co": DataRow.get("co", np.nan),
            "o3": DataRow.get("o3", np.nan),
        }
        FinalAqi, LeadPollutant, _ = CalculateCpcbAqi(ObservedPollutants, StrictCpcbRules=True)
        CalculatedAqiList.append(FinalAqi)
        ProminentPollutantList.append(LeadPollutant)

    CleanDataFrame["aqi"] = CalculatedAqiList
    CleanDataFrame["prominent_pollutant"] = ProminentPollutantList

    MissingAqiMask = CleanDataFrame["aqi"].isna() & (CleanDataFrame["qc_flag"] == "VALID")
    CleanDataFrame.loc[MissingAqiMask, "qc_flag"] = "INSUFFICIENT_POLLUTANTS"

    return CleanDataFrame


def SplitSpatialTemporalHoldouts(
    ObservationDataFrame: pd.DataFrame,
    SpatialHoldoutFraction: float = SPATIAL_HOLDOUT_RATIO,
    TemporalHoldoutFraction: float = TEMPORAL_HOLDOUT_RATIO,
    RandomSeedNumber: int = RANDOM_SEED,
) -> Tuple[List[str], List[str], pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """
    Partition Observations Strictly Into Train And Pure Unseen Holdout Partitions.
    Reserves Twenty Percent Of Stations And The Most Recent Twenty Percent Timestamps.
    """
    DistinctStations = sorted(ObservationDataFrame["station_id"].unique().tolist())
    HoldoutStationCount = max(1, int(len(DistinctStations) * SpatialHoldoutFraction))

    RandomGenerator = np.random.RandomState(RandomSeedNumber)
    TestStationList = sorted(RandomGenerator.choice(DistinctStations, size=HoldoutStationCount, replace=False).tolist())
    TrainStationList = [StationId for StationId in DistinctStations if StationId not in TestStationList]

    ChronologicalTimestamps = sorted(ObservationDataFrame["timestamp"].unique())
    TemporalCutoffIndex = int(len(ChronologicalTimestamps) * (1.0 - TemporalHoldoutFraction))
    TemporalCutoffTimestamp = ChronologicalTimestamps[TemporalCutoffIndex]

    TestPartitionMask = (
        (ObservationDataFrame["station_id"].isin(TestStationList))
        | (ObservationDataFrame["timestamp"] >= TemporalCutoffTimestamp)
    )
    TrainPartitionMask = (
        (ObservationDataFrame["station_id"].isin(TrainStationList))
        & (ObservationDataFrame["timestamp"] < TemporalCutoffTimestamp)
    )

    TrainDataFrame = ObservationDataFrame[TrainPartitionMask].copy().reset_index(drop=True)
    TestDataFrame = ObservationDataFrame[TestPartitionMask].copy().reset_index(drop=True)

    return TrainStationList, TestStationList, TemporalCutoffTimestamp, TrainDataFrame, TestDataFrame


def GenerateLosoDataset(
    StationObservations: pd.DataFrame,
    KNearest: int = 8,
    MaxRadiusKm: float = 25.0,
    DistancePower: float = 1.0,
    MinNeighbors: int = MIN_VALID_NEIGHBORS,
) -> pd.DataFrame:
    """
    Generate Leave-One-Station-Out (LOSO) Simulation Windows With Zero Leakage.
    Guarantees Target Station Is Strictly Excluded From Candidate Neighbors And Statistics.
    """
    ValidRecords = StationObservations[StationObservations["qc_flag"] == "VALID"].copy()
    ValidRecords = ValidRecords[~ValidRecords["aqi"].isna()]

    LosoRows: List[Dict] = []

    for TimestampKey, TimestampGroup in ValidRecords.groupby("timestamp"):
        HourOfDay = TimestampKey.hour
        RushHourIndicator = 1 if (HourOfDay in MORNING_RUSH_HOURS or HourOfDay in EVENING_RUSH_HOURS) else 0

        StationIdArray = TimestampGroup["station_id"].to_numpy()
        LatitudeArray = TimestampGroup["latitude"].to_numpy()
        LongitudeArray = TimestampGroup["longitude"].to_numpy()
        AqiArray = TimestampGroup["aqi"].to_numpy()
        Pm25Array = TimestampGroup["pm2_5"].to_numpy()
        Pm10Array = TimestampGroup["pm10"].to_numpy()

        for _, TargetRow in TimestampGroup.iterrows():
            TargetId = str(TargetRow["station_id"])
            TargetLat = float(TargetRow["latitude"])
            TargetLon = float(TargetRow["longitude"])
            TargetActualAqi = float(TargetRow["aqi"])
            TargetPm25 = float(TargetRow["pm2_5"])
            TargetPm10 = float(TargetRow["pm10"])

            # Strict Zero-Leakage Exclusion Of Target Station
            CandidateNeighborMask = StationIdArray != TargetId
            if np.sum(CandidateNeighborMask) < MinNeighbors:
                continue

            CandidateLats = LatitudeArray[CandidateNeighborMask]
            CandidateLons = LongitudeArray[CandidateNeighborMask]
            CandidateAqis = AqiArray[CandidateNeighborMask]
            CandidatePm25 = Pm25Array[CandidateNeighborMask]
            CandidatePm10 = Pm10Array[CandidateNeighborMask]
            CandidateStationIds = [Sid for Sid in StationIdArray if Sid != TargetId]

            # 1. Direct IDW AQI Estimation
            DirectIdwResult = IdwInterpolatePoint(
                TargetLatitude=TargetLat,
                TargetLongitude=TargetLon,
                StationLatitudes=CandidateLats,
                StationLongitudes=CandidateLons,
                StationValues=CandidateAqis,
                StationIdentifiers=CandidateStationIds,
                KNearest=KNearest,
                MaxRadiusKm=MaxRadiusKm,
                DistancePower=DistancePower,
                MinNeighbors=MinNeighbors,
            )

            # Mathematical Proof Of Zero Leakage
            assert TargetId not in DirectIdwResult.neighbor_ids, f"Leakage Detected! Target Station {TargetId} Found In Neighbor Pool"

            # 2. Pollutant-First IDW Interpolation
            Pm25IdwResult = IdwInterpolatePoint(
                TargetLatitude=TargetLat,
                TargetLongitude=TargetLon,
                StationLatitudes=CandidateLats,
                StationLongitudes=CandidateLons,
                StationValues=CandidatePm25,
                StationIdentifiers=CandidateStationIds,
                KNearest=KNearest,
                MaxRadiusKm=MaxRadiusKm,
                DistancePower=DistancePower,
                MinNeighbors=MinNeighbors,
            )
            Pm10IdwResult = IdwInterpolatePoint(
                TargetLatitude=TargetLat,
                TargetLongitude=TargetLon,
                StationLatitudes=CandidateLats,
                StationLongitudes=CandidateLons,
                StationValues=CandidatePm10,
                StationIdentifiers=CandidateStationIds,
                KNearest=KNearest,
                MaxRadiusKm=MaxRadiusKm,
                DistancePower=DistancePower,
                MinNeighbors=MinNeighbors,
            )

            PollutantFirstAqiValue = np.nan
            if Pm25IdwResult.status == "OK" and Pm10IdwResult.status == "OK":
                ConcentrationsDict = {"pm2_5": Pm25IdwResult.estimate, "pm10": Pm10IdwResult.estimate}
                PollutantFirstAqiValue, _, _ = CalculateCpcbAqi(ConcentrationsDict, StrictCpcbRules=False)

            if DirectIdwResult.status == "OK":
                ResidualValue = TargetActualAqi - DirectIdwResult.estimate

                LosoRows.append({
                    "timestamp": TimestampKey,
                    "target_station_id": TargetId,
                    "target_station_name": TargetRow["station_name"],
                    "target_lat": TargetLat,
                    "target_lon": TargetLon,
                    "actual_aqi": TargetActualAqi,
                    "actual_pm2_5": TargetPm25,
                    "actual_pm10": TargetPm10,
                    "idw_aqi_direct": DirectIdwResult.estimate,
                    "idw_aqi_pollutant_first": PollutantFirstAqiValue,
                    "idw_residual": ResidualValue,
                    "idw_status": DirectIdwResult.status,
                    "valid_neighbor_count": DirectIdwResult.valid_neighbor_count,
                    "nearest_distance_km": DirectIdwResult.nearest_distance_km,
                    "mean_distance_km": DirectIdwResult.mean_distance_km,
                    "max_distance_km": DirectIdwResult.max_distance_km,
                    "neighbor_min_aqi": DirectIdwResult.neighbor_min,
                    "neighbor_mean_aqi": DirectIdwResult.neighbor_mean,
                    "neighbor_max_aqi": DirectIdwResult.neighbor_max,
                    "neighbor_std_aqi": DirectIdwResult.neighbor_std,
                    "idw_pm2_5": Pm25IdwResult.estimate if Pm25IdwResult.status == "OK" else np.nan,
                    "idw_pm10": Pm10IdwResult.estimate if Pm10IdwResult.status == "OK" else np.nan,
                    "hour": HourOfDay,
                    "is_rush_hour": RushHourIndicator,
                    "temperature": TargetRow.get("temperature", np.nan),
                    "humidity": TargetRow.get("humidity", np.nan),
                    "wind_speed": TargetRow.get("wind_speed", np.nan),
                    "wind_direction": TargetRow.get("wind_direction", np.nan),
                    "pressure": TargetRow.get("pressure", np.nan),
                    "is_severe_episode": 1 if TargetActualAqi >= EPISODE_THRESHOLD_SEVERE else 0,
                })

    return pd.DataFrame(LosoRows)


def BuildAndSaveDataPipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute End-To-End Ingestion, Quality Control, And Holdout Partitioning Pipeline.
    Writes StationHistory.parquet, SpatialWindows.parquet, And TestTargets.parquet.
    """
    print("[Pipeline] 1. Ingesting And Synthesizing Synchronized Delhi Station Observations...")
    RawDataFrame = GenerateSynchronizedDelhiHistory(NumberOfDays=45, FrequencyHours=1)

    print("[Pipeline] 2. Applying Quality Control And Official CPCB NAQI Standard...")
    CleanQcDataFrame = ApplyQualityControl(RawDataFrame)

    CleanQcDataFrame.to_parquet(STATION_HISTORY_PARQUET, index=False)
    print(f"[Pipeline] -> Successfully Saved {len(CleanQcDataFrame)} Records To {STATION_HISTORY_PARQUET}")

    print("[Pipeline] 3. Partitioning Spatial And Temporal Holdouts...")
    TrainStations, TestStations, CutoffTimestamp, TrainSubset, TestSubset = SplitSpatialTemporalHoldouts(CleanQcDataFrame)
    print(f"[Pipeline] -> Training Stations: {len(TrainStations)}, Spatial Holdout Stations: {len(TestStations)} ({TestStations})")
    print(f"[Pipeline] -> Temporal Horizon Cutoff: {CutoffTimestamp}")

    print("[Pipeline] 4. Building Leave-One-Station-Out Training Windows...")
    TrainLosoDataFrame = GenerateLosoDataset(TrainSubset)
    TrainLosoDataFrame.to_parquet(SPATIAL_WINDOWS_PARQUET, index=False)
    print(f"[Pipeline] -> Successfully Saved {len(TrainLosoDataFrame)} Windows To {SPATIAL_WINDOWS_PARQUET}")

    print("[Pipeline] 5. Building Unseen Test Targets For Holdout Verification...")
    TestLosoDataFrame = GenerateLosoDataset(TestSubset)
    TestLosoDataFrame.to_parquet(TEST_TARGETS_PARQUET, index=False)
    print(f"[Pipeline] -> Successfully Saved {len(TestLosoDataFrame)} Test Targets To {TEST_TARGETS_PARQUET}")

    return CleanQcDataFrame, TrainLosoDataFrame, TestLosoDataFrame


# Lowercase Aliases For Backward Compatibility
generate_synchronized_delhi_history = GenerateSynchronizedDelhiHistory
apply_quality_control = ApplyQualityControl
split_spatial_temporal_holdouts = SplitSpatialTemporalHoldouts
generate_loso_dataset = GenerateLosoDataset
build_and_save_data_pipeline = BuildAndSaveDataPipeline


if __name__ == "__main__":
    BuildAndSaveDataPipeline()
