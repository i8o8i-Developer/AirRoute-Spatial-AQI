"""
High-Level Spatial Prediction Engine For AirRoute Model 3.
Provides Point-Level Coordinate Queries And Continuous 2D City-Wide Grid Surface Generation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from Model3.Src.Config import (
    DEFAULT_K,
    DEFAULT_POWER,
    DEFAULT_RADIUS_KM,
    GRID_CONFIG,
    MIN_VALID_NEIGHBORS,
    REGION_BOUNDS,
    STATION_HISTORY_PARQUET,
)
from Model3.Src.CpcbAqi import GetAqiCategory
from Model3.Src.Idw import IdwInterpolateGrid, IdwInterpolatePoint


class SpatialAqiPredictor:
    """
    Inference Service For Spatial AQI Estimation Queries Across Delhi NCR.
    """

    def __init__(self, StationDataFilePath: Optional[str] = None):
        ResolvedFilePath = StationDataFilePath or str(STATION_HISTORY_PARQUET)
        self.DatasetDataFrame = pd.read_parquet(ResolvedFilePath)
        # Select Most Recent Synchronized Observation For Each Monitored Station
        self.LatestObservations = (
            self.DatasetDataFrame[self.DatasetDataFrame["qc_flag"] == "VALID"]
            .sort_values("timestamp")
            .groupby("station_id")
            .last()
            .reset_index()
        )

    def PredictPoint(
        self,
        TargetLatitude: float,
        TargetLongitude: float,
        KNearestNeighbors: int = DEFAULT_K,
        SearchRadiusKm: float = DEFAULT_RADIUS_KM,
        PowerExponent: float = DEFAULT_POWER,
    ) -> Dict[str, Union[float, int, str]]:
        """
        Estimate Air Quality Index At Any Unmonitored Coordinate.
        """
        StationLats = self.LatestObservations["latitude"].to_numpy()
        StationLons = self.LatestObservations["longitude"].to_numpy()
        StationAqis = self.LatestObservations["aqi"].to_numpy()
        StationIdentifiers = self.LatestObservations["station_id"].tolist()

        PointResult = IdwInterpolatePoint(
            TargetLatitude=TargetLatitude,
            TargetLongitude=TargetLongitude,
            StationLatitudes=StationLats,
            StationLongitudes=StationLons,
            StationValues=StationAqis,
            StationIdentifiers=StationIdentifiers,
            KNearest=KNearestNeighbors,
            MaxRadiusKm=SearchRadiusKm,
            DistancePower=PowerExponent,
            MinNeighbors=MIN_VALID_NEIGHBORS,
        )

        CategoryDescriptor = GetAqiCategory(PointResult.estimate) if not np.isnan(PointResult.estimate) else "Unknown"

        # Geometric Confidence Score Calibration
        ConfidenceScore = 0.0
        if PointResult.status in ("OK", "EXACT_MATCH"):
            DistanceDecayFactor = np.clip(1.0 - (PointResult.nearest_distance_km / SearchRadiusKm), 0.0, 1.0)
            DensityFactor = np.clip(PointResult.valid_neighbor_count / KNearestNeighbors, 0.0, 1.0)
            ConfidenceScore = round(float(0.6 * DistanceDecayFactor + 0.4 * DensityFactor), 3)

        return {
            "latitude": TargetLatitude,
            "longitude": TargetLongitude,
            "estimated_aqi": round(PointResult.estimate, 1) if not np.isnan(PointResult.estimate) else None,
            "aqi_category": CategoryDescriptor,
            "confidence_score": ConfidenceScore,
            "estimation_status": PointResult.status,
            "valid_neighbor_count": PointResult.valid_neighbor_count,
            "nearest_distance_km": round(PointResult.nearest_distance_km, 2) if not np.isnan(PointResult.nearest_distance_km) else None,
            "mean_distance_km": round(PointResult.mean_distance_km, 2) if not np.isnan(PointResult.mean_distance_km) else None,
            "neighbor_min_aqi": round(PointResult.neighbor_min, 1) if not np.isnan(PointResult.neighbor_min) else None,
            "neighbor_max_aqi": round(PointResult.neighbor_max, 1) if not np.isnan(PointResult.neighbor_max) else None,
            "is_point_estimate": True,
            "model_architecture_stage": "Stage 1 — Tuned IDW Baseline",
        }

    def PredictGrid(
        self,
        LatitudeStep: float = GRID_CONFIG["lat_step"],
        LongitudeStep: float = GRID_CONFIG["lon_step"],
        KNearestNeighbors: int = DEFAULT_K,
        SearchRadiusKm: float = DEFAULT_RADIUS_KM,
        PowerExponent: float = DEFAULT_POWER,
    ) -> pd.DataFrame:
        """
        Generate City-Wide Continuous AQI Surface Matrix Across Delhi NCR.
        """
        LatitudeGridValues = np.arange(REGION_BOUNDS["lat_min"], REGION_BOUNDS["lat_max"] + 1e-5, LatitudeStep)
        LongitudeGridValues = np.arange(REGION_BOUNDS["lon_min"], REGION_BOUNDS["lon_max"] + 1e-5, LongitudeStep)
        MeshLatitudes, MeshLongitudes = np.meshgrid(LatitudeGridValues, LongitudeGridValues)
        FlattenedLatitudes = MeshLatitudes.ravel()
        FlattenedLongitudes = MeshLongitudes.ravel()

        StationLats = self.LatestObservations["latitude"].to_numpy()
        StationLons = self.LatestObservations["longitude"].to_numpy()
        StationAqis = self.LatestObservations["aqi"].to_numpy()
        StationIdentifiers = self.LatestObservations["station_id"].tolist()

        Estimates, Confidences, NeighborCounts = IdwInterpolateGrid(
            GridLatitudes=FlattenedLatitudes,
            GridLongitudes=FlattenedLongitudes,
            StationLatitudes=StationLats,
            StationLongitudes=StationLons,
            StationValues=StationAqis,
            StationIdentifiers=StationIdentifiers,
            KNearest=KNearestNeighbors,
            MaxRadiusKm=SearchRadiusKm,
            DistancePower=PowerExponent,
        )

        SurfaceGridDataFrame = pd.DataFrame({
            "latitude": FlattenedLatitudes,
            "longitude": FlattenedLongitudes,
            "estimated_aqi": np.round(Estimates, 1),
            "confidence_score": Confidences,
            "valid_neighbors": NeighborCounts,
        })
        SurfaceGridDataFrame["aqi_category"] = SurfaceGridDataFrame["estimated_aqi"].apply(GetAqiCategory)
        return SurfaceGridDataFrame

    # Lowercase Aliases For Backward Compatibility
    predict_point = PredictPoint
    predict_grid = PredictGrid


# Backward Compatibility Type Alias
SpatialAQIPredictor = SpatialAqiPredictor
