"""
Inverse Distance Weighting (IDW) Spatial Interpolation Engine.
Includes Robust Handling Of Edge Cases (Co-Located Points, Insufficient Station Coverage, NaNs)
And Extracts Rich Neighbor Diagnostic Metadata For The Stage 2 XGBoost Residual Layer.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from Model3.Src.SpatialFeatures import HaversineDistance


@dataclass
class IdwResult:
    """
    Structured Container For Spatial IDW Estimation And Associated Neighbor Diagnostics.
    """
    estimate: float
    status: str  # 'OK', 'EXACT_MATCH', 'INSUFFICIENT_COVERAGE'
    valid_neighbor_count: int
    nearest_distance_km: float
    mean_distance_km: float
    max_distance_km: float
    neighbor_min: float
    neighbor_mean: float
    neighbor_max: float
    neighbor_std: float
    neighbor_ids: List[str]

    def ToDict(self) -> Dict[str, Union[float, int, str]]:
        """
        Convert Result Object Into Dictionary For Parquet Serialization.
        """
        return {
            "idw_estimate": self.estimate,
            "idw_status": self.status,
            "valid_neighbor_count": self.valid_neighbor_count,
            "nearest_distance_km": self.nearest_distance_km,
            "mean_distance_km": self.mean_distance_km,
            "max_distance_km": self.max_distance_km,
            "neighbor_min": self.neighbor_min,
            "neighbor_mean": self.neighbor_mean,
            "neighbor_max": self.neighbor_max,
            "neighbor_std": self.neighbor_std,
        }

    to_dict = ToDict


# Backward Compatibility Type Alias
IDWResult = IdwResult


def IdwInterpolatePoint(
    TargetLatitude: float,
    TargetLongitude: float,
    StationLatitudes: np.ndarray,
    StationLongitudes: np.ndarray,
    StationValues: np.ndarray,
    StationIdentifiers: Optional[List[str]] = None,
    KNearest: int = 8,
    MaxRadiusKm: float = 25.0,
    DistancePower: float = 1.0,
    MinNeighbors: int = 2,
    EpsilonDistance: float = 1e-4,  # Within 10 Centimeters Treated As Co-Located Exact Match
) -> IdwResult:
    """
    Perform Point-Level Inverse Distance Weighting Interpolation For A Target Coordinate.

    Args:
        TargetLatitude, TargetLongitude: Unmonitored Target Coordinate In Degrees
        StationLatitudes, StationLongitudes: Array Of Available Station Coordinates
        StationValues: Scalar Observed Measurements (AQI Points Or Pollutant Concentrations)
        StationIdentifiers: Optional List Of Monitored Station Identification Strings
        KNearest: Number Of Closest Monitored Neighbors Selected
        MaxRadiusKm: Maximum Geographic Search Radius In Kilometers
        DistancePower: Distance-Decay Power Exponent P (Default 1.0 Per Tuning)
        MinNeighbors: Minimum Number Of Valid Stations Required In Radius
        EpsilonDistance: Distance Cutoff For Exact Co-Location

    Returns:
        IdwResult Object Containing Value Estimate And Neighbor Spatial Diagnostics.
    """
    StationLatsArray = np.asarray(StationLatitudes, dtype=float)
    StationLonsArray = np.asarray(StationLongitudes, dtype=float)
    StationValsArray = np.asarray(StationValues, dtype=float)

    if StationIdentifiers is None:
        StationIdentifiers = [str(i) for i in range(len(StationValsArray))]

    # Exclude Missing And NaN Sensor Readings
    ValidMask = ~np.isnan(StationValsArray) & ~np.isnan(StationLatsArray) & ~np.isnan(StationLonsArray)
    if not np.any(ValidMask):
        return IdwResult(
            estimate=np.nan,
            status="INSUFFICIENT_COVERAGE",
            valid_neighbor_count=0,
            nearest_distance_km=np.nan,
            mean_distance_km=np.nan,
            max_distance_km=np.nan,
            neighbor_min=np.nan,
            neighbor_mean=np.nan,
            neighbor_max=np.nan,
            neighbor_std=np.nan,
            neighbor_ids=[],
        )

    FilteredLats = StationLatsArray[ValidMask]
    FilteredLons = StationLonsArray[ValidMask]
    FilteredVals = StationValsArray[ValidMask]
    FilteredIds = [Sid for Sid, MaskBit in zip(StationIdentifiers, ValidMask) if MaskBit]

    # Calculate Geodesic Great-Circle Distances
    CalculatedDistances = HaversineDistance(TargetLatitude, TargetLongitude, FilteredLats, FilteredLons)

    # Edge Case: Co-Located Point Matches Physical Station Coordinate
    CoLocatedIndices = np.where(CalculatedDistances < EpsilonDistance)[0]
    if len(CoLocatedIndices) > 0:
        MatchIndex = CoLocatedIndices[0]
        return IdwResult(
            estimate=float(FilteredVals[MatchIndex]),
            status="EXACT_MATCH",
            valid_neighbor_count=1,
            nearest_distance_km=float(CalculatedDistances[MatchIndex]),
            mean_distance_km=float(CalculatedDistances[MatchIndex]),
            max_distance_km=float(CalculatedDistances[MatchIndex]),
            neighbor_min=float(FilteredVals[MatchIndex]),
            neighbor_mean=float(FilteredVals[MatchIndex]),
            neighbor_max=float(FilteredVals[MatchIndex]),
            neighbor_std=0.0,
            neighbor_ids=[FilteredIds[MatchIndex]],
        )

    # Filter Neighbors Within Configured Search Radius
    RadiusFilterMask = CalculatedDistances <= MaxRadiusKm
    if np.sum(RadiusFilterMask) < MinNeighbors:
        return IdwResult(
            estimate=np.nan,
            status="INSUFFICIENT_COVERAGE",
            valid_neighbor_count=int(np.sum(RadiusFilterMask)),
            nearest_distance_km=float(np.min(CalculatedDistances)) if len(CalculatedDistances) > 0 else np.nan,
            mean_distance_km=np.nan,
            max_distance_km=np.nan,
            neighbor_min=np.nan,
            neighbor_mean=np.nan,
            neighbor_max=np.nan,
            neighbor_std=np.nan,
            neighbor_ids=[],
        )

    RadiusDistances = CalculatedDistances[RadiusFilterMask]
    RadiusValues = FilteredVals[RadiusFilterMask]
    RadiusIds = [FilteredIds[i] for i, MaskBit in enumerate(RadiusFilterMask) if MaskBit]

    # Sort By Ascending Distance And Pick Up To K Neighbors
    SortedIndices = np.argsort(RadiusDistances)[:KNearest]
    KDistances = RadiusDistances[SortedIndices]
    KValues = RadiusValues[SortedIndices]
    KIds = [RadiusIds[i] for i in SortedIndices]

    # Inverse Distance Weighting Formulation: W_i = 1 / (D_i ^ P)
    InverseWeights = 1.0 / (KDistances ** DistancePower)
    SumOfWeights = np.sum(InverseWeights)

    if SumOfWeights <= 0:
        CalculatedEstimate = float(np.mean(KValues))
    else:
        CalculatedEstimate = float(np.sum(InverseWeights * KValues) / SumOfWeights)

    return IdwResult(
        estimate=CalculatedEstimate,
        status="OK",
        valid_neighbor_count=len(KValues),
        nearest_distance_km=float(KDistances[0]),
        mean_distance_km=float(np.mean(KDistances)),
        max_distance_km=float(KDistances[-1]),
        neighbor_min=float(np.min(KValues)),
        neighbor_mean=float(np.mean(KValues)),
        neighbor_max=float(np.max(KValues)),
        neighbor_std=float(np.std(KValues)) if len(KValues) > 1 else 0.0,
        neighbor_ids=KIds,
    )


def IdwInterpolateGrid(
    GridLatitudes: np.ndarray,
    GridLongitudes: np.ndarray,
    StationLatitudes: np.ndarray,
    StationLongitudes: np.ndarray,
    StationValues: np.ndarray,
    StationIdentifiers: Optional[List[str]] = None,
    KNearest: int = 8,
    MaxRadiusKm: float = 25.0,
    DistancePower: float = 1.0,
    MinNeighbors: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Batch Spatial IDW Estimation Across A 2D Continuous Grid.

    Returns:
        Tuple Of (Estimates Array, Confidence Array, Valid Neighbors Count Array).
    """
    NumberOfPoints = len(GridLatitudes)
    EstimatesArray = np.full(NumberOfPoints, np.nan, dtype=float)
    ConfidenceArray = np.full(NumberOfPoints, 0.0, dtype=float)
    ValidCountsArray = np.zeros(NumberOfPoints, dtype=int)

    for Index in range(NumberOfPoints):
        PointResult = IdwInterpolatePoint(
            TargetLatitude=GridLatitudes[Index],
            TargetLongitude=GridLongitudes[Index],
            StationLatitudes=StationLatitudes,
            StationLongitudes=StationLongitudes,
            StationValues=StationValues,
            StationIdentifiers=StationIdentifiers,
            KNearest=KNearest,
            MaxRadiusKm=MaxRadiusKm,
            DistancePower=DistancePower,
            MinNeighbors=MinNeighbors,
        )
        if PointResult.status in ("OK", "EXACT_MATCH"):
            EstimatesArray[Index] = PointResult.estimate
            ValidCountsArray[Index] = PointResult.valid_neighbor_count
            DistanceFactor = np.clip(1.0 - (PointResult.nearest_distance_km / MaxRadiusKm), 0.0, 1.0)
            DensityFactor = np.clip(PointResult.valid_neighbor_count / KNearest, 0.0, 1.0)
            ConfidenceArray[Index] = round(0.6 * DistanceFactor + 0.4 * DensityFactor, 3)

    return EstimatesArray, ConfidenceArray, ValidCountsArray


# Backward Compatibility Aliases
idw_interpolate_point = IdwInterpolatePoint
idw_interpolate_grid = IdwInterpolateGrid
