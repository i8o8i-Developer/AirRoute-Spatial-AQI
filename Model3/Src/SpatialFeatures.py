"""
Geospatial Calculations And Spatial Feature Extraction Engine.
Provides Haversine Great-Circle Distance And Forward Azimuth Compass Bearing Formulas,
Along With Vectorized Pairwise Coordinate Distance Matrix Operations.
"""

from typing import Tuple, Union
import numpy as np

# WGS-84 Mean Geodetic Earth Radius In Kilometers
EARTH_RADIUS_KM: float = 6371.0088


def HaversineDistance(
    LatitudeOne: Union[float, np.ndarray],
    LongitudeOne: Union[float, np.ndarray],
    LatitudeTwo: Union[float, np.ndarray],
    LongitudeTwo: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate The Great-Circle Haversine Geodesic Distance In Kilometers.

    Args:
        LatitudeOne: Starting Point Latitude In Degrees
        LongitudeOne: Starting Point Longitude In Degrees
        LatitudeTwo: Target Point Latitude In Degrees
        LongitudeTwo: Target Point Longitude In Degrees

    Returns:
        Great-Circle Geodesic Distance Measured In Kilometers.
    """
    PhiOne = np.radians(LatitudeOne)
    PhiTwo = np.radians(LatitudeTwo)
    DeltaPhi = np.radians(LatitudeTwo - LatitudeOne)
    DeltaLambda = np.radians(LongitudeTwo - LongitudeOne)

    ComponentA = (
        np.sin(DeltaPhi / 2.0) ** 2
        + np.cos(PhiOne) * np.cos(PhiTwo) * np.sin(DeltaLambda / 2.0) ** 2
    )
    # Clip Array To Prevent Floating Point Precision Violations In Arcsin
    ComponentA = np.clip(ComponentA, 0.0, 1.0)
    CentralAngleC = 2.0 * np.arctan2(np.sqrt(ComponentA), np.sqrt(1.0 - ComponentA))

    CalculatedDistance = EARTH_RADIUS_KM * CentralAngleC
    return float(CalculatedDistance) if np.isscalar(CalculatedDistance) else CalculatedDistance


def ForwardBearing(
    LatitudeOne: Union[float, np.ndarray],
    LongitudeOne: Union[float, np.ndarray],
    LatitudeTwo: Union[float, np.ndarray],
    LongitudeTwo: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate The Initial Forward Azimuth Compass Bearing In Angular Degrees.

    Args:
        LatitudeOne, LongitudeOne: Source Coordinate Point Coordinates In Degrees
        LatitudeTwo, LongitudeTwo: Destination Coordinate Point Coordinates In Degrees

    Returns:
        Compass Azimuth Bearing In Degrees Normalized Within 0 To 360 Degrees.
    """
    PhiOne = np.radians(LatitudeOne)
    PhiTwo = np.radians(LatitudeTwo)
    DeltaLambda = np.radians(LongitudeTwo - LongitudeOne)

    CoordinateY = np.sin(DeltaLambda) * np.cos(PhiTwo)
    CoordinateX = np.cos(PhiOne) * np.sin(PhiTwo) - np.sin(PhiOne) * np.cos(PhiTwo) * np.cos(DeltaLambda)

    InitialAzimuth = np.degrees(np.arctan2(CoordinateY, CoordinateX))
    NormalizedCompassBearing = (InitialAzimuth + 360.0) % 360.0
    return float(NormalizedCompassBearing) if np.isscalar(NormalizedCompassBearing) else NormalizedCompassBearing


def ComputePairwiseDistances(
    TargetLatitudes: np.ndarray,
    TargetLongitudes: np.ndarray,
    StationLatitudes: np.ndarray,
    StationLongitudes: np.ndarray,
) -> np.ndarray:
    """
    Compute Full Pairwise Geodesic Distance Matrix Between M Targets And N Stations.

    Args:
        TargetLatitudes, TargetLongitudes: Array Shape (M,)
        StationLatitudes, StationLongitudes: Array Shape (N,)

    Returns:
        Pairwise Geodetic Distance Matrix Array Shape (M, N) In Kilometers.
    """
    ExpandedTargetLat = TargetLatitudes[:, np.newaxis]
    ExpandedTargetLon = TargetLongitudes[:, np.newaxis]
    ExpandedStationLat = StationLatitudes[np.newaxis, :]
    ExpandedStationLon = StationLongitudes[np.newaxis, :]

    return HaversineDistance(ExpandedTargetLat, ExpandedTargetLon, ExpandedStationLat, ExpandedStationLon)


# Lowercase Aliases For Backward Compatibility
haversine_distance = HaversineDistance
forward_bearing = ForwardBearing
compute_pairwise_distances = ComputePairwiseDistances
