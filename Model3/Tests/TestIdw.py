"""
Unit Tests For Inverse Distance Weighting (IDW) Spatial Interpolation Engine.
Validates Critical Edge Cases: Co-Location, Coverage Thresholds, And Missing Value Filtering.
"""

import numpy as np
import pytest
from Model3.Src.Idw import IdwInterpolateGrid, IdwInterpolatePoint


def test_idw_exact_match_co_located():
    StationLats = np.array([28.60, 28.70, 28.80])
    StationLons = np.array([77.10, 77.20, 77.30])
    StationValues = np.array([150.0, 220.0, 310.0])
    StationIdentifiers = ["S1", "S2", "S3"]

    # Target Matches Station S2 Exactly
    InterpolationResult = IdwInterpolatePoint(
        TargetLatitude=28.70,
        TargetLongitude=77.20,
        StationLatitudes=StationLats,
        StationLongitudes=StationLons,
        StationValues=StationValues,
        StationIdentifiers=StationIdentifiers,
        KNearest=3,
        MaxRadiusKm=50.0,
        DistancePower=2.0,
    )
    assert InterpolationResult.status == "EXACT_MATCH"
    assert abs(InterpolationResult.estimate - 220.0) < 1e-4
    assert InterpolationResult.nearest_distance_km < 1e-4
    assert InterpolationResult.neighbor_ids == ["S2"]


def test_idw_insufficient_neighbors():
    # Only One Station Inside Search Radius When Minimum Valid Neighbors Is Two
    StationLats = np.array([28.60, 29.50])
    StationLons = np.array([77.10, 78.50])
    StationValues = np.array([120.0, 200.0])

    SparseResult = IdwInterpolatePoint(
        TargetLatitude=28.61,
        TargetLongitude=77.11,
        StationLatitudes=StationLats,
        StationLongitudes=StationLons,
        StationValues=StationValues,
        MaxRadiusKm=10.0,  # Second Station Is More Than One Hundred Kilometers Away
        MinNeighbors=2,
    )
    assert SparseResult.status == "INSUFFICIENT_COVERAGE"
    assert np.isnan(SparseResult.estimate)


def test_idw_nan_filtering():
    # Intermediate Station Has NaN Measurement
    StationLats = np.array([28.60, 28.65, 28.70])
    StationLons = np.array([77.10, 77.15, 77.20])
    StationValues = np.array([100.0, np.nan, 200.0])

    RobustResult = IdwInterpolatePoint(
        TargetLatitude=28.65,
        TargetLongitude=77.15,
        StationLatitudes=StationLats,
        StationLongitudes=StationLons,
        StationValues=StationValues,
        MinNeighbors=2,
        MaxRadiusKm=30.0,
    )
    assert RobustResult.status == "OK"
    assert not np.isnan(RobustResult.estimate)
    assert 100.0 <= RobustResult.estimate <= 200.0
    assert RobustResult.valid_neighbor_count == 2


def test_idw_grid_shape_and_confidence():
    GridLats = np.array([28.60, 28.65])
    GridLons = np.array([77.10, 77.15])
    StationLats = np.array([28.58, 28.62, 28.68])
    StationLons = np.array([77.08, 77.12, 77.18])
    StationVals = np.array([150.0, 160.0, 170.0])

    Estimates, Confidences, ValidCounts = IdwInterpolateGrid(
        GridLatitudes=GridLats,
        GridLongitudes=GridLons,
        StationLatitudes=StationLats,
        StationLongitudes=StationLons,
        StationValues=StationVals,
        KNearest=3,
        MaxRadiusKm=30.0,
    )

    assert len(Estimates) == 2
    assert len(Confidences) == 2
    assert np.all(Confidences >= 0.0)
    assert np.all(Confidences <= 1.0)
    assert np.all(ValidCounts > 0)
