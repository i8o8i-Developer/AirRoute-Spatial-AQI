"""
Unit Tests For Geodetic Haversine Distance And Compass Azimuth Calculations.
"""

import numpy as np
import pytest
from Model3.Src.SpatialFeatures import (
    ComputePairwiseDistances,
    ForwardBearing,
    HaversineDistance,
)


def test_haversine_same_point():
    # Co-Located Identical Point Yields Zero Geodesic Distance
    LatitudePoint, LongitudePoint = 28.6139, 77.2090
    CalculatedDistance = HaversineDistance(LatitudePoint, LongitudePoint, LatitudePoint, LongitudePoint)
    assert abs(CalculatedDistance) < 1e-6


def test_haversine_known_delhi_pairs():
    # Benchmark Geodetic Check: Connaught Place To Noida Sector 62 (Expected ~14.5 To 15.5 Km)
    CpLatitude, CpLongitude = 28.6315, 77.2167
    NoidaLatitude, NoidaLongitude = 28.6245, 77.3648
    DistanceToNoida = HaversineDistance(CpLatitude, CpLongitude, NoidaLatitude, NoidaLongitude)
    assert 14.0 <= DistanceToNoida <= 15.5

    # Benchmark Geodetic Check: Connaught Place To Gurugram Cyber Hub (Expected ~19.5 To 21.5 Km)
    GurugramLatitude, GurugramLongitude = 28.4950, 77.0895
    DistanceToGurugram = HaversineDistance(CpLatitude, CpLongitude, GurugramLatitude, GurugramLongitude)
    assert 19.0 <= DistanceToGurugram <= 21.5


def test_forward_bearing_cardinal_directions():
    BaseLatitude, BaseLongitude = 28.0, 77.0

    # Due North Azimuth Check
    BearingNorth = ForwardBearing(BaseLatitude, BaseLongitude, BaseLatitude + 1.0, BaseLongitude)
    assert abs(BearingNorth - 0.0) < 0.1 or abs(BearingNorth - 360.0) < 0.1

    # Due East Azimuth Check
    BearingEast = ForwardBearing(BaseLatitude, BaseLongitude, BaseLatitude, BaseLongitude + 1.0)
    assert abs(BearingEast - 90.0) < 1.0

    # Due South Azimuth Check
    BearingSouth = ForwardBearing(BaseLatitude, BaseLongitude, BaseLatitude - 1.0, BaseLongitude)
    assert abs(BearingSouth - 180.0) < 0.1

    # Due West Azimuth Check
    BearingWest = ForwardBearing(BaseLatitude, BaseLongitude, BaseLatitude, BaseLongitude - 1.0)
    assert abs(BearingWest - 270.0) < 1.0


def test_pairwise_distance_matrix_shape():
    TargetLatitudes = np.array([28.5, 28.6, 28.7])
    TargetLongitudes = np.array([77.1, 77.2, 77.3])
    StationLatitudes = np.array([28.4, 28.5, 28.6, 28.7, 28.8])
    StationLongitudes = np.array([77.0, 77.1, 77.2, 77.3, 77.4])

    DistanceMatrix = ComputePairwiseDistances(TargetLatitudes, TargetLongitudes, StationLatitudes, StationLongitudes)
    assert DistanceMatrix.shape == (3, 5)
    assert np.all(DistanceMatrix >= 0.0)
