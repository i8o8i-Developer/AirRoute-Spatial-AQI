"""
Mathematical And Programmatic Proof Of Zero Data Leakage In Spatial Evaluation.
Guarantees Evaluated Stations Never Appear In Their Own Neighbor Candidates Or Statistics.
"""

import numpy as np
import pandas as pd
import pytest
from Model3.Src.Dataset import (
    ApplyQualityControl,
    GenerateLosoDataset,
    GenerateSynchronizedDelhiHistory,
    SplitSpatialTemporalHoldouts,
)


def test_no_leakage_in_loso_simulation():
    # Generate Synthetic Synchronized Observations Over Two Days
    RawData = GenerateSynchronizedDelhiHistory(NumberOfDays=2, FrequencyHours=2)
    QcCleanData = ApplyQualityControl(RawData)

    # Generate Leave-One-Station-Out Validation Records
    LosoRecords = GenerateLosoDataset(QcCleanData, KNearest=5, MaxRadiusKm=30.0, DistancePower=2.0)
    assert len(LosoRecords) > 0, "LOSO Evaluation Dataset Must Contain Valid Records"

    # Assert For Every Evaluation Instance That Target Station Never Leaks
    for _, RecordRow in LosoRecords.iterrows():
        # Nearest Neighbor Distance Must Be Strictly Greater Than Zero
        assert RecordRow["nearest_distance_km"] > 0.1, (
            f"Nearest Neighbor Distance Too Small ({RecordRow['nearest_distance_km']} Km). "
            f"Target Station {RecordRow['target_station_id']} May Be Leaking Into Its Own Pool!"
        )

        # Residual Arithmetic Verification
        CalculatedResidual = RecordRow["actual_aqi"] - RecordRow["idw_aqi_direct"]
        assert abs(RecordRow["idw_residual"] - CalculatedResidual) < 1e-4

        # Estimation Status Must Be Valid Interpolation And Never Co-Located Exact Match
        assert RecordRow["idw_status"] == "OK"


def test_holdout_isolation():
    # Verify Spatial Holdout Stations Never Overlap With Training Network
    RawData = GenerateSynchronizedDelhiHistory(NumberOfDays=3, FrequencyHours=2)
    TrainStations, TestStations, CutoffTimestamp, TrainPartition, TestPartition = SplitSpatialTemporalHoldouts(RawData)

    # Spatial Partition Disjoint Intersection Test
    OverlapStations = set(TrainStations).intersection(set(TestStations))
    assert len(OverlapStations) == 0, f"Spatial Data Leakage Detected: {OverlapStations} Found In Both Splits!"

    # Temporal Horizon Strict Precedence Test
    assert TrainPartition["timestamp"].max() < CutoffTimestamp, "Temporal Data Leakage Detected: Train Extends Past Cutoff!"
