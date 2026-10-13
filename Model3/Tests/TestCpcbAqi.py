"""
Unit Tests For Official Indian CPCB NAQI Calculation Engine And Breakpoints.
"""

import numpy as np
import pytest
from Model3.Src.CpcbAqi import (
    CalculateCpcbAqi,
    CalculateSubIndex,
    GetAqiCategory,
)


def test_sub_index_pm25_breakpoints():
    # PM2.5 Good Interval Check: [0, 30] To [0, 50]
    IndexAtFifteen = CalculateSubIndex("pm2_5", 15.0)
    assert abs(IndexAtFifteen - 25.0) < 0.5

    # PM2.5 Satisfactory Interval Check: [30.1, 60.0] To [51, 100]
    IndexAtFortyFive = CalculateSubIndex("pm2_5", 45.0)
    assert 70.0 <= IndexAtFortyFive <= 80.0

    # PM2.5 Poor Interval Check: [90.1, 120.0] To [201, 300]
    IndexAtOneHundredFive = CalculateSubIndex("pm2_5", 105.0)
    assert 240.0 <= IndexAtOneHundredFive <= 260.0

    # Negative Outlier Input Must Return NaN
    assert np.isnan(CalculateSubIndex("pm2_5", -5.0))

    # Extreme Value Exceeding Max Breakpoint Capped At Five Hundred
    assert CalculateSubIndex("pm2_5", 800.0) == 500.0


def test_overall_cpcb_aqi_aggregation():
    # Valid Multi-Pollutant Profile Check
    SamplePollutants = {
        "pm2_5": 110.0,  # Sub-Index Approx 267 (Poor Category)
        "pm10": 150.0,   # Sub-Index Approx 133 (Moderate Category)
        "no2": 50.0,     # Sub-Index Approx 63 (Satisfactory Category)
    }
    CalculatedAqi, ProminentPollutant, SubIndicesDict = CalculateCpcbAqi(SamplePollutants, StrictCpcbRules=True)
    assert not np.isnan(CalculatedAqi)
    assert ProminentPollutant == "pm2_5"
    assert 250.0 <= CalculatedAqi <= 280.0


def test_strict_cpcb_rules_enforcement():
    # Rule Failure Test: Only Two Pollutants Available
    TwoPollutantsProfile = {"pm2_5": 50.0, "no2": 30.0}
    CalculatedAqi, _, _ = CalculateCpcbAqi(TwoPollutantsProfile, StrictCpcbRules=True)
    assert np.isnan(CalculatedAqi)

    # Rule Failure Test: Three Pollutants Present But Missing Both PM2.5 And PM10
    NoParticulatesProfile = {"no2": 30.0, "so2": 15.0, "co": 1.2}
    AqiWithoutParticulates, _, _ = CalculateCpcbAqi(NoParticulatesProfile, StrictCpcbRules=True)
    assert np.isnan(AqiWithoutParticulates)


def test_category_labels():
    assert GetAqiCategory(35.0) == "Good"
    assert GetAqiCategory(75.0) == "Satisfactory"
    assert GetAqiCategory(150.0) == "Moderate"
    assert GetAqiCategory(250.0) == "Poor"
    assert GetAqiCategory(350.0) == "Very Poor"
    assert GetAqiCategory(450.0) == "Severe"
