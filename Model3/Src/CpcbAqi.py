"""
Indian Central Pollution Control Board (CPCB) NAQI Calculation Engine.
Implements Piecewise Linear Sub-Index Formulas And Official Multi-Pollutant Rules.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from Model3.Src.Config import AQI_CATEGORIES, CPCB_BREAKPOINTS


def CalculateSubIndex(Pollutant: str, Concentration: float) -> float:
    """
    Calculate The Numerical Sub-Index For A Pollutant Using CPCB Breakpoints.

    Args:
        Pollutant: Standard Chemical Name ('pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3')
        Concentration: Measured Ambient Pollutant Concentration

    Returns:
        Interpolated Sub-Index Scalar Between 0.0 And 500.0, Or NaN If Invalid.
    """
    if Concentration is None or np.isnan(Concentration) or Concentration < 0:
        return np.nan

    PollutantKey = Pollutant.lower().replace(".", "_").replace("-", "_")
    if PollutantKey not in CPCB_BREAKPOINTS:
        raise ValueError(f"Unknown Pollutant Requested: {Pollutant}. Supported Pollutants: {list(CPCB_BREAKPOINTS.keys())}")

    Breakpoints = CPCB_BREAKPOINTS[PollutantKey]

    # Upper Saturation Limit Capped At 500
    MaxUpperLimit = Breakpoints[-1][1]
    if Concentration >= MaxUpperLimit:
        return 500.0

    # Piecewise Linear Interpolation
    for BreakpointLow, BreakpointHigh, IndexLow, IndexHigh in Breakpoints:
        if BreakpointLow <= Concentration <= BreakpointHigh:
            SubIndexValue = IndexLow + ((IndexHigh - IndexLow) / (BreakpointHigh - BreakpointLow)) * (Concentration - BreakpointLow)
            return float(np.clip(SubIndexValue, 0.0, 500.0))

    # Boundary Tolerance Fallback
    return float(np.clip(Breakpoints[-1][3], 0.0, 500.0))


def CalculateCpcbAqi(
    Pollutants: Dict[str, float],
    StrictCpcbRules: bool = True,
) -> Tuple[float, Optional[str], Dict[str, float]]:
    """
    Calculate The Overall CPCB NAQI Given A Dictionary Of Pollutant Values.

    Official CPCB Minimum Criteria:
    - Minimum Three Pollutants Must Be Monitored Simultaneously.
    - At Least One Primary Particulate (PM2.5 Or PM10) Must Be Present.
    - Overall AQI Equals The Maximum Value Among Available Sub-Indices.

    Args:
        Pollutants: Key-Value Dictionary Mapping Pollutant Keys To Observed Concentrations.
        StrictCpcbRules: When True, Strictly Enforces The Three-Pollutant Minimum Rule.

    Returns:
        Tuple Containing (Overall AQI, Prominent Pollutant Name, Sub-Indices Dictionary).
    """
    SubIndicesDict: Dict[str, float] = {}

    for PolName, PolValue in Pollutants.items():
        PolKey = PolName.lower().replace(".", "_").replace("-", "_")
        if PolKey in CPCB_BREAKPOINTS and PolValue is not None and not np.isnan(PolValue):
            ComputedIndex = CalculateSubIndex(PolKey, PolValue)
            if not np.isnan(ComputedIndex):
                SubIndicesDict[PolKey] = ComputedIndex

    ValidCount = len(SubIndicesDict)
    HasParticulate = ("pm2_5" in SubIndicesDict) or ("pm10" in SubIndicesDict)

    if StrictCpcbRules:
        if ValidCount < 3 or not HasParticulate:
            return np.nan, None, SubIndicesDict

    if not SubIndicesDict:
        return np.nan, None, {}

    ProminentPollutant = max(SubIndicesDict, key=SubIndicesDict.get)
    OverallAqi = round(SubIndicesDict[ProminentPollutant], 1)
    return OverallAqi, ProminentPollutant, SubIndicesDict


def GetAqiCategory(AqiValue: float) -> str:
    """
    Map Numerical AQI Value To Standard CPCB Qualitative Health Descriptor.
    """
    if AqiValue is None or np.isnan(AqiValue):
        return "Unknown"

    ClampedValue = np.clip(AqiValue, 0.0, 500.0)
    for LowerBound, UpperBound, CategoryLabel in AQI_CATEGORIES:
        if LowerBound <= round(ClampedValue) <= UpperBound:
            return CategoryLabel
    return "Severe" if ClampedValue > 400 else "Good"


def ComputeAqiFromSeries(RowSeries: Union[pd.Series, dict], StrictFlag: bool = False) -> float:
    """
    Convenience Helper For Applying CPCB NAQI Estimation Across Pandas Rows.
    """
    PollutantsDict = {
        KeyName: RowSeries.get(KeyName, np.nan)
        for KeyName in ("pm2_5", "pm10", "no2", "so2", "co", "o3")
        if KeyName in RowSeries
    }
    ComputedAqi, _, _ = CalculateCpcbAqi(PollutantsDict, StrictCpcbRules=StrictFlag)
    return ComputedAqi
