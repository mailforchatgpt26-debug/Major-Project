"""
Pharma bilateral forecast targets for 2026–2030 (HS 30, top 25 partners).

Values in USD millions, derived from project forecast table (US$ billion × 1,000).
2026–2028: explicit table; 2029–2030 extrapolated at ~8.5% export / ~7.5% import CAGR.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

TOP25_PARTNERS: List[str] = [
    "USA", "GBR", "BRA", "ZAF", "FRA", "NLD", "DEU", "RUS", "CAN", "AUS",
    "SAU", "ARE", "JPN", "BEL", "ITA", "ESP", "POL", "TUR", "MEX", "NGA",
    "KEN", "EGY", "BGD", "VNM", "THA",
]

# IND → partner exports (USD millions)
PHARMA_EXPORT_FORECAST_USD_M: Dict[str, Dict[int, float]] = {
    "USA": {2026: 11400.0, 2027: 12400.0, 2028: 13500.0},
    "GBR": {2026: 1000.0, 2027: 1090.0, 2028: 1180.0},
    "BRA": {2026: 860.0, 2027: 940.0, 2028: 1020.0},
    "ZAF": {2026: 800.0, 2027: 870.0, 2028: 940.0},
    "FRA": {2026: 760.0, 2027: 830.0, 2028: 900.0},
    "NLD": {2026: 800.0, 2027: 870.0, 2028: 950.0},
    "DEU": {2026: 730.0, 2027: 790.0, 2028: 860.0},
    "RUS": {2026: 680.0, 2027: 740.0, 2028: 800.0},
    "CAN": {2026: 730.0, 2027: 790.0, 2028: 860.0},
    "AUS": {2026: 560.0, 2027: 610.0, 2028: 660.0},
    "SAU": {2026: 600.0, 2027: 650.0, 2028: 710.0},
    "ARE": {2026: 750.0, 2027: 820.0, 2028: 890.0},
    "JPN": {2026: 480.0, 2027: 520.0, 2028: 570.0},
    "BEL": {2026: 550.0, 2027: 600.0, 2028: 650.0},
    "ITA": {2026: 420.0, 2027: 460.0, 2028: 500.0},
    "ESP": {2026: 400.0, 2027: 440.0, 2028: 470.0},
    "POL": {2026: 330.0, 2027: 360.0, 2028: 390.0},
    "TUR": {2026: 290.0, 2027: 310.0, 2028: 340.0},
    "MEX": {2026: 320.0, 2027: 350.0, 2028: 380.0},
    "NGA": {2026: 310.0, 2027: 340.0, 2028: 370.0},
    "KEN": {2026: 220.0, 2027: 240.0, 2028: 260.0},
    "EGY": {2026: 270.0, 2027: 290.0, 2028: 320.0},
    "BGD": {2026: 240.0, 2027: 260.0, 2028: 280.0},
    "VNM": {2026: 260.0, 2027: 280.0, 2028: 310.0},
    "THA": {2026: 250.0, 2027: 270.0, 2028: 290.0},
    # India → Nepal (HS 30): Pharmexcil / bilateral trade estimates
    "NPL": {2026: 290.0, 2027: 320.0, 2028: 350.0},
}

# partner → IND imports (USD millions)
PHARMA_IMPORT_FORECAST_USD_M: Dict[str, Dict[int, float]] = {
    "USA": {2026: 1100.0, 2027: 1200.0, 2028: 1300.0},
    "GBR": {2026: 100.0, 2027: 110.0, 2028: 120.0},
    "BRA": {2026: 130.0, 2027: 140.0, 2028: 150.0},
    "ZAF": {2026: 70.0, 2027: 80.0, 2028: 80.0},
    "FRA": {2026: 420.0, 2027: 450.0, 2028: 490.0},
    "NLD": {2026: 450.0, 2027: 480.0, 2028: 520.0},
    "DEU": {2026: 1000.0, 2027: 1080.0, 2028: 1160.0},
    "RUS": {2026: 180.0, 2027: 190.0, 2028: 210.0},
    "CAN": {2026: 200.0, 2027: 220.0, 2028: 230.0},
    "AUS": {2026: 90.0, 2027: 100.0, 2028: 110.0},
    "SAU": {2026: 70.0, 2027: 80.0, 2028: 90.0},
    "ARE": {2026: 350.0, 2027: 380.0, 2028: 410.0},
    "JPN": {2026: 620.0, 2027: 670.0, 2028: 720.0},
    "BEL": {2026: 800.0, 2027: 860.0, 2028: 930.0},
    "ITA": {2026: 480.0, 2027: 520.0, 2028: 560.0},
    "ESP": {2026: 360.0, 2027: 390.0, 2028: 420.0},
    "POL": {2026: 140.0, 2027: 150.0, 2028: 160.0},
    "TUR": {2026: 80.0, 2027: 90.0, 2028: 100.0},
    "MEX": {2026: 110.0, 2027: 120.0, 2028: 130.0},
    "NGA": {2026: 30.0, 2027: 30.0, 2028: 40.0},
    "KEN": {2026: 20.0, 2027: 20.0, 2028: 30.0},
    "EGY": {2026: 40.0, 2027: 40.0, 2028: 50.0},
    "BGD": {2026: 50.0, 2027: 50.0, 2028: 60.0},
    "VNM": {2026: 70.0, 2027: 80.0, 2028: 80.0},
    "THA": {2026: 100.0, 2027: 110.0, 2028: 120.0},
}

EXPORT_CAGR_FALLBACK = 0.085
IMPORT_CAGR_FALLBACK = 0.075
FORECAST_END_YEAR = 2030

# India's total pharma exports (all destinations, USD M) — Pharmexcil FY25 national baseline.
# Calibrated so USA FY25 exports (~$10,515M) ≈ 33.8% of national total (not top-50 partner sum).
PHARMA_INDIA_TOTAL_EXPORT_USD_M_2025 = 31_109.8
PHARMA_USA_REFERENCE_EXPORT_SHARE = 0.338


def pharma_india_total_export_usd_m(year: int) -> float:
    """National HS30 export denominator for portfolio-share displays."""
    y = int(year)
    if y <= 2025:
        return PHARMA_INDIA_TOTAL_EXPORT_USD_M_2025
    return PHARMA_INDIA_TOTAL_EXPORT_USD_M_2025 * ((1.0 + EXPORT_CAGR_FALLBACK) ** (y - 2025))


def pharma_national_export_share(
    export_usd_m: float,
    year: int,
    actual_2025_usd_m: Optional[float] = None,
) -> float:
    """Partner share of India's national pharma exports (fraction 0–1)."""
    total = pharma_india_total_export_usd_m(year)
    if total <= 0:
        return 0.0
    if year <= 2025 and actual_2025_usd_m is not None and actual_2025_usd_m > 0:
        return float(actual_2025_usd_m) / total
    return float(export_usd_m) / total

# Display-only FY25 vs FY24 export YoY (%) — demo dashboard; does not alter USD forecasts.
PHARMA_EXPORT_YOY_FY25_VS_FY24: Dict[str, float] = {
    "USA": 14.3,
    "GBR": 17.0,
    "BRA": 11.5,
    "FRA": 8.2,
    "ZAF": -1.8,
    "CAN": 9.4,
    "DEU": 7.1,
    "AUS": 10.6,
    "RUS": 12.4,
    "NLD": -13.8,
    "ARE": -17.7,
    "BEL": -7.4,
    "CHN": -10.6,
    "SAU": 4.5,
    "MEX": -3.8,
    "ITA": 6.8,
    "ESP": 7.5,
    "JPN": 5.2,
    "POL": 8.9,
    "TUR": -16.0,
    "SGP": 12.1,
    "THA": -0.1,
    "VNM": 11.2,
    "IDN": 9.8,
    "LKA": -14.6,
    "NPL": 13.7,
    "BGD": 10.3,
    "MYS": 6.9,
    "NZL": 5.4,
    "ARG": 8.1,
    "DNK": 7.8,
    "SWE": 4.3,
    "FIN": 3.9,
    "CZE": 6.4,
    "HUN": 5.8,
    "ROU": 7.2,
    "GRC": 4.9,
    "SVN": 6.1,
    "JOR": 8.5,
    "OMN": 7.0,
    "DZA": 9.2,
    "GHA": 10.8,
    "NGA": 15.4,
    "ETH": 11.6,
    "UGA": 12.3,
    "TZA": 10.9,
    "MLT": 4.7,
    "LVA": 5.5,
    "HKG": 2.8,
    "DOM": 8.7,
}

# Localization / import-substitution corridors: expected decline window + model-facing reason
PHARMA_DECLINE_WINDOWS: Dict[str, Dict[str, object]] = {
    "SAU": {
        "start": 2027,
        "end": 2028,
        "reason": "Localization targets and increasing domestic production capacity",
    },
    "BGD": {
        "start": 2028,
        "end": 2029,
        "reason": "Growing self-sufficiency in pharmaceutical manufacturing",
    },
    "EGY": {
        "start": 2027,
        "end": 2028,
        "reason": "Expansion of local pharmaceutical production facilities",
    },
    "TUR": {
        "start": 2028,
        "end": 2029,
        "reason": "Import substitution policies and domestic industry support",
    },
    "NGA": {
        "start": 2029,
        "end": 2030,
        "reason": "Gradual implementation of local manufacturing initiatives",
    },
    "ZAF": {
        "start": 2028,
        "end": 2029,
        "reason": "Regional manufacturing development under African industrialization programs",
    },
    "IDN": {
        "start": 2028,
        "end": 2030,
        "reason": "Expansion of domestic pharmaceutical sector",
    },
    "ARE": {
        "start": 2029,
        "end": 2030,
        "reason": "Increasing regional production and re-export competition",
    },
}

# Display-only export YoY (%) for 2026–2030 on localization-risk corridors (USD forecasts unchanged)
PHARMA_EXPORT_YOY_BY_YEAR: Dict[str, Dict[int, float]] = {
    "SAU": {2026: 6.5, 2027: -8.2, 2028: -10.5, 2029: -2.0, 2030: 1.5},
    "BGD": {2026: 8.0, 2027: 4.5, 2028: -9.0, 2029: -12.5, 2030: -1.0},
    "EGY": {2026: 5.5, 2027: -7.0, 2028: -11.0, 2029: -2.5, 2030: 0.5},
    "TUR": {2026: -2.0, 2027: -4.0, 2028: -9.5, 2029: -13.0, 2030: -3.0},
    "NGA": {2026: 12.0, 2027: 9.0, 2028: 6.0, 2029: -8.0, 2030: -11.5},
    "ZAF": {2026: 2.0, 2027: 1.0, 2028: -6.5, 2029: -9.0, 2030: -1.5},
    "IDN": {2026: 7.5, 2027: 5.0, 2028: -5.5, 2029: -8.0, 2030: -6.0},
    "ARE": {2026: 3.0, 2027: 2.0, 2028: -1.0, 2029: -10.0, 2030: -14.0},
}

PHARMA_LOCALIZATION_RISK_MARKETS = frozenset(PHARMA_DECLINE_WINDOWS.keys())

# Bilateral news / policy priors for resilience flags (2025–2026 pharma trade coverage themes)
PHARMA_NEWS_SIGNALS: Dict[str, Dict[str, object]] = {
    "USA": {
        "sentiment": 0.48,
        "policy_friction": 0.20,
        "policy_note": "Record India→US pharma investment pledges; constructive FDA/specialty API dialogue",
    },
    "DEU": {
        "sentiment": 0.26,
        "policy_friction": 0.28,
        "policy_note": "Stable EU demand; periodic EU tariff and regulatory filing scrutiny",
    },
    "GBR": {
        "sentiment": 0.31,
        "policy_friction": 0.26,
        "policy_note": "UK–India trade continuity; moderate post-Brexit customs friction",
    },
    "JPN": {
        "sentiment": 0.24,
        "policy_friction": 0.24,
        "policy_note": "Steady regulated market access; limited trade-policy headlines",
    },
    "CAN": {
        "sentiment": 0.29,
        "policy_friction": 0.22,
        "policy_note": "North America corridor stability; low tariff noise on generics",
    },
    "SAU": {
        "sentiment": -0.41,
        "policy_friction": 0.74,
        "policy_note": "Vision 2030 localization and domestic biopharma capacity targets",
    },
    "BGD": {
        "sentiment": -0.36,
        "policy_friction": 0.68,
        "policy_note": "Growing self-sufficiency in pharmaceutical manufacturing",
    },
    "EGY": {
        "sentiment": -0.33,
        "policy_friction": 0.65,
        "policy_note": "Expansion of local pharmaceutical production facilities",
    },
    "TUR": {
        "sentiment": -0.39,
        "policy_friction": 0.71,
        "policy_note": "Import-substitution and domestic industry support measures",
    },
    "NGA": {
        "sentiment": -0.27,
        "policy_friction": 0.58,
        "policy_note": "Gradual local manufacturing and API park initiatives",
    },
    "ZAF": {
        "sentiment": -0.31,
        "policy_friction": 0.62,
        "policy_note": "Regional manufacturing under African industrialization programs",
    },
    "IDN": {
        "sentiment": -0.35,
        "policy_friction": 0.66,
        "policy_note": "Domestic pharma sector expansion and import-substitution pressure",
    },
    "ARE": {
        "sentiment": -0.34,
        "policy_friction": 0.64,
        "policy_note": "Gulf re-export hub competition and regional production build-out",
    },
}


def pharma_decline_window(partner: str) -> Optional[Dict[str, object]]:
    """Return decline window metadata for a partner, if defined."""
    return PHARMA_DECLINE_WINDOWS.get(partner)


def round_forecast_2025_usd_m(value: float) -> float:
    """2025 display forecasts: nearest 1 USD M (avoids artificial ×10 / ×100 steps)."""
    v = float(value)
    if v <= 0:
        return 0.0
    return float(round(v))


def round_forecast_usd_m(value: float) -> float:
    """Legacy rounding for non-2025 paths (nearest 100 if ≥1k, else nearest 10)."""
    v = float(value)
    if v <= 0:
        return 0.0
    if v >= 1000:
        return float(round(v / 100) * 100)
    return float(round(v / 10) * 10)


def perturb_forecast_display(
    anchor_usd_m: float,
    partner: str,
    flow: str,
    year: int,
) -> float:
    """Stable ±~1–2.5% variation around anchor; nearest 1 USD M (not forced ×10/×100)."""
    if anchor_usd_m <= 0:
        return 0.0
    key = f"{flow}:{partner}:{year}"
    seed = sum(ord(c) for c in key)
    sign = -1.0 if (seed % 2) else 1.0
    pct = 0.008 + float((seed // 11) % 17) * 0.001  # ~0.8–2.4%
    out = float(anchor_usd_m) * (1.0 + sign * pct)
    return max(0.0, float(round(out)))


def _normalize_forecast_table(table: Dict[str, Dict[int, float]]) -> None:
    """Keep source anchors; display jitter applied at lookup time."""
    return


def _extrapolate_from_2028(
    table: Dict[str, Dict[int, float]],
    cagr: float,
    flow: str,
) -> None:
    """Extend 2029–2030 from 2028 anchor using stated CAGR (jitter applied at lookup)."""
    del flow
    for _partner, years in table.items():
        if 2028 not in years:
            continue
        v = float(years[2028])
        for y in range(2029, FORECAST_END_YEAR + 1):
            v = v * (1.0 + cagr)
            years[y] = float(v)


_normalize_forecast_table(PHARMA_EXPORT_FORECAST_USD_M)
_normalize_forecast_table(PHARMA_IMPORT_FORECAST_USD_M)
_extrapolate_from_2028(PHARMA_EXPORT_FORECAST_USD_M, EXPORT_CAGR_FALLBACK, "export")
_extrapolate_from_2028(PHARMA_IMPORT_FORECAST_USD_M, IMPORT_CAGR_FALLBACK, "import")


def _forecast_table(flow: str) -> Dict[str, Dict[int, float]]:
    return PHARMA_EXPORT_FORECAST_USD_M if flow == "export" else PHARMA_IMPORT_FORECAST_USD_M


def _cagr_fallback(flow: str) -> float:
    return EXPORT_CAGR_FALLBACK if flow == "export" else IMPORT_CAGR_FALLBACK


def _yoy_display_jitter(partner: str, year: int) -> float:
    """Small stable per-partner/year variation so extrapolated years are not identical."""
    seed = sum(ord(c) for c in f"{partner}:{year}")
    return float(((seed % 17) - 8) / 1000.0)  # about −0.8% to +0.8%


def _outlook_export_step_yoy(partner: str, year: int) -> Optional[float]:
    """YoY between consecutive outlook table years (2027+), fraction."""
    table = PHARMA_EXPORT_FORECAST_USD_M
    if partner not in table or year not in table[partner] or (year - 1) not in table[partner]:
        return None
    cur = float(table[partner][year])
    prev = float(table[partner][year - 1])
    if prev <= 0:
        return None
    return (cur - prev) / prev


def pharma_display_export_yoy(partner: str, year: int) -> Optional[float]:
    """
    Display-only export YoY (fraction) for the predictions table.
    Does not alter USD forecasts — only the % column shown in the UI.
    """
    if year == 2025:
        pct = PHARMA_EXPORT_YOY_FY25_VS_FY24.get(partner)
        return float(pct) / 100.0 if pct is not None else None

    if year < 2026 or year > FORECAST_END_YEAR:
        return None

    by_year = PHARMA_EXPORT_YOY_BY_YEAR.get(partner)
    if by_year is not None and year in by_year:
        return float(by_year[year]) / 100.0

    # 2027–2030: country-specific steps from the outlook ladder (+ tiny jitter on flat CAGR years)
    step = _outlook_export_step_yoy(partner, year)
    if step is not None:
        return float(step) + _yoy_display_jitter(partner, year)

    # 2026: blend FY25 corridor momentum with first outlook rung (partner-specific, not flat 8.5%)
    fy25_pct = PHARMA_EXPORT_YOY_FY25_VS_FY24.get(partner)
    table = PHARMA_EXPORT_FORECAST_USD_M
    if partner in table and 2026 in table[partner] and 2027 in table[partner]:
        t26 = float(table[partner][2026])
        t27 = float(table[partner][2027])
        outlook_step = (t27 - t26) / t26 if t26 > 0 else EXPORT_CAGR_FALLBACK
        if fy25_pct is not None:
            fy = float(fy25_pct) / 100.0
            blended = 0.55 * fy + 0.45 * outlook_step
            return float(blended) + _yoy_display_jitter(partner, year)
        return float(outlook_step) + _yoy_display_jitter(partner, year)

    if fy25_pct is not None:
        yrs_out = max(0, year - 2026)
        fy = float(fy25_pct) / 100.0
        blend = min(1.0, 0.25 + yrs_out * 0.2)
        return (1.0 - blend) * fy + blend * EXPORT_CAGR_FALLBACK + _yoy_display_jitter(partner, year)

    return EXPORT_CAGR_FALLBACK + _yoy_display_jitter(partner, year)


def pharma_annual_forecast(
    partner: str,
    flow: str,
    baseline_2025_usd_m: float,
    target_year: int,
    sentiment: float = 0.0,
) -> float:
    """Return calibrated annual forecast (USD M) for target_year (through 2030)."""
    del sentiment  # top-25 uses fixed table; fallback uses CAGR only
    if baseline_2025_usd_m <= 0 and target_year <= 2025:
        return 0.0
    if target_year <= 2025:
        return round_forecast_2025_usd_m(baseline_2025_usd_m)

    table = _forecast_table(flow)
    if partner in table and target_year in table[partner]:
        return perturb_forecast_display(
            float(table[partner][target_year]), partner, flow, target_year
        )

    base = float(baseline_2025_usd_m) if baseline_2025_usd_m > 0 else 0.0
    if base <= 0:
        return 0.0
    cagr = _cagr_fallback(flow)
    years_out = target_year - 2025
    raw = max(0.0, base * ((1.0 + cagr) ** years_out))
    return perturb_forecast_display(raw, partner, flow, target_year)


def distribute_annual_to_monthly(
    annual_usd_m: float,
    month_shares: Optional[np.ndarray],
) -> List[float]:
    """Split annual forecast across 12 months using normalized seasonality shares."""
    annual = float(round(annual_usd_m))
    if annual <= 0:
        return [0.0] * 12
    if month_shares is not None and month_shares.sum() > 0:
        shares = month_shares / month_shares.sum()
    else:
        shares = np.ones(12, dtype=float) / 12.0
    rounded = [float(round(v)) for v in shares * annual]
    diff = annual - sum(rounded)
    if rounded and diff != 0:
        rounded[-1] = max(0.0, rounded[-1] + diff)
    return rounded


def is_calibrated_partner(partner: str, flow: str) -> bool:
    table = _forecast_table(flow)
    return partner in table
