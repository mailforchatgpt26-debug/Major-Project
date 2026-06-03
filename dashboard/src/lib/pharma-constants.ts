/** National India pharma export total (USD M) — aligned with API `pharma_india_total_export_usd_m`. */
export const PHARMA_INDIA_TOTAL_EXPORT_USD_M_2025 = 31_109.8

/** FY2025 India→partner pharma exports (USD M) — mirrors `GOVT_PHARMA_EXPORT_ACTUAL_2025_USD_M` in API. */
export const GOVT_PHARMA_EXPORT_ACTUAL_2025_USD_M: Record<string, number> = {
  USA: 10515.11,
  GBR: 913.97,
  BRA: 778.49,
  FRA: 720.43,
  ZAF: 740,
  CAN: 620.08,
  DEU: 597.58,
  AUS: 469.76,
  RUS: 577.22,
  NLD: 616,
  ARE: 520,
  BEL: 450,
  CHN: 530,
  SAU: 211.37,
  MEX: 300,
  ITA: 244.59,
  ESP: 246.23,
  JPN: 231.52,
  POL: 203.57,
  TUR: 250,
  SGP: 160,
  THA: 210,
  VNM: 140,
  IDN: 130,
  LKA: 220,
  NPL: 260,
  BGD: 100,
  MYS: 95,
  NZL: 90,
  ARG: 85,
  DNK: 80,
  SWE: 75,
  FIN: 60,
  CZE: 55,
  HUN: 50,
  ROU: 50,
  GRC: 45,
  SVN: 40,
  JOR: 35,
  OMN: 35,
  DZA: 30,
  GHA: 30,
  NGA: 535.35,
  ETH: 25,
  UGA: 20,
  TZA: 25,
  MLT: 15,
  LVA: 10,
  HKG: 40,
  DOM: 20,
}

const EXPORT_CAGR = 0.085

export function pharmaIndiaTotalExportUsdM(year: number): number {
  if (year <= 2025) return PHARMA_INDIA_TOTAL_EXPORT_USD_M_2025
  return PHARMA_INDIA_TOTAL_EXPORT_USD_M_2025 * (1 + EXPORT_CAGR) ** (year - 2025)
}

/**
 * Share of India's national pharma exports (0–1).
 * Prefer FY2025 actual USD M when provided (e.g. Pharmexcil/Comtrade).
 */
export function pharmaNationalExportShare(
  exportUsdM: number,
  year: number,
  actual2025UsdM?: number | null
): number {
  const total = pharmaIndiaTotalExportUsdM(year)
  if (total <= 0) return 0
  if (year <= 2025 && actual2025UsdM != null && actual2025UsdM > 0) {
    return actual2025UsdM / total
  }
  return exportUsdM / total
}

/** Display share for resilience / risk UI — always uses national denominator + FY25 actual when known. */
export function displayPharmaExportShare(
  partnerCode: string,
  exportForecastUsdM: number,
  year = 2025
): number {
  const actual = GOVT_PHARMA_EXPORT_ACTUAL_2025_USD_M[partnerCode]
  return pharmaNationalExportShare(
    exportForecastUsdM,
    year,
    actual ?? null
  )
}

export function formatPharmaExportSharePct(
  partnerCode: string,
  exportForecastUsdM: number,
  apiShareFallback?: number,
  year = 2025
): string {
  const share =
    GOVT_PHARMA_EXPORT_ACTUAL_2025_USD_M[partnerCode] != null
      ? displayPharmaExportShare(partnerCode, exportForecastUsdM, year)
      : apiShareFallback ?? displayPharmaExportShare(partnerCode, exportForecastUsdM, year)
  return (share * 100).toFixed(1)
}
