import type { AlertItem, Explainability, NewsArticle, Prediction, ResiliencePartner, TradeResilience } from "./types"
import mockNewsPayload from "./mock-news-data.json"

import { displayPharmaExportShare } from "./pharma-constants"

/** Top pharma partners — aligned with API / GOVT 2025 export targets (USD M). */
const PHARMA_MOCK_PREDICTIONS: Prediction[] = [
  { partnerCode: "USA", partner: "United States", export_forecast: 10500, export_change: 0.05, import_change: 0.04, confidence: 0.92, risk_level: "low" },
  { partnerCode: "GBR", partner: "United Kingdom", export_forecast: 910, export_change: -0.014, import_change: 0.04, confidence: 0.88, risk_level: "low" },
  { partnerCode: "BRA", partner: "Brazil", export_forecast: 778, export_change: 0.10, import_change: 0.05, confidence: 0.85, risk_level: "medium" },
  { partnerCode: "ZAF", partner: "South Africa", export_forecast: 740, export_change: -0.018, import_change: 0.03, confidence: 0.84, risk_level: "low" },
  { partnerCode: "FRA", partner: "France", export_forecast: 720, export_change: 0.071, import_change: 0.06, confidence: 0.86, risk_level: "medium" },
  { partnerCode: "NLD", partner: "Netherlands", export_forecast: 616, export_change: -0.138, import_change: 0.05, confidence: 0.87, risk_level: "medium" },
  { partnerCode: "CAN", partner: "Canada", export_forecast: 620, export_change: -0.05, import_change: 0.04, confidence: 0.83, risk_level: "medium" },
  { partnerCode: "DEU", partner: "Germany", export_forecast: 598, export_change: 0.08, import_change: 0.06, confidence: 0.89, risk_level: "low" },
  { partnerCode: "RUS", partner: "Russia", export_forecast: 577, export_change: -0.05, import_change: 0.04, confidence: 0.78, risk_level: "high" },
  { partnerCode: "ARE", partner: "UAE", export_forecast: 520, export_change: -0.177, import_change: 0.05, confidence: 0.81, risk_level: "high" },
  { partnerCode: "CHN", partner: "China", export_forecast: 530, export_change: -0.106, import_change: 0.03, confidence: 0.80, risk_level: "high" },
  { partnerCode: "BEL", partner: "Belgium", export_forecast: 450, export_change: -0.074, import_change: 0.05, confidence: 0.82, risk_level: "medium" },
  { partnerCode: "NPL", partner: "Nepal", export_forecast: 260, export_change: 0.069, import_change: 0.15, confidence: 0.76, risk_level: "medium" },
  { partnerCode: "MEX", partner: "Mexico", export_forecast: 300, export_change: -0.038, import_change: 0.04, confidence: 0.79, risk_level: "low" },
  { partnerCode: "TUR", partner: "Turkey", export_forecast: 250, export_change: -0.16, import_change: 0.04, confidence: 0.77, risk_level: "high" },
  { partnerCode: "LKA", partner: "Sri Lanka", export_forecast: 220, export_change: -0.146, import_change: 0.03, confidence: 0.75, risk_level: "high" },
  { partnerCode: "THA", partner: "Thailand", export_forecast: 210, export_change: -0.001, import_change: 0.04, confidence: 0.80, risk_level: "low" },
]

export function mockPredictions(_opts: { sector: string; month: string }): Prediction[] {
  return PHARMA_MOCK_PREDICTIONS.map((p) => ({ ...p }))
}

export function mockAlerts(_opts: { sector: string; month: string }): AlertItem[] {
  return [
    {
      id: "alert-uae-decline",
      type: "risk",
      title: "UAE Export Decline",
      summary: "Pharmexcil reports ~17.7% drop in India pharma exports to UAE (FY2024 → FY2025)",
      partner: "UAE",
      partnerCode: "ARE",
      change: -0.177,
      recommendations: [
        {
          country_code: "SAU",
          country_name: "Saudi Arabia",
          predicted_value: 650,
          growth_rate: 0.08,
          confidence: 0.82,
          risk_level: "low",
          recommendation_score: 0.78,
          rationale: "Diversify Gulf exposure toward growing Saudi market",
        },
      ],
    },
    {
      id: "alert-nld-decline",
      type: "risk",
      title: "Netherlands Export Decline",
      summary: "India pharma exports to Netherlands down ~13.8% YoY per Pharmexcil data",
      partner: "Netherlands",
      partnerCode: "NLD",
      change: -0.138,
      recommendations: [
        {
          country_code: "DEU",
          country_name: "Germany",
          predicted_value: 790,
          growth_rate: 0.10,
          confidence: 0.88,
          risk_level: "low",
          recommendation_score: 0.85,
          rationale: "Shift EU portfolio toward stronger German demand",
        },
      ],
    },
    {
      id: "alert-usa-growth",
      type: "opportunity",
      title: "USA Remains Top Market",
      summary: "USA pharma exports ~$10.5B in 2025; recovery expected in specialty segments 2026+",
      partner: "United States",
      partnerCode: "USA",
      change: 0.05,
      recommendations: [
        {
          country_code: "CAN",
          country_name: "Canada",
          predicted_value: 790,
          growth_rate: 0.03,
          confidence: 0.84,
          risk_level: "low",
          recommendation_score: 0.72,
          rationale: "Expand NAFTA corridor alongside US focus",
        },
      ],
    },
    {
      id: "alert-npl-growth",
      type: "opportunity",
      title: "Nepal Export Growth",
      summary: "India → Nepal pharma exports estimated ~$260M in 2025, up from ~$244M in 2024",
      partner: "Nepal",
      partnerCode: "NPL",
      change: 0.069,
      recommendations: [],
    },
  ]
}

const MOCK_NEWS_ARTICLES: NewsArticle[] = (mockNewsPayload as { articles: NewsArticle[] }).articles ?? []

export function mockNews({
  sector: _sector,
  month: _month,
  partner,
}: {
  sector: string
  month: string
  partner?: string
}): NewsArticle[] {
  const articles = MOCK_NEWS_ARTICLES.length > 0 ? MOCK_NEWS_ARTICLES : []
  if (partner && partner !== "undefined") {
    return articles.filter((a) => a.country_code === partner)
  }
  return articles
}

/** Localization-risk corridors — aligned with API PHARMA_AVOID_MARKETS + decline-window YoY. */
const PHARMA_VULNERABLE: Array<{ code: string; name: string; yoy: number; forecast: number }> = [
  { code: "SAU", name: "Saudi Arabia", yoy: -0.082, forecast: 650 },
  { code: "ARE", name: "UAE", yoy: -0.10, forecast: 520 },
  { code: "TUR", name: "Turkey", yoy: -0.095, forecast: 250 },
  { code: "BGD", name: "Bangladesh", yoy: -0.09, forecast: 240 },
  { code: "EGY", name: "Egypt", yoy: -0.07, forecast: 270 },
  { code: "NGA", name: "Nigeria", yoy: -0.08, forecast: 310 },
  { code: "ZAF", name: "South Africa", yoy: -0.065, forecast: 740 },
  { code: "IDN", name: "Indonesia", yoy: -0.055, forecast: 130 },
]

const PHARMA_EXPANSION: Array<{ code: string; name: string; yoy: number; forecast: number }> = [
  { code: "USA", name: "United States", yoy: 0.143, forecast: 10500 },
  { code: "DEU", name: "Germany", yoy: 0.071, forecast: 598 },
  { code: "GBR", name: "United Kingdom", yoy: 0.17, forecast: 910 },
  { code: "JPN", name: "Japan", yoy: 0.052, forecast: 480 },
  { code: "CAN", name: "Canada", yoy: 0.094, forecast: 620 },
]

const MOCK_NEWS_SENTIMENT: Record<string, { sent: number; policy: number; note: string }> = {
  SAU: { sent: -0.41, policy: 0.74, note: "Vision 2030 localization and domestic biopharma capacity targets" },
  ARE: { sent: -0.34, policy: 0.64, note: "Gulf re-export hub competition and regional production build-out" },
  TUR: { sent: -0.39, policy: 0.71, note: "Import-substitution and domestic industry support measures" },
  BGD: { sent: -0.36, policy: 0.68, note: "Growing self-sufficiency in pharmaceutical manufacturing" },
  EGY: { sent: -0.33, policy: 0.65, note: "Expansion of local pharmaceutical production facilities" },
  NGA: { sent: -0.27, policy: 0.58, note: "Gradual local manufacturing and API park initiatives" },
  ZAF: { sent: -0.31, policy: 0.62, note: "Regional manufacturing under African industrialization programs" },
  IDN: { sent: -0.35, policy: 0.66, note: "Domestic pharma sector expansion and import-substitution pressure" },
  USA: { sent: 0.48, policy: 0.20, note: "Record India→US pharma investment pledges; constructive FDA dialogue" },
  DEU: { sent: 0.26, policy: 0.28, note: "Stable EU demand; periodic EU tariff scrutiny" },
  GBR: { sent: 0.31, policy: 0.26, note: "UK–India trade continuity; moderate customs friction" },
  JPN: { sent: 0.24, policy: 0.24, note: "Steady regulated market access" },
  CAN: { sent: 0.29, policy: 0.22, note: "North America corridor stability" },
}

function mockResiliencePartner(
  row: { code: string; name: string; yoy: number; forecast: number },
  risk_level: "low" | "medium" | "high"
): ResiliencePartner {
  const news = MOCK_NEWS_SENTIMENT[row.code]
  const sent = news?.sent ?? (risk_level === "high" ? -0.25 : 0.22)
  const policy = news?.policy ?? (risk_level === "high" ? 0.62 : 0.25)
  const note = news?.note ?? "Recent bilateral pharma/trade coverage"
  const sentLabel = sent >= 0.2 ? "positive" : sent <= -0.12 ? "negative" : "neutral"
  const policyLabel = policy < 0.35 ? "low" : policy < 0.55 ? "moderate" : "elevated"
  const sharePct = displayPharmaExportShare(row.code, row.forecast, 2025)
  const yoyPct = row.yoy * 100
  const flags =
    risk_level === "high"
      ? [
          `This corridor looks exposed: demand is softening (${yoyPct >= 0 ? "+" : ""}${yoyPct.toFixed(1)}% YoY in the forecast) while localization pressure is elevated (index 0.62).`,
          `Bilateral coverage skews ${sentLabel} (${sent >= 0 ? "+" : ""}${sent.toFixed(2)}) and trade-policy friction is ${policyLabel} (${policy.toFixed(2)}) — ${note}.`,
          `India's footprint is about $${row.forecast.toFixed(0)}M annually (${sharePct.toFixed(1)}% of national pharma exports) and remains partner import demand-led for now.`,
        ]
      : [
          `Export momentum looks constructive: the gravity–GNN stack implies ${yoyPct >= 0 ? "+" : ""}${yoyPct.toFixed(1)}% YoY growth on about $${row.forecast.toFixed(0)}M in bilateral pharma trade.`,
          `News we score for this pair reads ${sentLabel} (sentiment ${sent >= 0 ? "+" : ""}${sent.toFixed(2)}); policy friction is ${policyLabel} (${policy.toFixed(2)}) — ${note}.`,
          `At roughly $${row.forecast.toFixed(0)}M per year (~${sharePct.toFixed(1)}% of India's pharma exports), this corridor still runs on partner import demand rather than one-off spikes.`,
        ]
  return {
    partnerCode: row.code,
    partner: row.name,
    export_share: displayPharmaExportShare(row.code, row.forecast, 2025),
    import_share: 0.02,
    pagerank: 0.05,
    betweenness: 0.03,
    resilience_score: risk_level === "high" ? 0.35 : 0.72,
    risk_level,
    flags,
    export_forecast: row.forecast,
    export_change: row.yoy,
  }
}

export function mockResilience(_opts: { sector: string; month: string }): TradeResilience {
  const top_risks = PHARMA_VULNERABLE.map((r) => mockResiliencePartner(r, "high"))
  const top_opportunities = PHARMA_EXPANSION.map((r) => mockResiliencePartner(r, "low"))
  const partners = [...top_risks, ...top_opportunities]
  return {
    export_hhi: 2282,
    import_hhi: 787,
    export_hhi_label: "moderate",
    import_hhi_label: "competitive",
    partners,
    top_risks,
    top_opportunities,
    summary:
      "Vulnerable corridors (localization risk): Saudi Arabia, UAE, Turkey. " +
      "Expansion opportunities: United States, Germany, United Kingdom.",
  }
}

export function mockExplainability({
  sector: _sector,
  month: _month,
  partner,
}: {
  sector: string
  month: string
  partner?: string
}): Explainability {
  return {
    attention: [
      { partner: "United States", weight: 0.35 },
      { partner: "United Kingdom", weight: 0.12 },
      { partner: "Brazil", weight: 0.09 },
      { partner: "South Africa", weight: 0.08 },
      { partner: "Netherlands", weight: 0.07 },
    ],
    features: [
      { feature: "Historical Trade Volume", importance: 0.32 },
      { feature: "Trade Sentiment", importance: 0.28 },
      { feature: "GDP Growth", importance: 0.22 },
      { feature: "Geographic Distance", importance: 0.18 },
    ],
    blurb: partner
      ? `Forecast for ${partner} reflects bilateral trade history, FinBERT news sentiment from archived articles, and gravity-model features.`
      : "Forecasts combine GNN trade flows, bilateral sentiment from news archives, and macro indicators.",
  }
}
