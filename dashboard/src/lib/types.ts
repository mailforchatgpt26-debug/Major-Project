export interface Prediction {
  partnerCode: string
  partner: string
  // Export: India → partner
  export_actual?: number
  export_actual_year?: number
  export_forecast: number
  export_peak_month?: string
  export_low_month?: string
  export_change: number
  // Import: partner → India
  import_actual?: number
  import_actual_year?: number
  import_forecast?: number
  import_change: number
  confidence: number
  risk_level: "low" | "medium" | "high"
}

export interface AlertItem {
  id: string
  type: "opportunity" | "risk"
  title: string
  summary: string
  partner: string
  partnerCode: string
  change: number
  recommendations?: Array<{
    text?: string
    country_code?: string
    country_name?: string
    rationale?: string
    recommendation_score?: number
    risk_level?: string
    [key: string]: unknown
  }>
}

export interface NewsArticle {
  id: string
  title: string
  snippet: string
  source: string
  url: string
  date: string
  sentiment: number // -1.0 to 1.0
  relevance_score: number
  country_code?: string
}

export interface Explainability {
  attention: Array<{
    partner: string
    weight: number
  }>
  features: Array<{
    feature: string
    importance: number
  }>
  blurb: string
}
export interface ResiliencePartner {
  partnerCode: string
  partner: string
  export_share: number
  import_share: number
  pagerank: number
  betweenness: number
  resilience_score: number
  risk_level: "low" | "medium" | "high"
  flags: string[]
  export_forecast: number
  export_change: number
}

export interface TradeResilience {
  export_hhi: number
  import_hhi: number
  export_hhi_label: string
  import_hhi_label: string
  partners: ResiliencePartner[]
  top_risks: ResiliencePartner[]
  top_opportunities: ResiliencePartner[]
  summary: string
}

export interface SimulationResult {
  baseline: number
  counterfactual: number
  delta: number
  pct_impact: number
  global_impact: number
  partner_share: number   // fraction 0–1: this partner's share of India's total sector exports
  explanation: string
}

export interface PartnerMonthlySeries {
  partnerCode: string
  partner: string
  flow: "export" | "import"
  unit: string
  month_labels: string[]
  compare_2025: {
    actual: number[]
    forecast: number[]
    actual_chart?: number[]
    forecast_chart?: number[]
  }
  trend: Array<{ year: number; month: number; label: string; value: number }>
  annual_forecast: Record<string, number>
}
