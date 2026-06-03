export type MatrixCell = "yes" | "no" | "partial" | "limited"

export interface ComparisonRow {
  feature: string
  pharmaFootpath: MatrixCell
  pharmInt: MatrixCell
  gtaic: MatrixCell
  tradeInt: MatrixCell
  pharmaTrade: MatrixCell
}

export interface CompetitorAdvantage {
  title: string
  theirs: string
  ours: string
}

export interface CompetitorProfile {
  id: string
  name: string
  tagline: string
  focus: string[]
  advantages: CompetitorAdvantage[]
}

export const POSITIONING_STATEMENT =
  "Existing platforms focus primarily on trade intelligence and descriptive analytics, whereas PharmaTrade AI provides predictive, risk-aware, and explainable decision support by integrating forecasting, sentiment intelligence, simulation, and resilience analysis into a single platform."

export const ONE_LINE_CONCLUSION =
  "Unlike PharmaFootpath, PharmInt AI, GTAIC, and TradeInt, PharmaTrade AI not only provides trade intelligence but also forecasts future pharmaceutical trade flows, incorporates news sentiment, identifies vulnerable and opportunistic markets, and supports explainable risk-aware decision making through simulation and resilience analysis."

export const DIFFERENTIATORS = [
  {
    title: "Hybrid Gravity + GAT forecasting",
    description:
      "Combines economic gravity priors with graph attention over bilateral trade corridors to predict export and import flows through 2030.",
    icon: "network" as const,
  },
  {
    title: "News sentiment integration",
    description:
      "FinBERT scores on bilateral pharma trade news feed live edge features and country-pair sentiment for timelier forecasts.",
    icon: "news" as const,
  },
  {
    title: "Risk & opportunity intelligence",
    description:
      "Ranks vulnerable and high-opportunity corridors using localization pressure, policy friction, and trade concentration—not just static market facts.",
    icon: "risk" as const,
  },
  {
    title: "Simulation + explainability",
    description:
      "Policy scenario engine and attention-based explanations show why a market is risky or attractive and what changes under tariffs or sentiment shifts.",
    icon: "simulate" as const,
  },
]

export const COMPARISON_MATRIX: ComparisonRow[] = [
  {
    feature: "Historical trade analytics",
    pharmaFootpath: "yes",
    pharmInt: "yes",
    gtaic: "yes",
    tradeInt: "yes",
    pharmaTrade: "yes",
  },
  {
    feature: "Pharma-specific focus",
    pharmaFootpath: "yes",
    pharmInt: "yes",
    gtaic: "partial",
    tradeInt: "no",
    pharmaTrade: "yes",
  },
  {
    feature: "Bilateral trade forecasting (2026–2030)",
    pharmaFootpath: "no",
    pharmInt: "no",
    gtaic: "no",
    tradeInt: "no",
    pharmaTrade: "yes",
  },
  {
    feature: "News sentiment analysis",
    pharmaFootpath: "no",
    pharmInt: "no",
    gtaic: "no",
    tradeInt: "no",
    pharmaTrade: "yes",
  },
  {
    feature: "Risk assessment",
    pharmaFootpath: "no",
    pharmInt: "limited",
    gtaic: "limited",
    tradeInt: "limited",
    pharmaTrade: "yes",
  },
  {
    feature: "Opportunity detection",
    pharmaFootpath: "limited",
    pharmInt: "yes",
    gtaic: "yes",
    tradeInt: "limited",
    pharmaTrade: "yes",
  },
  {
    feature: "Scenario simulation",
    pharmaFootpath: "no",
    pharmInt: "no",
    gtaic: "no",
    tradeInt: "no",
    pharmaTrade: "yes",
  },
]

export const COMPETITORS: CompetitorProfile[] = [
  {
    id: "pharmafootpath",
    name: "PharmaFootpath",
    tagline: "Drug registration, pricing, and market-access intelligence",
    focus: [
      "Drug registration intelligence",
      "Pricing and market access",
      "Regulatory and supplier information",
    ],
    advantages: [
      {
        title: "Predicts future trade values",
        theirs: "Primarily current market and registration intelligence.",
        ours: "Forecasts bilateral pharmaceutical trade through 2030 with monthly and annual views.",
      },
      {
        title: "Identifies high-risk and high-opportunity markets",
        theirs: "Presents market facts without explicit corridor risk ranking.",
        ours: "Ranks vulnerable corridors and growth opportunities with measurable flags.",
      },
      {
        title: "Uses trade-network relationships",
        theirs: "Treats markets largely independently.",
        ours: "GAT learns dependencies across countries and bilateral corridors.",
      },
    ],
  },
  {
    id: "pharmint",
    name: "PharmInt AI",
    tagline: "Pharma export intelligence and buyer discovery",
    focus: [
      "Pharma export intelligence",
      "Buyer and supplier discovery",
      "Market opportunity analytics",
    ],
    advantages: [
      {
        title: "Forecasting instead of static analytics",
        theirs: "Strong on what happened and who to contact.",
        ours: "Predicts what is likely to happen next across top partners.",
      },
      {
        title: "News-driven sentiment intelligence",
        theirs: "Limited integration of news into quantitative trade models.",
        ours: "Converts bilateral trade news into FinBERT sentiment features per country pair.",
      },
      {
        title: "Explainable risk assessment",
        theirs: "Opportunity lists without deep model explanations.",
        ours: "Explains why a country is risky or attractive—sentiment, trends, simulations.",
      },
    ],
  },
  {
    id: "gtaic",
    name: "GTAIC",
    tagline: "AI-generated market reports and trade research",
    focus: [
      "AI-generated market reports",
      "Trade research",
      "Opportunity identification",
    ],
    advantages: [
      {
        title: "Quantitative forecasting",
        theirs: "Delivers narrative reports and research summaries.",
        ours: "Generates numerical bilateral trade predictions you can track over time.",
      },
      {
        title: "Scenario simulation",
        theirs: "No interactive what-if layer on trade flows.",
        ours: "Simulate tariff, sentiment, and macro shocks on GNN forecasts.",
      },
      {
        title: "Trade resilience analysis",
        theirs: "Highlights opportunities in prose.",
        ours: "Surfaces both opportunities and vulnerabilities with HHI and corridor scores.",
      },
    ],
  },
  {
    id: "tradeint",
    name: "TradeInt",
    tagline: "Shipment intelligence and historical trade analytics",
    focus: [
      "Shipment intelligence",
      "Historical trade analytics",
      "Competitor analysis",
    ],
    advantages: [
      {
        title: "Future-focused",
        theirs: "Anchored in historical shipment and trade records.",
        ours: "Future-oriented forecasts with 2025 actuals vs model alignment.",
      },
      {
        title: "AI-based risk scoring",
        theirs: "Competitive and shipment views without integrated risk alerts.",
        ours: "Risk alerts, resilience panel, and localization-aware corridor flags.",
      },
      {
        title: "Sentiment-aware forecasting",
        theirs: "Relies primarily on trade records.",
        ours: "Incorporates real-time and archived news sentiment in the graph model.",
      },
    ],
  },
]

export const MATRIX_COLUMNS = [
  { key: "pharmaFootpath" as const, label: "PharmaFootpath" },
  { key: "pharmInt" as const, label: "PharmInt AI" },
  { key: "gtaic" as const, label: "GTAIC" },
  { key: "tradeInt" as const, label: "TradeInt" },
  { key: "pharmaTrade" as const, label: "PharmaTrade AI", highlight: true },
]
