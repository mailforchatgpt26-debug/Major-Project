import type { AlertItem, Explainability, NewsArticle, Prediction } from "./types"

export function mockPredictions({ sector, month }: { sector: string; month: string }): Prediction[] {
  const countries = [
    { code: "USA", name: "United States", baseValue: 2500, change: 0.08 },
    { code: "CHN", name: "China", baseValue: 1800, change: -0.12 },
    { code: "DEU", name: "Germany", baseValue: 950, change: 0.15 },
    { code: "GBR", name: "United Kingdom", baseValue: 720, change: 0.05 },
    { code: "JPN", name: "Japan", baseValue: 680, change: -0.08 },
    { code: "FRA", name: "France", baseValue: 580, change: 0.12 },
    { code: "NLD", name: "Netherlands", baseValue: 520, change: 0.18 },
    { code: "ITA", name: "Italy", baseValue: 480, change: -0.05 },
    { code: "CAN", name: "Canada", baseValue: 450, change: 0.09 },
    { code: "KOR", name: "South Korea", baseValue: 420, change: -0.15 },
  ]

  return countries.map((country) => {
    const exportForecast = Math.round(country.baseValue * (1 + Math.random() * 0.2 - 0.1))
    const exportChange = country.change + (Math.random() * 0.1 - 0.05)
    return {
      partnerCode: country.code,
      partner: country.name,
      export_forecast: exportForecast,
      export_change: exportChange,
      import_change: exportChange * 0.8,
      confidence: 0.7 + Math.random() * 0.3,
      risk_level: country.change < -0.1 ? "high" : country.change > 0.1 ? "low" : "medium" as const,
    }
  })
}

export function mockAlerts({ sector, month }: { sector: string; month: string }): AlertItem[] {
  return [
    {
      id: "alert-1",
      type: "risk",
      title: "China Export Decline",
      summary: "Predicted 12% drop in pharmaceutical exports to China due to trade tensions",
      partner: "China",
      partnerCode: "CHN",
      change: -0.12,
      recommendations: [
        {
          country_code: "VNM",
          country_name: "Vietnam",
          predicted_value: 450,
          growth_rate: 0.22,
          confidence: 0.85,
          risk_level: "low",
          recommendation_score: 0.9,
          rationale: "Strong regional alternative with 22% predicted growth",
        },
        {
          country_code: "THA",
          country_name: "Thailand",
          predicted_value: 380,
          growth_rate: 0.18,
          confidence: 0.78,
          risk_level: "medium",
          recommendation_score: 0.8,
          rationale: "Diversify supply chain to reduce China dependency",
        },
      ],
    },
    {
      id: "alert-2",
      type: "opportunity",
      title: "Germany Market Growth",
      summary: "15% increase expected in textile exports to Germany",
      partner: "Germany",
      partnerCode: "DEU",
      change: 0.15,
      recommendations: [
        {
          country_code: "FRA",
          country_name: "France",
          predicted_value: 620,
          growth_rate: 0.12,
          confidence: 0.92,
          risk_level: "low",
          recommendation_score: 0.85,
          rationale: "Leverage proximity to capitalize on European growth",
        },
        {
          country_code: "NLD",
          country_name: "Netherlands",
          predicted_value: 540,
          growth_rate: 0.18,
          confidence: 0.88,
          risk_level: "low",
          recommendation_score: 0.88,
          rationale: "Use Dutch ports for optimized distribution",
        },
      ],
    },
    {
      id: "alert-3",
      type: "risk",
      title: "South Korea Sentiment Shift",
      summary: "Negative news sentiment affecting Korean market confidence",
      partner: "South Korea",
      partnerCode: "KOR",
      change: -0.15,
      recommendations: [
        {
          country_code: "JPN",
          country_name: "Japan",
          predicted_value: 710,
          growth_rate: 0.05,
          confidence: 0.9,
          risk_level: "low",
          recommendation_score: 0.75,
          rationale: "Stable alternative market with lower volatility",
        },
      ],
    },
  ]
}

export function mockNews({ sector, month, partner }: { sector: string; month: string; partner?: string }): NewsArticle[] {
  const articles = [
    {
      id: "news-1",
      title: "China Announces New Pharmaceutical Import Regulations",
      snippet: "New regulatory framework expected to impact foreign pharmaceutical imports...",
      source: "Reuters",
      url: "https://www.reuters.com/business/healthcare-pharmaceuticals/",
      date: "2024-01-15",
      sentiment: -0.3,
      relevance_score: 0.85,
      country_code: "CHN",
    },
    {
      id: "news-2",
      title: "Germany's Healthcare Sector Shows Strong Recovery",
      snippet: "German pharmaceutical market demonstrates robust growth in Q4...",
      source: "Bloomberg",
      url: "https://www.bloomberg.com/news/articles/2024-01-12/germany-s-healthcare-sector-recovery",
      date: "2024-01-12",
      sentiment: 0.6,
      relevance_score: 0.92,
      country_code: "DEU",
    },
    {
      id: "news-3",
      title: "US Textile Industry Faces Supply Chain Challenges",
      snippet: "Ongoing supply chain disruptions affecting textile imports...",
      source: "Wall Street Journal",
      url: "https://www.wsj.com/articles/supply-chain-crisis-textile-industry-11634563801",
      date: "2024-01-10",
      sentiment: -0.4,
      relevance_score: 0.78,
      country_code: "USA",
    },
  ]

  return (partner ? articles.filter(a => a.country_code === partner) : articles).map(art => ({
    ...art,
    // ONLY use search fallback if the URL is not a direct reuters/bloomberg/wsj link
    url: art.url.includes("reuters.com") || art.url.includes("bloomberg.com") || art.url.includes("wsj.com")
      ? art.url
      : `https://www.google.com/search?q=${encodeURIComponent(art.title)}&tbm=nws`
  }))
}

export function mockExplainability({ sector, month, partner }: { sector: string; month: string; partner?: string }): Explainability {
  return {
    attention: [
      { partner: "United States", weight: 0.35 },
      { partner: "China", weight: 0.28 },
      { partner: "Germany", weight: 0.22 },
      { partner: "United Kingdom", weight: 0.15 },
    ],
    features: [
      { feature: "GDP Growth", importance: 0.32 },
      { feature: "Trade Sentiment", importance: 0.28 },
      { feature: "Historical Trade Volume", importance: 0.24 },
      { feature: "Geographic Distance", importance: 0.16 },
    ],
    blurb: `The model predicts a ${partner ? `change for ${partner}` : 'general trend'} based primarily on economic indicators and trade sentiment. GDP growth shows the strongest correlation with export volumes, followed by recent news sentiment analysis.`,
  }
}