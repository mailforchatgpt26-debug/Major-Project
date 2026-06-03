"use client"

import { create } from "zustand"
import type { AlertItem, Explainability, NewsArticle, Prediction, SimulationResult, TradeResilience } from "@/lib/types"
import { mockAlerts, mockExplainability, mockNews, mockPredictions, mockResilience } from "@/lib/mock"
import { apiFetchInit, getApiBaseUrl } from "@/lib/api-base"

type State = {
  sector: "pharma"
  month: string // YYYY-MM
  selectedPartner?: string
  predictions: Prediction[]
  alerts: AlertItem[]
  news: NewsArticle[]
  explainability?: Explainability
  resilience?: TradeResilience
  loading: {
    predictions: boolean
    alerts: boolean
    news: boolean
    explainability: boolean
    simulation: boolean
    resilience: boolean
  }
  error: {
    predictions?: string
    alerts?: string
    news?: string
    explainability?: string
    resilience?: string
  }
  apiConnected: boolean
  simulationResult?: SimulationResult
}

type Actions = {
  setSector: (s: "pharma") => void
  setMonth: (m: string) => void
  selectPartner: (countryCode: string) => void
  loadPredictions: () => Promise<void>
  loadAlerts: () => Promise<void>
  loadNews: (partner?: string) => Promise<void>
  loadExplainability: (partner?: string) => Promise<void>
  loadResilience: () => Promise<void>
  runSimulation: (target_country: string, feature: string, change_percent: number) => Promise<void>
}

export const useDashboardStore = create<State & Actions>((set, get) => ({
  sector: "pharma",
  month: "2025-01",
  predictions: [],
  alerts: [],
  news: [],
  explainability: undefined,
  resilience: undefined,
  loading: {
    predictions: false,
    alerts: false,
    news: false,
    explainability: false,
    simulation: false,
    resilience: false,
  },
  error: {},
  apiConnected: false,
  simulationResult: undefined,

  setSector: (sector) => set({ sector }),
  setMonth: (month) => set({ month }),
  selectPartner: (selectedPartner) => set({ selectedPartner }),

  loadPredictions: async () => {
    const { sector, month } = get()

    set((state) => ({
      loading: { ...state.loading, predictions: true },
      error: { ...state.error, predictions: undefined },
    }))

    try {
      // Try real API first
      const res = await fetch(
        `${getApiBaseUrl()}/api/predictions?sector=${sector}&month=${month}`,
        { ...apiFetchInit, signal: AbortSignal.timeout(20000) }
      )

      if (!res.ok) throw new Error(`API returned ${res.status}`)

      const predictions = await res.json()

      set((state) => ({
        predictions,
        apiConnected: true,
        loading: { ...state.loading, predictions: false },
      }))
      console.log(`✓ Loaded ${predictions.length} predictions from API`)
    } catch (error) {
      console.error("Failed to load predictions:", error)
      const fallbackPredictions = mockPredictions({ sector, month })
      set((state) => ({
        predictions: fallbackPredictions,
        apiConnected: false,
        loading: { ...state.loading, predictions: false },
      }))
      console.log(`⚠ Using ${fallbackPredictions.length} mock predictions (backend unavailable)`)
    }
  },

  loadAlerts: async () => {
    const { sector } = get()
    const month = "2025-01"

    set((state) => ({
      loading: { ...state.loading, alerts: true },
      error: { ...state.error, alerts: undefined },
    }))

    try {
      const res = await fetch(
        `${getApiBaseUrl()}/api/alerts?sector=${sector}&month=${month}`,
        { ...apiFetchInit, signal: AbortSignal.timeout(20000) }
      )

      if (!res.ok) throw new Error(`API returned ${res.status}`)

      const alerts = await res.json()

      set((state) => ({
        alerts,
        loading: { ...state.loading, alerts: false },
      }))
      console.log(`✓ Loaded ${alerts.length} alerts from API`)
    } catch (error) {
      console.error("Failed to load alerts:", error)
      const fallbackAlerts = mockAlerts({ sector, month })
      set((state) => ({
        alerts: fallbackAlerts,
        loading: { ...state.loading, alerts: false },
      }))
      console.log(`⚠ Using ${fallbackAlerts.length} mock alerts (backend unavailable)`)
    }
  },

  loadNews: async (partner) => {
    const { sector, month } = get()
    const buildFallbackNews = () => {
      const fallbackNews = mockNews({ sector, month })
      return partner && partner !== "undefined"
        ? fallbackNews.filter((article) => article.country_code === partner)
        : fallbackNews
    }

    set((state) => ({
      loading: { ...state.loading, news: true },
      error: { ...state.error, news: undefined },
    }))

    try {
      const res = await fetch(
        `${getApiBaseUrl()}/api/news?sector=${sector}&month=${month}`,
        { ...apiFetchInit, signal: AbortSignal.timeout(65000) }
      )

      if (!res.ok) {
        const filteredFallbackNews = buildFallbackNews()
        set((state) => ({
          news: filteredFallbackNews,
          loading: { ...state.loading, news: false },
        }))
        return
      }

      const news = await res.json()
      const filteredNews =
        partner && partner !== "undefined"
          ? news.filter((article: NewsArticle) => article.country_code === partner)
          : news

      set((state) => ({
        news: filteredNews,
        loading: { ...state.loading, news: false },
      }))
      console.log(`✓ Loaded ${news.length} general articles from API (${filteredNews.length} after filter)`)
    } catch (error) {
      const filteredFallbackNews = buildFallbackNews()
      set((state) => ({
        news: filteredFallbackNews,
        loading: { ...state.loading, news: false },
      }))
      console.log(`⚠ Using fallback news (${filteredFallbackNews.length} after filter)`)
    }
  },

  loadExplainability: async (partner) => {
    const { sector, month, selectedPartner } = get()
    const targetPartner = partner || selectedPartner

    if (!targetPartner || targetPartner === "undefined") {
      set({ explainability: undefined })
      return
    }

    set((state) => ({
      loading: { ...state.loading, explainability: true },
      error: { ...state.error, explainability: undefined },
    }))

    try {
      const res = await fetch(
        `${getApiBaseUrl()}/api/explainability?sector=${sector}&month=${month}&partner=${targetPartner}`,
        { ...apiFetchInit, signal: AbortSignal.timeout(20000) }
      )

      if (!res.ok) throw new Error(`API returned ${res.status}`)

      const explainability = await res.json()

      set((state) => ({
        explainability,
        loading: { ...state.loading, explainability: false },
      }))
      console.log(`✓ Loaded explainability from API for ${targetPartner}`)
    } catch (error) {
      console.error("Failed to load explainability:", error)
      const fallbackExplainability = mockExplainability({ sector, month, partner: targetPartner })
      set((state) => ({
        explainability: fallbackExplainability,
        loading: { ...state.loading, explainability: false },
      }))
      console.log(`⚠ Using mock explainability for ${targetPartner} (backend unavailable)`)
    }
  },

  loadResilience: async () => {
    const { sector } = get()
    const month = "2025-01"
    set((state) => ({ loading: { ...state.loading, resilience: true }, error: { ...state.error, resilience: undefined } }))
    try {
      const res = await fetch(
        `${getApiBaseUrl()}/api/resilience?sector=${sector}&month=${month}`,
        { ...apiFetchInit, signal: AbortSignal.timeout(30000) }
      )
      if (!res.ok) throw new Error(`API returned ${res.status}`)
      const resilience = await res.json()
      set((state) => ({ resilience, loading: { ...state.loading, resilience: false } }))
    } catch (error) {
      console.error("Failed to load resilience:", error)
      const fallbackResilience = mockResilience({ sector, month })
      set((state) => ({
        resilience: fallbackResilience,
        loading: { ...state.loading, resilience: false },
      }))
      console.log("⚠ Using mock resilience (backend unavailable)")
    }
  },

  runSimulation: async (targetCountry: string, feature: string, changePercent: number) => {
    const { sector, month } = get()

    set((state) => ({
      loading: { ...state.loading, simulation: true },
    }))

    try {
      // Keep spinner visible for at least 2–4s so users perceive active fetching.
      const minDelayMs = 2000 + Math.floor(Math.random() * 2000)
      const minDelay = new Promise((resolve) => setTimeout(resolve, minDelayMs))
      const fetchPromise = fetch(`${getApiBaseUrl()}/api/v1/simulate`, {
        method: "POST",
        ...apiFetchInit,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_country: targetCountry,
          feature,
          change_percent: changePercent,
          sector,
          month
        })
      })

      const [res] = await Promise.all([fetchPromise, minDelay])

      if (!res.ok) throw new Error("Simulation failed")

      const result = await res.json()
      set({ simulationResult: result })
    } catch (error) {
      console.error("Simulation failed:", error)
    } finally {
      set((state) => ({
        loading: { ...state.loading, simulation: false },
      }))
    }
  },
}))
