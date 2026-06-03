"use client"

import { useEffect } from "react"
import { useDashboardStore } from "./store"
import { TradeNetwork } from "../trade-network"
import { PredictionsTable } from "../predictions-table"
import { NewsPanel } from "../panels/news-panel"
import { RiskSnapshotPanel } from "../panels/risk-snapshot-panel"
import { PolicyScenarioModal } from "../policy-scenario-modal"

export default function DashboardClient() {
  const {
    sector,
    month,
    selectedPartner,
    predictions,
    news,
    explainability,
    resilience,
    loading,
    apiConnected,
    selectPartner,
    loadPredictions,
    loadNews,
    loadExplainability,
    loadResilience,
  } = useDashboardStore()

  useEffect(() => {
    loadPredictions()
    loadNews(selectedPartner)
    loadExplainability(selectedPartner)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sector, month, selectedPartner])

  useEffect(() => {
    loadResilience()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sector])

  // Scroll to #predictions / #news when arriving from Risk & Resilience (or other pages)
  useEffect(() => {
    const hash = window.location.hash
    if (!hash) return
    const id = hash.replace(/^#/, "")
    const scroll = () => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" })
    scroll()
    const t = window.setTimeout(scroll, 400)
    return () => window.clearTimeout(t)
  }, [])

  return (
    <div className="mx-auto max-w-[1400px] space-y-6">
      {/* API connection indicator */}
      <div className="flex items-center gap-2 text-xs">
        <span className="size-2 rounded-full bg-green-500 animate-pulse" />
        <span className="text-muted-foreground">
          Live — Connected to GNN backend
        </span>
      </div>

      <section id="overview" className="scroll-mt-24">
        <div className="flex flex-col lg:flex-row gap-0 rounded-xl border bg-card/80 shadow-sm overflow-hidden">
          {/* Globe */}
          <div className="flex-1 min-w-0">
            <TradeNetwork
              data={predictions}
              selectedPartner={selectedPartner}
              onSelectPartner={(cc) => selectPartner(cc)}
            />
          </div>

          <RiskSnapshotPanel resilience={resilience} loading={loading.resilience} />
        </div>
      </section>

      <section id="predictions" className="rounded-xl border bg-card/80 shadow-sm scroll-mt-24">
        <PredictionsTable
          data={predictions}
          selectedPartner={selectedPartner}
          explainability={explainability}
          onRowSelect={(cc) => selectPartner(cc)}
        />
      </section>

      <section id="news" className="rounded-xl border bg-card/80 shadow-sm scroll-mt-24">
        <NewsPanel articles={news} />
      </section>

      <PolicyScenarioModal />
    </div>
  )
}
