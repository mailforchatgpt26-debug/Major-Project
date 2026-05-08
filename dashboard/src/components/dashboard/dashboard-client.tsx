"use client"

import { useEffect } from "react"
import Link from "next/link"
import { useDashboardStore } from "./store"
import { TradeNetwork } from "../trade-network"
import { PredictionsTable } from "../predictions-table"
import { NewsPanel } from "../panels/news-panel"
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

          {/* Risk snapshot panel */}
          <div className="w-full lg:w-72 shrink-0 border-t lg:border-t-0 lg:border-l border-border flex flex-col">
            <div className="px-4 pt-4 pb-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Risk Snapshot</p>
              <p className="text-sm font-semibold mt-0.5">Trade Resilience</p>
            </div>

            {/* HHI */}
            <div className="px-4 py-2 space-y-2">
              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-muted-foreground">Export concentration (HHI)</span>
                  <span className="font-medium tabular-nums">
                    {loading.resilience && !resilience ? "—" : (resilience?.export_hhi?.toFixed(0) ?? "—")}
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      (resilience?.export_hhi ?? 0) < 1500
                        ? "bg-green-500"
                        : (resilience?.export_hhi ?? 0) < 2500
                        ? "bg-amber-500"
                        : "bg-red-500"
                    }`}
                    style={{ width: `${Math.min(100, ((resilience?.export_hhi ?? 0) / 10000) * 100)}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground mt-0.5 capitalize">{resilience?.export_hhi_label ?? ""}</p>
              </div>

              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-muted-foreground">Import concentration (HHI)</span>
                  <span className="font-medium tabular-nums">
                    {loading.resilience && !resilience ? "—" : (resilience?.import_hhi?.toFixed(0) ?? "—")}
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      (resilience?.import_hhi ?? 0) < 1500
                        ? "bg-green-500"
                        : (resilience?.import_hhi ?? 0) < 2500
                        ? "bg-amber-500"
                        : "bg-red-500"
                    }`}
                    style={{ width: `${Math.min(100, ((resilience?.import_hhi ?? 0) / 10000) * 100)}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground mt-0.5 capitalize">{resilience?.import_hhi_label ?? ""}</p>
              </div>
            </div>

            {/* Top risks */}
            {(resilience?.top_risks?.length ?? 0) > 0 && (
              <div className="px-4 py-2 border-t border-border">
                <p className="text-xs font-medium text-destructive mb-1.5">Vulnerable Corridors</p>
                <ul className="space-y-1.5">
                  {resilience!.top_risks.slice(0, 3).map((r) => (
                    <li key={r.partnerCode} className="flex items-center justify-between text-xs">
                      <span className="font-medium truncate max-w-[120px]">{r.partner}</span>
                      <div className="flex items-center gap-2 shrink-0">
                        <span className="text-destructive tabular-nums">{(r.export_change * 100).toFixed(1)}%</span>
                        <span
                          className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                            r.risk_level === "high"
                              ? "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300"
                              : "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300"
                          }`}
                        >
                          {r.risk_level}
                        </span>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Top opportunities */}
            {(resilience?.top_opportunities?.length ?? 0) > 0 && (
              <div className="px-4 py-2 border-t border-border">
                <p className="text-xs font-medium mb-1.5" style={{ color: "var(--color-chart-1)" }}>
                  Growth Opportunities
                </p>
                <ul className="space-y-1.5">
                  {resilience!.top_opportunities.slice(0, 3).map((r) => (
                    <li key={r.partnerCode} className="flex items-center justify-between text-xs">
                      <span className="font-medium truncate max-w-[120px]">{r.partner}</span>
                      <span className="text-green-600 dark:text-green-400 tabular-nums shrink-0">
                        +{(r.export_change * 100).toFixed(1)}%
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Skeleton when loading */}
            {loading.resilience && !resilience && (
              <div className="px-4 py-2 space-y-2">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="h-4 rounded bg-muted/60 animate-pulse" />
                ))}
              </div>
            )}

            {/* CTA */}
            <div className="mt-auto px-4 py-4 border-t border-border">
              <Link
                href="/resilience"
                className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-border bg-muted/40 px-3 py-2 text-xs font-medium hover:bg-muted transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                Full risk analysis →
              </Link>
            </div>
          </div>
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
