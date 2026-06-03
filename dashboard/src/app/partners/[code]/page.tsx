"use client"

import Link from "next/link"
import { useEffect, useMemo } from "react"
import { useParams } from "next/navigation"
import { useDashboardStore } from "@/components/dashboard/store"
import { NewsPanel } from "@/components/panels/news-panel"
import { ExplainabilityPanel } from "@/components/panels/explainability-panel"
import { TradeNetwork } from "@/components/trade-network" // show a globe for this country

export default function PartnerDetailsPage() {
  const { code } = useParams<{ code: string }>()
  const { predictions, news, explainability, loadPredictions, loadNews, loadExplainability } =
    useDashboardStore()

  useEffect(() => {
    // Load partner-focused data for this page.
    loadPredictions()
    loadNews(code)
    loadExplainability(code)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [code])

  const partnerName = useMemo(
    () => predictions.find((p) => p.partnerCode === code)?.partner || code,
    [predictions, code],
  )
  const partnerPrediction = useMemo(
    () => predictions.find((p) => p.partnerCode === code),
    [predictions, code],
  )

  const fmtUsdM = (v?: number) => (typeof v === "number" ? `${v.toLocaleString(undefined, { maximumFractionDigits: 1 })} USDm` : "—")
  const fmtPct = (v?: number) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—")
  const riskLevel = partnerPrediction?.risk_level
  const riskBadgeClass =
    riskLevel === "high"
      ? "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300"
      : riskLevel === "medium"
        ? "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300"
        : "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300"

  return (
    <main className="min-h-dvh">
      <div className="mx-auto max-w-[1200px] px-4 lg:px-6 py-6 space-y-6">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-pretty">{partnerName}</h1>
            <p className="text-sm text-muted-foreground">Detailed export/import view and latest trade news</p>
          </div>
          <Link href="/" className="text-sm underline underline-offset-4 text-muted-foreground hover:text-foreground">
            Back to Dashboard
          </Link>
        </header>

        <section className="grid gap-4 md:grid-cols-5">
          <div className="md:col-span-3 rounded-lg border bg-card/80 shadow-sm p-3">
            <h2 className="text-sm font-semibold mb-2">Trade Network — Focus</h2>
            <TradeNetwork
              data={predictions.filter((p) => p.partnerCode === code)}
              selectedPartner={code}
              onSelectPartner={() => { }}
            />
          </div>
          <div className="md:col-span-2 rounded-lg border bg-card/80 shadow-sm p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold">Forecast Snapshot</h2>
              <span className={`rounded px-2 py-1 text-xs font-semibold uppercase tracking-wide ${riskBadgeClass}`}>
                {riskLevel ?? "low"} risk
              </span>
            </div>
            <div className="rounded-md border p-3">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Export (India → {partnerName})</h3>
              <div className="text-sm space-y-1">
                <div>Actual {partnerPrediction?.export_actual_year ?? "2025"}: {fmtUsdM(partnerPrediction?.export_actual)}</div>
                <div>Forecast: {fmtUsdM(partnerPrediction?.export_forecast)}</div>
                <div>
                  YoY change (vs FY2024):{" "}
                  <span style={{ color: (partnerPrediction?.export_change ?? 0) >= 0 ? "var(--color-chart-1)" : "var(--destructive)" }}>
                    {fmtPct(partnerPrediction?.export_change)}
                  </span>
                </div>
              </div>
            </div>
            <div className="rounded-md border p-3">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Import ({partnerName} → India)</h3>
              <div className="text-sm space-y-1">
                <div>Actual {partnerPrediction?.import_actual_year ?? "2025"}: {fmtUsdM(partnerPrediction?.import_actual)}</div>
                <div>Forecast: {fmtUsdM(partnerPrediction?.import_forecast)}</div>
                <div>
                  Change vs actual:{" "}
                  <span style={{ color: (partnerPrediction?.import_change ?? 0) >= 0 ? "var(--color-chart-1)" : "var(--destructive)" }}>
                    {fmtPct(partnerPrediction?.import_change)}
                  </span>
                </div>
              </div>
            </div>
            <div className="rounded-md border p-3">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Seasonality</h3>
              <div className="text-sm space-y-1">
                <div>Peak month: {partnerPrediction?.export_peak_month ?? "—"}</div>
                <div>Low month: {partnerPrediction?.export_low_month ?? "—"}</div>
              </div>
            </div>
            <div className="text-xs text-muted-foreground">
              Confidence: {partnerPrediction ? `${Math.round(partnerPrediction.confidence * 100)}%` : "—"}
            </div>
          </div>
        </section>

        <section id="news" className="rounded-lg border bg-card/80 shadow-sm">
          <NewsPanel articles={news} />
        </section>

        <section id="explainability" className="rounded-lg border bg-card/80 shadow-sm">
          <div className="px-4 pt-4 text-sm font-semibold">What drives this forecast</div>
          <ExplainabilityPanel explainability={explainability} />
        </section>
      </div>
    </main>
  )
}
