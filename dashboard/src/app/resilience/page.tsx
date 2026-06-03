"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Header } from "@/components/header"
import { ThemeToggle } from "@/components/theme-toggle"
import { useDashboardStore } from "@/components/dashboard/store"
import type { AlertItem, ResiliencePartner, TradeResilience } from "@/lib/types"
import { formatPharmaExportSharePct } from "@/lib/pharma-constants"
import { apiFetchInit, getApiBaseUrl } from "@/lib/api-base"
import { corridorCardInsights } from "@/lib/corridor-insights"

const FIXED_MONTH = "2025-01"

// ── helpers ──────────────────────────────────────────────────────────────────

function ResilienceBar({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const color = pct >= 70 ? "bg-green-500" : pct >= 50 ? "bg-amber-500" : "bg-red-500"
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs tabular-nums w-8 text-right">{pct}/100</span>
    </div>
  )
}

function RiskBadge({ level }: { level: string }) {
  const styles: Record<string, string> = {
    high: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
    medium: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
    low: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300",
  }
  const label: Record<string, string> = { high: "High risk", medium: "Moderate", low: "Low risk" }
  return (
    <span className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${styles[level] ?? styles.low}`}>
      {label[level] ?? level}
    </span>
  )
}

// ── partner card ──────────────────────────────────────────────────────────────

function PartnerCard({ p, variant }: { p: ResiliencePartner; variant: "risk" | "opportunity" }) {
  const changePct = p.export_change * 100
  const changeLabel = changePct >= 0 ? `+${changePct.toFixed(1)}%` : `${changePct.toFixed(1)}%`
  const changeColor = changePct >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
  const exportSharePct = formatPharmaExportSharePct(p.partnerCode, p.export_forecast, p.export_share)
  const visibleFlags = corridorCardInsights(p.flags)

  const borderColor = variant === "risk"
    ? "border-red-200 dark:border-red-900/50"
    : "border-green-200 dark:border-green-900/50"

  return (
    <div className={`rounded-xl border ${borderColor} bg-card/80 p-4 space-y-3`}>
      {/* header row */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <span className="font-semibold text-sm">{p.partner}</span>
          <div className="mt-0.5">
            <RiskBadge level={p.risk_level} />
          </div>
        </div>
        <div className="text-right shrink-0">
          <div className={`text-xl font-bold tabular-nums ${changeColor}`}>{changeLabel}</div>
          <div className="text-xs text-muted-foreground">forecast change</div>
        </div>
      </div>

      {/* key facts */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <div>
          India's export share
          <span className="ml-1 font-medium text-foreground">{exportSharePct}%</span>
        </div>
        <div>
          Forecast
          <span className="ml-1 font-medium text-foreground">${p.export_forecast.toFixed(0)}M / yr</span>
        </div>
      </div>

      {/* resilience bar */}
      <div className="space-y-1">
        <div className="text-xs text-muted-foreground">Market stability score</div>
        <ResilienceBar score={p.resilience_score} />
      </div>

      {/* plain-English flags */}
      {visibleFlags.length > 0 && (
        <ul className="space-y-0.5">
          {visibleFlags.map((f) => (
            <li key={f} className="flex items-start gap-1.5 text-xs text-muted-foreground">
              <span className="mt-0.5 shrink-0">{variant === "risk" ? "⚠" : "✓"}</span>
              <span>{f}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// ── alert card ────────────────────────────────────────────────────────────────

function AlertCard({ alert }: { alert: AlertItem }) {
  // alert.change is a fraction (e.g. -0.322 = -32.2%)
  const changePct = alert.change * 100
  const changeLabel = changePct >= 0 ? `+${changePct.toFixed(1)}%` : `${changePct.toFixed(1)}%`
  const isRisk = alert.type === "risk"
  const changeColor = isRisk ? "text-red-600 dark:text-red-400" : "text-green-600 dark:text-green-400"
  const borderColor = isRisk ? "border-red-200 dark:border-red-800" : "border-green-200 dark:border-green-800"
  const bgColor = isRisk ? "bg-red-50 dark:bg-red-950/30" : "bg-green-50 dark:bg-green-950/30"

  return (
    <div className={`rounded-xl border ${borderColor} ${bgColor} p-5 space-y-3`}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="font-semibold text-base">{alert.title}</h3>
          <p className="text-sm text-muted-foreground mt-1">{alert.summary}</p>
        </div>
        <span className={`text-2xl font-bold whitespace-nowrap tabular-nums ${changeColor}`}>
          {changeLabel}
        </span>
      </div>

      {alert.recommendations && alert.recommendations.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
            What to do
          </p>
          <ul className="space-y-2">
            {alert.recommendations.map((rec, i) => {
              const text = (rec as Record<string, unknown>).text as string | undefined
                ?? (rec as Record<string, unknown>).rationale as string | undefined
              if (!text) return null
              return (
                <li key={i} className="flex gap-2 text-sm">
                  <span className="text-primary shrink-0">→</span>
                  <span>{text}</span>
                </li>
              )
            })}
          </ul>
        </div>
      )}

      <Link
        href={`/partners/${alert.partnerCode}`}
        className="text-xs text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
      >
        Full analysis →
      </Link>
    </div>
  )
}

// ── concentration summary ─────────────────────────────────────────────────────

function ConcentrationCard({ title, hhi, label }: { title: string; hhi: number; label: string }) {
  const pct = Math.min(100, (hhi / 10000) * 100)
  const barColor = hhi < 1500 ? "bg-green-500" : hhi < 2500 ? "bg-amber-500" : "bg-red-500"
  const plain: Record<string, string> = {
    competitive: "Well diversified — risk is spread across many partners",
    moderate: "Moderately concentrated — some reliance on a few key partners",
    concentrated: "Highly concentrated — heavy reliance on a small number of partners",
  }
  return (
    <div className="rounded-xl border border-border bg-card/80 shadow-sm p-5">
      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">{title}</p>
      <p className="mt-2 text-3xl font-bold tabular-nums">{hhi.toFixed(0)}</p>
      <p className="text-sm font-medium capitalize mt-0.5">{label}</p>
      <div className="mt-3 h-2 w-full rounded-full bg-muted overflow-hidden">
        <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-2 text-xs text-muted-foreground">{plain[label] ?? label}</p>
    </div>
  )
}

// ── skeleton ──────────────────────────────────────────────────────────────────

function SkeletonCard() {
  return <div className="animate-pulse h-32 rounded-xl bg-muted/50" />
}

// ── page ──────────────────────────────────────────────────────────────────────

export default function ResiliencePage() {
  const { sector } = useDashboardStore()

  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const [resilience, setResilience] = useState<TradeResilience | undefined>(undefined)
  const [loadingAlerts, setLoadingAlerts] = useState(false)
  const [loadingResilience, setLoadingResilience] = useState(false)

  useEffect(() => {
    // Always use 2025 data regardless of the year selected on the main dashboard
    setLoadingAlerts(true)
    fetch(`${getApiBaseUrl()}/api/alerts?sector=${sector}&month=${FIXED_MONTH}`, apiFetchInit)
      .then((r) => r.json())
      .then((data) => setAlerts(data))
      .catch(() => setAlerts([]))
      .finally(() => setLoadingAlerts(false))

    setLoadingResilience(true)
    fetch(`${getApiBaseUrl()}/api/resilience?sector=${sector}&month=${FIXED_MONTH}`, apiFetchInit)
      .then((r) => r.json())
      .then((data) => setResilience(data))
      .catch(() => setResilience(undefined))
      .finally(() => setLoadingResilience(false))
  }, [sector])

  const loading = { alerts: loadingAlerts, resilience: loadingResilience }
  const riskAlerts = alerts.filter((a) => a.type === "risk")
  const opportunityAlerts = alerts.filter((a) => a.type === "opportunity")

  return (
    <main className="min-h-dvh grid grid-rows-[auto_1fr]">
      <Header rightSlot={<ThemeToggle />} />

      <div className="max-w-[1200px] mx-auto w-full px-4 py-8 space-y-10">
        {/* title */}
        <div>
          <Link href="/" className="text-sm text-muted-foreground hover:text-foreground rounded">
            ← Back to Dashboard
          </Link>
          <h1 className="mt-3 text-2xl font-bold tracking-tight">Trade Resilience & Risk Analysis</h1>
          <p className="mt-1 text-muted-foreground text-sm">
            {loading.resilience
              ? "Loading..."
              : resilience?.summary || "Concentration risk and trade corridor analysis for India's pharma sector."}
          </p>
        </div>

        {/* how spread out are India's exports? */}
        <section>
          <h2 className="text-lg font-semibold mb-1">How spread out are India's trade partners?</h2>
          <p className="text-sm text-muted-foreground mb-4">
            A concentrated portfolio means one disrupted market can cause a large impact. Lower is better.
          </p>
          {loading.resilience && !resilience ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <SkeletonCard /><SkeletonCard />
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <ConcentrationCard
                title="Exports"
                hhi={resilience?.export_hhi ?? 0}
                label={resilience?.export_hhi_label ?? "—"}
              />
              <ConcentrationCard
                title="Imports"
                hhi={resilience?.import_hhi ?? 0}
                label={resilience?.import_hhi_label ?? "—"}
              />
            </div>
          )}
        </section>

        {/* vulnerable corridors — card grid */}
        <section>
          <h2 className="text-lg font-semibold mb-1">Vulnerable corridors</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Trade routes with the lowest resilience — most exposed to disruption.
          </p>
          {loading.resilience && !resilience ? (
            <SkeletonCard />
          ) : (resilience?.top_risks ?? []).length === 0 ? (
            <p className="text-sm text-muted-foreground">No data.</p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {(resilience?.top_risks ?? []).map((p) => (
                <PartnerCard key={p.partnerCode} p={p} variant="risk" />
              ))}
            </div>
          )}
        </section>

        {/* diversification targets — card grid */}
        <section>
          <h2 className="text-lg font-semibold mb-1">Where India could expand</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Markets with positive momentum and room to grow without adding concentration risk.
          </p>
          {loading.resilience && !resilience ? (
            <SkeletonCard />
          ) : (resilience?.top_opportunities ?? []).length === 0 ? (
            <p className="text-sm text-muted-foreground">No data.</p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {(resilience?.top_opportunities ?? []).map((p) => (
                <PartnerCard key={p.partnerCode} p={p} variant="opportunity" />
              ))}
            </div>
          )}
        </section>

        {/* risk alerts */}
        <section>
          <h2 className="text-lg font-semibold mb-1">Markets showing declining exports</h2>
          <p className="text-sm text-muted-foreground mb-4">
            These trade corridors are forecast to shrink. The model flags them as potential risks.
          </p>
          {loading.alerts && alerts.length === 0 ? (
            <div className="space-y-4"><SkeletonCard /><SkeletonCard /></div>
          ) : riskAlerts.length === 0 ? (
            <p className="text-sm text-muted-foreground">No risk alerts found.</p>
          ) : (
            <div className="space-y-4">
              {riskAlerts.map((alert) => <AlertCard key={alert.id} alert={alert} />)}
            </div>
          )}
        </section>

        {/* opportunities */}
        <section>
          <h2 className="text-lg font-semibold mb-1">Markets with strong growth potential</h2>
          <p className="text-sm text-muted-foreground mb-4">
            These corridors are forecast to grow. Expanding here could improve India's export diversity.
          </p>
          {loading.alerts && alerts.length === 0 ? (
            <div className="space-y-4"><SkeletonCard /><SkeletonCard /></div>
          ) : opportunityAlerts.length === 0 ? (
            <p className="text-sm text-muted-foreground">No growth opportunities found.</p>
          ) : (
            <div className="space-y-4">
              {opportunityAlerts.map((alert) => <AlertCard key={alert.id} alert={alert} />)}
            </div>
          )}
        </section>
      </div>
    </main>
  )
}
