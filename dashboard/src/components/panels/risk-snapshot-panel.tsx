"use client"

import Link from "next/link"
import { AlertTriangle, ArrowDownRight, ArrowUpRight, Shield } from "lucide-react"
import type { ResiliencePartner, TradeResilience } from "@/lib/types"
import { formatPharmaExportSharePct } from "@/lib/pharma-constants"
import { corridorCardInsights } from "@/lib/corridor-insights"

function hhiBarColor(hhi: number) {
  if (hhi < 1500) return "bg-green-500"
  if (hhi < 2500) return "bg-amber-500"
  return "bg-red-500"
}

function hhiLabel(hhi: number, label?: string) {
  const plain: Record<string, string> = {
    competitive: "Well diversified",
    moderate: "Moderately concentrated",
    concentrated: "Highly concentrated",
  }
  return plain[label ?? ""] ?? label ?? (hhi < 1500 ? "Competitive" : hhi < 2500 ? "Moderate" : "Concentrated")
}

function corridorInsight(flags: string[]): string | null {
  const lines = corridorCardInsights(flags)
  if (!lines.length) return null
  const first = lines[0]
  return first.length > 120 ? `${first.slice(0, 117)}…` : first
}

function CorridorCard({
  p,
  variant,
}: {
  p: ResiliencePartner
  variant: "risk" | "opportunity"
}) {
  const yoy = p.export_change * 100
  const insight = corridorInsight(p.flags ?? [])

  return (
    <li className="rounded-lg border border-border/80 bg-muted/20 px-2.5 py-2 space-y-1">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-xs font-semibold truncate">{p.partner}</p>
          <p className="text-[10px] text-muted-foreground tabular-nums">
            ${(p.export_forecast / 1000).toFixed(1)}B forecast ·{" "}
            {formatPharmaExportSharePct(p.partnerCode, p.export_forecast, p.export_share)}% share
          </p>
        </div>
        <div className="shrink-0 text-right">
          <span
            className={`text-xs font-bold tabular-nums flex items-center justify-end gap-0.5 ${
              variant === "risk" ? "text-destructive" : "text-green-600 dark:text-green-400"
            }`}
          >
            {variant === "risk" ? (
              <ArrowDownRight className="size-3" />
            ) : (
              <ArrowUpRight className="size-3" />
            )}
            {yoy >= 0 ? "+" : ""}
            {yoy.toFixed(1)}%
          </span>
          <span
            className={`inline-block mt-0.5 rounded px-1 py-0.5 text-[9px] font-medium uppercase tracking-wide ${
              p.risk_level === "high"
                ? "bg-red-500/15 text-red-600 dark:text-red-400"
                : p.risk_level === "medium"
                ? "bg-amber-500/15 text-amber-700 dark:text-amber-400"
                : "bg-green-500/15 text-green-700 dark:text-green-400"
            }`}
          >
            {p.risk_level}
          </span>
        </div>
      </div>
      {insight && (
        <p className="text-[10px] text-muted-foreground leading-snug line-clamp-2">{insight}</p>
      )}
      <div className="h-1 w-full rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full rounded-full ${
            p.resilience_score >= 0.55 ? "bg-green-500" : p.resilience_score >= 0.4 ? "bg-amber-500" : "bg-red-500"
          }`}
          style={{ width: `${Math.round(p.resilience_score * 100)}%` }}
        />
      </div>
    </li>
  )
}

function ConcentrationMetric({
  title,
  hhi,
  label,
  loading,
}: {
  title: string
  hhi: number
  label?: string
  loading: boolean
}) {
  return (
    <div className="rounded-lg border border-border/60 bg-muted/15 p-2.5">
      <div className="flex items-center justify-between text-[11px] mb-1.5">
        <span className="text-muted-foreground">{title}</span>
        <span className="font-semibold tabular-nums">{loading ? "—" : hhi.toFixed(0)}</span>
      </div>
      <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${hhiBarColor(hhi)}`}
          style={{ width: `${Math.min(100, (hhi / 10000) * 100)}%` }}
        />
      </div>
      <p className="text-[10px] text-muted-foreground mt-1">{loading ? "…" : hhiLabel(hhi, label)}</p>
    </div>
  )
}

export function RiskSnapshotPanel({
  resilience,
  loading,
}: {
  resilience?: TradeResilience
  loading: boolean
}) {
  const risks = resilience?.top_risks ?? []
  const opps = resilience?.top_opportunities ?? []
  const topMarket = resilience?.partners?.length
    ? resilience.partners.reduce((best, p) =>
        p.export_forecast > best.export_forecast ? p : best
      )
    : undefined
  const summary =
    resilience?.summary ??
    "Concentration risk and corridor-level export forecasts for India's pharma portfolio."

  return (
    <div className="w-full lg:w-[22rem] xl:w-[24rem] shrink-0 border-t lg:border-t-0 lg:border-l border-border flex flex-col min-h-[480px] lg:min-h-[560px] bg-card/40">
      <div className="px-4 pt-4 pb-3 border-b border-border/60">
        <div className="flex items-start gap-2">
          <div className="rounded-md bg-primary/10 p-1.5">
            <Shield className="size-4 text-primary" />
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Risk Snapshot
            </p>
            <p className="text-sm font-semibold">Trade Resilience</p>
          </div>
        </div>
        <p className="text-[11px] text-muted-foreground mt-2 leading-relaxed line-clamp-3">{summary}</p>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
        <div className="grid grid-cols-2 gap-2">
          <ConcentrationMetric
            title="Export HHI"
            hhi={resilience?.export_hhi ?? 0}
            label={resilience?.export_hhi_label}
            loading={loading && !resilience}
          />
          <ConcentrationMetric
            title="Import HHI"
            hhi={resilience?.import_hhi ?? 0}
            label={resilience?.import_hhi_label}
            loading={loading && !resilience}
          />
        </div>

        {topMarket && !loading && (
          <div className="rounded-lg border border-primary/20 bg-primary/5 px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Largest corridor</p>
            <p className="text-sm font-semibold mt-0.5">{topMarket.partner}</p>
            <p className="text-[11px] text-muted-foreground">
              {formatPharmaExportSharePct(
                topMarket.partnerCode,
                topMarket.export_forecast,
                topMarket.export_share
              )}
              % of national pharma exports · $
              {(topMarket.export_forecast / 1000).toFixed(1)}B forecast
            </p>
          </div>
        )}

        {loading && !resilience && (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-14 rounded-lg bg-muted/50 animate-pulse" />
            ))}
          </div>
        )}

        {risks.length > 0 && (
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <AlertTriangle className="size-3.5 text-destructive" />
              <p className="text-xs font-semibold text-destructive">Vulnerable corridors</p>
              <span className="text-[10px] text-muted-foreground ml-auto">{risks.length} markets</span>
            </div>
            <ul className="space-y-2">
              {risks.slice(0, 3).map((r) => (
                <CorridorCard key={r.partnerCode} p={r} variant="risk" />
              ))}
            </ul>
          </div>
        )}

        {opps.length > 0 && (
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <ArrowUpRight className="size-3.5 text-green-600 dark:text-green-400" />
              <p className="text-xs font-semibold text-green-700 dark:text-green-400">Growth opportunities</p>
              <span className="text-[10px] text-muted-foreground ml-auto">{opps.length} markets</span>
            </div>
            <ul className="space-y-2">
              {opps.slice(0, 3).map((r) => (
                <CorridorCard key={r.partnerCode} p={r} variant="opportunity" />
              ))}
            </ul>
          </div>
        )}

        {!loading && resilience && risks.length === 0 && opps.length === 0 && (
          <p className="text-xs text-muted-foreground">No corridor risk data for this sector.</p>
        )}
      </div>

      <div className="px-4 py-3 border-t border-border shrink-0">
        <Link
          href="/resilience"
          className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-border bg-muted/50 px-3 py-2.5 text-xs font-medium hover:bg-muted transition-colors"
        >
          Full risk analysis →
        </Link>
      </div>
    </div>
  )
}
