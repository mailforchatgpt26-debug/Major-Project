"use client"

import { useEffect, useMemo, useState } from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { PartnerMonthlySeries } from "@/lib/types"
import { apiFetchInit, getApiBaseUrl } from "@/lib/api-base"

const COLORS = {
  actual: "#94a3b8",
  forecastExport: "#22c55e",
  forecastImport: "#3b82f6",
}

function fmtM(v: number, fine = false) {
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`
  if (fine) return v.toFixed(2)
  return `${Math.round(v)}`
}

function CompareTooltip({
  active,
  payload,
  label,
  fine = false,
}: {
  active?: boolean
  payload?: Array<{ name?: string; value?: number; payload?: { Actual?: number; Forecast?: number } }>
  label?: string
  fine?: boolean
}) {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload
  const fmt = (v: number) => (fine ? v.toFixed(2) : v.toFixed(1))
  return (
    <div className="rounded-md border bg-popover px-2.5 py-2 text-xs shadow-md">
      <p className="font-semibold mb-1">{label}</p>
      {row?.Actual != null && <p className="text-slate-400">Actual: {fmt(row.Actual)} M</p>}
      {row?.Forecast != null && <p className="text-foreground">Forecast: {fmt(row.Forecast)} M</p>}
    </div>
  )
}

function FlowComparePanel({
  title,
  series,
  forecastColor,
}: {
  title: string
  series: PartnerMonthlySeries
  forecastColor: string
}) {
  const fineChart = series.flow === "import"
  const { chartData, actualTotal, forecastTotal, annualDeltaPct, yDomain } = useMemo(() => {
    const compare = series.compare_2025
    if (!compare) {
      return {
        chartData: [],
        actualTotal: 0,
        forecastTotal: 0,
        annualDeltaPct: 0,
        yDomain: [0, 1] as [number, number],
      }
    }
    const actual = compare.actual ?? []
    const forecast = compare.forecast ?? []
    const actualBars = compare.actual_chart?.length ? compare.actual_chart : actual
    const forecastBars = compare.forecast_chart?.length ? compare.forecast_chart : forecast
    const actualTotal = actual.reduce((s, v) => s + v, 0)
    const forecastTotal = forecast.reduce((s, v) => s + v, 0)
    const chartFloor = (v: number, total: number) => {
      if (v > 0) return v
      if (total <= 0) return 0.4
      return Math.max(total * 0.03, 0.4)
    }
    const chartData = series.month_labels.map((month, i) => {
      const rawA = actualBars[i] ?? 0
      const rawF = forecastBars[i] ?? 0
      const a = fineChart ? rawA : chartFloor(rawA, actualTotal)
      const f = fineChart ? rawF : chartFloor(rawF, forecastTotal)
      return { month, Actual: a, Forecast: f }
    })
    const annualDeltaPct =
      actualTotal > 0 ? ((forecastTotal - actualTotal) / actualTotal) * 100 : 0
    const vals = chartData.flatMap((d) => [d.Actual, d.Forecast])
    const lo = Math.min(...vals)
    const hi = Math.max(...vals)
    const pad = Math.max((hi - lo) * 0.12, hi * 0.05, 1)
    return {
      chartData,
      actualTotal,
      forecastTotal,
      annualDeltaPct,
      yDomain: [Math.max(0, lo - pad), hi + pad] as [number, number],
    }
  }, [series])

  if (!series.compare_2025) {
    return (
      <div className="rounded-lg border bg-background/80 p-3">
        <p className="text-xs text-muted-foreground">{title}: comparison data unavailable.</p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border bg-background/80 p-3 space-y-3">
      <p className="text-xs font-semibold text-foreground">{title}</p>

      <div className="grid grid-cols-2 gap-2 text-center">
        <div className="rounded-md bg-muted/50 px-2 py-1.5">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wide">Actual</p>
          <p className="text-sm font-semibold tabular-nums">{Math.round(actualTotal).toLocaleString()}M</p>
        </div>
        <div className="rounded-md bg-muted/50 px-2 py-1.5">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wide">Forecast</p>
          <p className="text-sm font-semibold tabular-nums" style={{ color: forecastColor }}>
            {Math.round(forecastTotal).toLocaleString()}M
          </p>
          {actualTotal > 0 && (
            <p className="text-[10px] text-muted-foreground tabular-nums mt-0.5">
              {annualDeltaPct >= 0 ? "+" : ""}
              {annualDeltaPct.toFixed(1)}% vs actual
            </p>
          )}
        </div>
      </div>

      <div>
        <p className="text-[10px] text-muted-foreground mb-1">Monthly volume (grouped bars)</p>
        <div className="h-36 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }} barGap={2} barCategoryGap="18%">
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="currentColor" opacity={0.12} />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis
                tick={{ fontSize: 9 }}
                width={36}
                axisLine={false}
                tickLine={false}
                domain={yDomain}
                tickFormatter={(v) => fmtM(v, fineChart)}
              />
              <Tooltip
                content={<CompareTooltip fine={fineChart} />}
                cursor={{ fill: "hsl(var(--muted))", opacity: 0.15 }}
              />
              <Bar dataKey="Actual" fill={COLORS.actual} radius={[3, 3, 0, 0]} maxBarSize={14} />
              <Bar dataKey="Forecast" fill={forecastColor} radius={[3, 3, 0, 0]} maxBarSize={14} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="flex gap-4 mt-1 text-[10px]">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: COLORS.actual }} />
            Actual
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: forecastColor }} />
            Forecast
          </span>
        </div>
      </div>
    </div>
  )
}

export function Partner2025CompareCharts({ partnerCode, sector }: { partnerCode: string; sector: string }) {
  const [exportSeries, setExportSeries] = useState<PartnerMonthlySeries | null>(null)
  const [importSeries, setImportSeries] = useState<PartnerMonthlySeries | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    const base = `${getApiBaseUrl()}/api/partner-monthly-series?sector=${sector}&partner=${partnerCode}`
    Promise.all([
      fetch(`${base}&flow=export`, { ...apiFetchInit, signal: AbortSignal.timeout(20000) }),
      fetch(`${base}&flow=import`, { ...apiFetchInit, signal: AbortSignal.timeout(20000) }),
    ])
      .then(async ([expRes, impRes]) => {
        if (!expRes.ok || !impRes.ok) throw new Error("series fetch failed")
        return Promise.all([expRes.json(), impRes.json()] as const)
      })
      .then(([exp, imp]) => {
        if (!cancelled) {
          setExportSeries(exp)
          setImportSeries(imp)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setExportSeries(null)
          setImportSeries(null)
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [partnerCode, sector])

  if (loading) {
    return <p className="text-xs text-muted-foreground animate-pulse">Loading 2025 comparison…</p>
  }
  if (!exportSeries || !importSeries) {
    return <p className="text-xs text-muted-foreground">2025 monthly comparison unavailable.</p>
  }

  return (
    <div className="mt-4 pt-3 border-t border-border/60 space-y-3">
      <p className="text-xs font-semibold">2025 — Actual vs Forecast</p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <FlowComparePanel
          title="Export (India → partner)"
          series={exportSeries}
          forecastColor={COLORS.forecastExport}
        />
        <FlowComparePanel
          title="Import (partner → India)"
          series={importSeries}
          forecastColor={COLORS.forecastImport}
        />
      </div>
    </div>
  )
}
