"use client"

import React, { useMemo, useState } from "react"
import type { Prediction, Explainability } from "@/lib/types"
import { useDashboardStore } from "@/components/dashboard/store"
import { Partner2025CompareCharts } from "@/components/partner-2025-compare-chart"

const FORECAST_YEARS = ["2025", "2026", "2027", "2028", "2029", "2030"]

type Props = {
  data: Prediction[]
  selectedPartner?: string
  explainability?: Explainability
  onRowSelect: (partnerCode: string) => void
}

type SortKey = "partner" | "export_actual" | "export_forecast" | "import_actual" | "import_forecast" | "confidence"

function fmtVal(v?: number | null): React.ReactNode {
  if (v == null || v <= 0) return <span className="text-muted-foreground/40">—</span>
  if (v < 0.005) return <span className="text-muted-foreground/60">&lt;&nbsp;1</span>
  if (v < 1) return <>{v.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 2 })}</>
  return <>{v.toLocaleString(undefined, { maximumFractionDigits: 0 })}</>
}

function fmtChange(c: number | undefined) {
  if (c == null || !Number.isFinite(c)) {
    return <span className="text-muted-foreground/40">—</span>
  }
  const color = c >= 0 ? "var(--color-chart-1)" : "var(--destructive)"
  const sign = c >= 0 ? "+" : ""
  return <span style={{ color }}>{sign}{(c * 100).toFixed(1)}%</span>
}

function fmtImportYoy(c: number, importForecast?: number | null) {
  if (importForecast == null || importForecast < 0.005 || !Number.isFinite(c)) {
    return <span className="text-muted-foreground/40">—</span>
  }
  const pct = c * 100
  // Huge percentages usually come from tiny prior-year base; cap for readability.
  if (Math.abs(pct) > 500) {
    const color = c >= 0 ? "var(--color-chart-1)" : "var(--destructive)"
    return (
      <span style={{ color }} title="Large YoY due to low base in previous year">
        {c >= 0 ? "+" : "−"}500%+
      </span>
    )
  }
  return fmtChange(c)
}

function MiniBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-muted rounded-full h-1.5 min-w-[60px]">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${Math.max(value * 100, 2)}%` }} />
      </div>
      <span className="text-xs text-muted-foreground w-7 text-right">{Math.round(value * 100)}%</span>
    </div>
  )
}

function InlineExplainability({
  explainability,
  partnerName,
  partnerCode,
  showCompare2025,
  colSpan,
  seasonalityLabel,
  peakMonth,
  lowMonth,
}: {
  explainability?: Explainability
  partnerName: string
  partnerCode: string
  showCompare2025?: boolean
  colSpan: number
  seasonalityLabel?: string
  peakMonth?: string
  lowMonth?: string
}) {
  const { sector } = useDashboardStore()

  if (!explainability) {
    return (
      <td colSpan={colSpan} className="px-4 py-3 bg-accent/20 border-b">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="animate-pulse">Loading explanation for {partnerName}…</span>
        </div>
      </td>
    )
  }

  return (
    <td colSpan={colSpan} className="px-4 py-4 bg-accent/20 border-b">
      <p className="text-xs font-semibold mb-3 text-foreground">Why this prediction?</p>
      <div className="grid grid-cols-1 gap-y-1">
        {(peakMonth || lowMonth) && (
          <div className="mb-2">
            <p className="text-xs text-muted-foreground">{seasonalityLabel ?? "Seasonality"}</p>
            <p className="text-xs mt-0.5">
              <span className="text-muted-foreground">Peak:</span> <span className="font-medium">{peakMonth ?? "—"}</span>
              <span className="mx-2 text-muted-foreground">·</span>
              <span className="text-muted-foreground">Low:</span> <span className="font-medium">{lowMonth ?? "—"}</span>
            </p>
          </div>
        )}
        <div>
          <p className="text-xs text-muted-foreground mb-2">What drives this forecast</p>
          {(explainability.features ?? []).map((f) => (
            <div key={f.feature} className="flex items-center gap-2 mb-1.5">
              <span className="text-xs w-28 truncate">{f.feature}</span>
              <MiniBar value={f.importance} color="bg-chart-2" />
            </div>
          ))}
        </div>
      </div>
      <p className="text-xs text-muted-foreground mt-3 leading-relaxed">{explainability.blurb}</p>
      {showCompare2025 ? (
        <Partner2025CompareCharts partnerCode={partnerCode} sector={sector} />
      ) : null}
    </td>
  )
}

export function PredictionsTable({ data, selectedPartner, explainability, onRowSelect }: Props) {
  const { month, setMonth } = useDashboardStore()
  const forecastYear = (() => {
    const y = month.split("-")[0]
    return FORECAST_YEARS.includes(y) ? y : FORECAST_YEARS[0]
  })()
  const showYoYChange = forecastYear !== "2025"
  const exportColSpan = showYoYChange ? 2 : 1
  const importColSpan = showYoYChange ? 2 : 1
  const tableColCount = 1 + exportColSpan + importColSpan

  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({
    key: "export_forecast",
    dir: "desc",
  })

  const sorted = useMemo(() => {
    const arr = [...data]
    arr.sort((a, b) => {
      const key = sort.key
      const va =
        key === "partner" ? a.partner :
        key === "export_actual" ? (a.export_actual ?? 0) :
        key === "export_forecast" ? a.export_forecast :
        key === "import_actual" ? (a.import_actual ?? 0) :
        key === "import_forecast" ? (a.import_forecast ?? 0) :
        a.confidence
      const vb =
        key === "partner" ? b.partner :
        key === "export_actual" ? (b.export_actual ?? 0) :
        key === "export_forecast" ? b.export_forecast :
        key === "import_actual" ? (b.import_actual ?? 0) :
        key === "import_forecast" ? (b.import_forecast ?? 0) :
        b.confidence
      if (typeof va === "string" && typeof vb === "string")
        return sort.dir === "asc" ? va.localeCompare(vb) : vb.localeCompare(va)
      return sort.dir === "asc" ? (va as number) - (vb as number) : (vb as number) - (va as number)
    })
    return arr
  }, [data, sort])

  function handleSort(key: SortKey) {
    setSort((s) => s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" })
  }

  return (
    <div className="p-3">
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-2 gap-4 flex-wrap">
        <div>
          <h3 className="text-sm font-semibold">Bilateral Trade Predictions</h3>
          <span className="text-xs text-muted-foreground">Values in USD millions — click a row for explanation</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground">Forecast year:</span>
          <div className="flex gap-1">
            {FORECAST_YEARS.map((y) => (
              <button
                key={y}
                onClick={() => setMonth(`${y}-01`)}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                  forecastYear === y ? "bg-primary text-primary-foreground" : "border hover:bg-accent"
                }`}
              >
                {y}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm table-fixed">
          <colgroup>
            <col className={showYoYChange ? "w-[34%]" : "w-[42%]"} />
            <col className={showYoYChange ? "w-[16%]" : "w-[29%]"} />
            {showYoYChange ? <col className="w-[16%]" /> : null}
            <col className={showYoYChange ? "w-[17%]" : "w-[29%]"} />
            {showYoYChange ? <col className="w-[17%]" /> : null}
          </colgroup>
          <thead className="bg-secondary/60">
            <tr>
              <Th onClick={() => handleSort("partner")} active={sort.key === "partner"} dir={sort.dir}>
                Country
              </Th>
              <th
                colSpan={exportColSpan}
                className="px-3 py-1 text-center text-xs font-semibold text-green-600 dark:text-green-400 border-l border-border"
              >
                India → Partner (Export)
              </th>
              <th
                colSpan={importColSpan}
                className="px-3 py-1 text-center text-xs font-semibold text-blue-600 dark:text-blue-400 border-l border-border"
              >
                Partner → India (Import)
              </th>
            </tr>
            <tr className="text-xs">
              <th className="px-3 pb-2 text-left font-normal text-muted-foreground" />
              <Th onClick={() => handleSort("export_forecast")} active={sort.key === "export_forecast"} dir={sort.dir} className="border-l border-border text-right">
                {forecastYear} Fcst
              </Th>
              {showYoYChange ? (
                <th className="px-3 pb-2 text-right font-normal text-muted-foreground">YoY %</th>
              ) : null}
              <Th onClick={() => handleSort("import_forecast")} active={sort.key === "import_forecast"} dir={sort.dir} className="border-l border-border text-right">
                {forecastYear} Fcst
              </Th>
              {showYoYChange ? (
                <th className="px-3 pb-2 text-right font-normal text-muted-foreground">YoY %</th>
              ) : null}
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={tableColCount} className="px-3 py-8 text-center text-muted-foreground">
                  No predictions available.
                </td>
              </tr>
            ) : (
              sorted.map((row) => {
                const isSelected = selectedPartner === row.partnerCode
                return (
                  <React.Fragment key={row.partnerCode}>
                    <tr
                      className={`cursor-pointer border-b hover:bg-accent ${isSelected ? "bg-accent font-medium" : ""}`}
                      onClick={() => onRowSelect(row.partnerCode)}
                    >
                      <td className="px-3 py-2">
                        <span className="flex items-center gap-1.5">
                          {isSelected && <span className="text-primary">▼</span>}
                          {row.partner}
                        </span>
                      </td>
                      <td className="px-3 py-2 font-medium text-green-700 dark:text-green-400 border-l border-border text-right tabular-nums">
                        {fmtVal(row.export_forecast)}
                      </td>
                      {showYoYChange ? (
                        <td className="px-3 py-2 text-right tabular-nums">
                          {fmtChange(row.export_change)}
                        </td>
                      ) : null}
                      <td className="px-3 py-2 font-medium text-blue-700 dark:text-blue-400 border-l border-border text-right tabular-nums">
                        {fmtVal(row.import_forecast)}
                      </td>
                      {showYoYChange ? (
                        <td className="px-3 py-2 text-right tabular-nums">
                          {fmtImportYoy(row.import_change, row.import_forecast)}
                        </td>
                      ) : null}
                    </tr>
                    {isSelected && (
                      <tr>
                        <InlineExplainability
                          explainability={explainability}
                          partnerName={row.partner}
                          partnerCode={row.partnerCode}
                          showCompare2025={forecastYear === "2025"}
                          colSpan={tableColCount}
                          seasonalityLabel={forecastYear === "2025" ? "2025 forecast seasonality" : `${forecastYear} seasonality (historical pattern)`}
                          peakMonth={row.export_peak_month}
                          lowMonth={row.export_low_month}
                        />
                      </tr>
                    )}
                  </React.Fragment>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Th({
  children, active, dir, onClick, className = "",
}: {
  children: React.ReactNode
  active?: boolean
  dir?: "asc" | "desc"
  onClick?: () => void
  className?: string
}) {
  return (
    <th
      role="columnheader"
      scope="col"
      onClick={onClick}
      className={`px-3 py-2 font-medium select-none ${className}`}
      aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button className={`inline-flex items-center gap-1 hover:underline w-full ${className.includes("text-right") ? "justify-end" : ""}`}>
        {children}
        {active ? <span aria-hidden>{dir === "asc" ? "↑" : "↓"}</span> : null}
      </button>
    </th>
  )
}
