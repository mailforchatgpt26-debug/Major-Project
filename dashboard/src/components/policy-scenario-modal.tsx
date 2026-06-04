"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { useDashboardStore } from "./dashboard/store"
import { pharmaNationalExportShare } from "@/lib/pharma-constants"
import {
  X,
  Play,
  TrendingDown,
  TrendingUp,
  ArrowRight,
  ChevronDown,
  FlaskConical,
  Search,
  ChevronRight,
} from "lucide-react"

// ─── helpers ────────────────────────────────────────────────────────────────

// All trade values from the backend are in USD millions (raw USD ÷ 1e6 in preprocessing)
function fmt(v: number) {
  const abs = Math.abs(v)
  if (abs >= 1_000) return `$${(v / 1_000).toFixed(2)}B`
  if (abs >= 1)     return `$${v.toFixed(1)}M`
  if (abs >= 0.001) return `$${(v * 1_000).toFixed(0)}K`
  return `< $1K`
}

const FEATURES = [
  {
    key: "gdp",
    icon: "📈",
    label: "Economic Output",
    sublabel: "GDP grows or shrinks",
    hint: "Tests how a change in the partner country's total economic size reshapes bilateral trade — captured via the gravity model's mass parameter.",
  },
  {
    key: "sentiment",
    icon: "🤝",
    label: "Diplomatic Climate",
    sublabel: "Relations improve or worsen",
    hint: "Reflects shifts in bilateral news sentiment scored by FinBERT. The GNN encodes this as an edge weight — warmer relations reduce effective trade friction.",
  },
  {
    key: "tariff",
    icon: "🚧",
    label: "Trade Barriers",
    sublabel: "Tariffs rise or fall",
    hint: "Proxied via a log-space shift in the bilateral lag features. Higher barriers compress the effective trade volume signal the GNN sees in its input.",
  },
]

// ─── sub-components ──────────────────────────────────────────────────────────

function FeatureHint({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="text-xs text-muted-foreground">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 hover:text-foreground transition-colors"
      >
        <ChevronDown className={`size-3 transition-transform ${open ? "rotate-180" : ""}`} />
        How does the model use this?
      </button>
      {open && (
        <p className="mt-1.5 pl-4 leading-relaxed border-l border-border">{text}</p>
      )}
    </div>
  )
}

function ImpactBar({ pct }: { pct: number }) {
  const clamped = Math.max(-100, Math.min(100, pct))
  const isPos = clamped >= 0
  return (
    <div className="w-full h-2 rounded-full bg-muted overflow-hidden relative">
      <div className="absolute inset-y-0 left-1/2 w-px bg-border z-10" />
      {isPos ? (
        <div
          className="absolute inset-y-0 left-1/2 rounded-full bg-green-500 transition-all duration-500"
          style={{ width: `${(clamped / 100) * 50}%` }}
        />
      ) : (
        <div
          className="absolute inset-y-0 right-1/2 rounded-full bg-red-500 transition-all duration-500"
          style={{ width: `${(Math.abs(clamped) / 100) * 50}%` }}
        />
      )}
    </div>
  )
}

/** Inline country picker shown inside the modal */
function CountryPicker({
  predictions,
  value,
  onChange,
}: {
  predictions: { partnerCode: string; partner: string; export_forecast: number }[]
  value: string | undefined
  onChange: (code: string) => void
}) {
  const [query, setQuery] = useState("")
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    // auto-focus search when picker mounts
    inputRef.current?.focus()
  }, [])

  const filtered = predictions.filter(
    (p) =>
      p.partner.toLowerCase().includes(query.toLowerCase()) ||
      p.partnerCode.toLowerCase().includes(query.toLowerCase()),
  )

  return (
    <div className="space-y-2">
      <div className="relative">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground" />
        <input
          ref={inputRef}
          type="text"
          placeholder="Search country…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full pl-8 pr-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-1 focus:ring-primary"
        />
      </div>
      <div className="max-h-48 overflow-y-auto rounded-lg border divide-y divide-border">
        {filtered.length === 0 && (
          <p className="px-3 py-3 text-xs text-muted-foreground text-center">No matches</p>
        )}
        {filtered.map((p) => (
          <button
            key={p.partnerCode}
            onClick={() => onChange(p.partnerCode)}
            className={`w-full flex items-center justify-between px-3 py-2 text-left text-sm hover:bg-muted transition-colors ${
              value === p.partnerCode ? "bg-primary/5 text-primary font-semibold" : ""
            }`}
          >
            <span>{p.partner}</span>
            <span className="text-xs text-muted-foreground tabular-nums">
              {fmt(p.export_forecast)}
            </span>
          </button>
        ))}
      </div>
      <p className="text-[10px] text-muted-foreground text-right">
        Forecast value shown · select to analyse
      </p>
    </div>
  )
}

// ─── main modal ──────────────────────────────────────────────────────────────

export function PolicyScenarioModal() {
  const {
    selectedPartner,
    predictions,
    selectPartner,
    runSimulation,
    simulationResult,
    simulationError,
    loading,
    sector,
  } = useDashboardStore()

  const [open, setOpen] = useState(false)
  const [feature, setFeature] = useState("gdp")
  const [change, setChange] = useState(20)

  // Default shock direction matches economic meaning of each lever
  useEffect(() => {
    if (feature === "tariff" || feature === "sentiment") {
      setChange(-20)
    } else {
      setChange(20)
    }
  }, [feature])
  // local override — user can pick a country inside the modal without affecting the table
  const [localPartner, setLocalPartner] = useState<string | undefined>(undefined)
  const [showPicker, setShowPicker] = useState(false)

  const close = useCallback(() => setOpen(false), [])

  // sync local partner when modal opens
  useEffect(() => {
    if (open) {
      setLocalPartner(selectedPartner)
      // if nothing pre-selected, show the picker automatically
      setShowPicker(!selectedPartner)
    }
  }, [open, selectedPartner])

  // close on Escape
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => e.key === "Escape" && close()
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [open, close])

  const activePartner = localPartner ?? selectedPartner
  const partnerName =
    predictions.find((p) => p.partnerCode === activePartner)?.partner ??
    activePartner ??
    ""

  const featureMeta = FEATURES.find((f) => f.key === feature)!

  const handlePickCountry = (code: string) => {
    setLocalPartner(code)
    selectPartner(code)   // also sync with table highlight
    setShowPicker(false)
  }

  const handleRun = () => {
    if (!activePartner) return
    runSimulation(activePartner, feature, change)
  }

  const r = simulationResult
  // Use the predictions table's export_forecast as the baseline — same number shown in the table.
  const predictionForecast = predictions.find((p) => p.partnerCode === activePartner)?.export_forecast
  const displayBaseline = predictionForecast ?? r?.baseline ?? 0
  const displayCounterfactual = r ? displayBaseline * (1 + r.pct_impact / 100) : 0
  const displayDelta = r ? Math.abs(displayCounterfactual - displayBaseline) : 0
  const tradeDir = r && r.pct_impact >= 0 ? "more" : "less"

  const partnerPred = predictions.find((p) => p.partnerCode === selectedPartner)
  const portfolioForecast = predictions.reduce((s, p) => s + p.export_forecast, 0)
  const partnerShareFrontend =
    sector === "pharma"
      ? pharmaNationalExportShare(
          displayBaseline,
          2025,
          partnerPred?.export_actual ?? null
        )
      : portfolioForecast > 0
      ? displayBaseline / portfolioForecast
      : 0

  // ── floating trigger button ──
  const trigger = (
    <div className="fixed bottom-6 right-6 z-40 flex flex-col items-end gap-1.5">
      {!selectedPartner && !open && (
        <div className="px-3 py-1.5 rounded-lg bg-card border shadow-sm text-xs text-muted-foreground max-w-[180px] text-right animate-in fade-in slide-in-from-bottom-1">
          Pick any country to run a scenario — or open and search inside ↓
        </div>
      )}
      <button
        id="open-scenario-engine"
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 px-4 py-3 rounded-full bg-primary text-primary-foreground text-sm font-semibold shadow-lg shadow-primary/30 hover:scale-105 active:scale-95 transition-transform"
      >
        <FlaskConical className="size-4" />
        Policy Scenario Engine
        {!selectedPartner && (
          <span className="size-2 rounded-full bg-amber-400 animate-pulse" />
        )}
      </button>
    </div>
  )

  if (!open) return trigger

  return (
    <>
      {trigger}

      {/* backdrop */}
      <div
        className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm animate-in fade-in"
        onClick={close}
      />

      {/* modal */}
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
        <div
          className="pointer-events-auto w-full max-w-3xl max-h-[90dvh] overflow-y-auto rounded-2xl border bg-card shadow-2xl animate-in fade-in slide-in-from-bottom-4"
          onClick={(e) => e.stopPropagation()}
        >
          {/* modal header */}
          <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-4 border-b bg-card/95 backdrop-blur-sm">
            <div className="flex items-center gap-3">
              <FlaskConical className="size-5 text-primary" />
              <div>
                <h2 className="text-base font-bold">Policy Scenario Engine</h2>
                <p className="text-xs text-muted-foreground">
                  GNN-grounded counterfactual analysis ·{" "}
                  {sector === "pharma" ? "India pharmaceuticals" : `India ${sector}`}
                </p>
              </div>
            </div>
            <button
              onClick={close}
              className="p-1.5 rounded-lg hover:bg-muted transition-colors"
              aria-label="Close"
            >
              <X className="size-4" />
            </button>
          </div>

          {/* modal body */}
          <div className="grid md:grid-cols-2 gap-0 divide-y md:divide-y-0 md:divide-x divide-border">

            {/* ── LEFT: controls ── */}
            <div className="p-6 space-y-5">

              {/* country selector */}
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  Step 1 — Choose a country
                </p>

                {activePartner && !showPicker ? (
                  <button
                    onClick={() => setShowPicker(true)}
                    className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg bg-primary/5 border border-primary/20 hover:bg-primary/10 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-primary">{partnerName}</span>
                      <span className="text-xs text-muted-foreground">
                        ({activePartner})
                      </span>
                    </div>
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      Change
                      <ChevronRight className="size-3" />
                    </div>
                  </button>
                ) : (
                  <CountryPicker
                    predictions={predictions}
                    value={activePartner}
                    onChange={handlePickCountry}
                  />
                )}
              </div>

              {/* feature selection */}
              <div className="space-y-2">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  Step 2 — Choose what changes
                </p>
                <div className="grid grid-cols-2 gap-2">
                  {FEATURES.map((f) => (
                    <button
                      key={f.key}
                      onClick={() => setFeature(f.key)}
                      className={`px-3 py-2.5 rounded-xl border text-left transition-all ${
                        feature === f.key
                          ? "bg-primary text-primary-foreground border-primary shadow-sm"
                          : "bg-background hover:bg-muted border-border"
                      }`}
                    >
                      <span className="text-lg leading-none block mb-1">{f.icon}</span>
                      <span className="text-xs font-semibold block">{f.label}</span>
                      <span
                        className={`text-[10px] block mt-0.5 ${
                          feature === f.key
                            ? "text-primary-foreground/70"
                            : "text-muted-foreground"
                        }`}
                      >
                        {f.sublabel}
                      </span>
                    </button>
                  ))}
                </div>
                <FeatureHint text={featureMeta.hint} />
              </div>

              {/* magnitude */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                    Step 3 — By how much?
                  </p>
                  <span
                    className={`text-sm font-bold tabular-nums ${
                      change < 0 ? "text-red-500" : "text-green-500"
                    }`}
                  >
                    {change > 0 ? "+" : ""}
                    {change}%
                  </span>
                </div>
                <input
                  type="range"
                  min="-50"
                  max="50"
                  step="5"
                  value={change}
                  onChange={(e) => setChange(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                />
                <div className="flex justify-between text-[10px] text-muted-foreground font-medium">
                  <span>−50% (severe decline)</span>
                  <span>+50% (strong growth)</span>
                </div>
              </div>

              <button
                onClick={handleRun}
                disabled={loading.simulation || !activePartner}
                className="w-full py-3 rounded-xl bg-primary text-primary-foreground font-bold shadow-md hover:scale-[1.02] active:scale-[0.99] transition-transform flex items-center justify-center gap-2 disabled:opacity-50 disabled:scale-100 disabled:cursor-not-allowed"
              >
                {loading.simulation ? (
                  <>
                    <div className="size-4 border-2 border-primary-foreground/30 border-t-primary-foreground animate-spin rounded-full" />
                    Simulating on GNN graph…
                  </>
                ) : (
                  <>
                    <Play className="size-4 fill-current" />
                    Run Scenario Analysis
                  </>
                )}
              </button>
            </div>

            {/* ── RIGHT: results ── */}
            <div className="p-6 space-y-5">
              {!r ? (
                <div className="flex flex-col items-center justify-center h-full min-h-[300px] text-center text-muted-foreground gap-4">
                  <div className="size-16 rounded-full bg-muted/40 flex items-center justify-center">
                    <FlaskConical className="size-7 opacity-30" />
                  </div>
                  {simulationError ? (
                    <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-left max-w-sm">
                      <p className="text-sm font-medium text-destructive">Simulation failed</p>
                      <p className="text-xs mt-1 text-muted-foreground">{simulationError}</p>
                      <p className="text-[10px] mt-2 opacity-70">
                        If the API just started, wait until /health shows model_loaded: true, then try again.
                      </p>
                    </div>
                  ) : (
                  <div>
                    <p className="text-sm font-medium">No scenario run yet</p>
                    <p className="text-xs mt-1 opacity-60">
                      Choose a country and variable, then click Run
                    </p>
                  </div>
                  )}
                  <div className="text-left space-y-2 w-full max-w-xs mt-2">
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                      What makes this different?
                    </p>
                    {[
                      "Each scenario patches the live GNN graph — not a formula",
                      "The model accounts for network effects through connected countries",
                      "Gravity model + GAT layers capture non-linear trade relationships",
                    ].map((t) => (
                      <div key={t} className="flex items-start gap-2 text-xs text-muted-foreground">
                        <span className="text-primary mt-0.5">→</span>
                        <span>{t}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="space-y-4 animate-in fade-in slide-in-from-right-2">
                  <div>
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
                      Scenario outcome
                    </p>
                    <p className="text-sm font-semibold leading-snug">
                      {feature === "gdp" &&
                        `${partnerName}'s economy ${change >= 0 ? "grows" : "shrinks"} by ${Math.abs(change)}%`}
                      {feature === "sentiment" &&
                        `Diplomatic climate with ${partnerName} ${change >= 0 ? "improves" : "deteriorates"} by ${Math.abs(change)}%`}
                      {feature === "tariff" &&
                        `${partnerName} ${change >= 0 ? "raises" : "lowers"} trade barriers by ${Math.abs(change)}%`}
                    </p>
                  </div>

                  {/* before → after */}
                  <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2">
                    <div className="p-3 rounded-xl bg-muted/40 border text-center">
                      <div className="text-[10px] text-muted-foreground font-bold uppercase mb-1.5">
                        2025 Forecast
                      </div>
                      <div className="text-xl font-bold">{fmt(displayBaseline)}</div>
                      <div className="text-[10px] text-muted-foreground mt-0.5">
                        without this change
                      </div>
                    </div>
                    <ArrowRight
                      className={`size-5 shrink-0 ${
                        r.pct_impact >= 0 ? "text-green-500" : "text-red-500"
                      }`}
                    />
                    <div
                      className={`p-3 rounded-xl border text-center ${
                        r.pct_impact >= 0
                          ? "bg-green-500/5 border-green-500/25"
                          : "bg-red-500/5 border-red-500/25"
                      }`}
                    >
                      <div
                        className={`text-[10px] font-bold uppercase mb-1.5 ${
                          r.pct_impact >= 0
                            ? "text-green-600 dark:text-green-400"
                            : "text-red-500"
                        }`}
                      >
                        Simulated
                      </div>
                      <div className="text-xl font-bold">{fmt(displayCounterfactual)}</div>
                      <div
                        className={`text-[10px] mt-0.5 font-semibold ${
                          r.pct_impact >= 0
                            ? "text-green-600 dark:text-green-400"
                            : "text-red-500"
                        }`}
                      >
                        {r.pct_impact >= 0 ? "+" : ""}
                        {r.pct_impact.toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {/* impact bar */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[10px] text-muted-foreground">
                      <span>Large decline</span>
                      <span>No change</span>
                      <span>Large growth</span>
                    </div>
                    <ImpactBar pct={r.pct_impact} />
                  </div>

                  {/* insight cards */}
                  <div className="space-y-2">
                    <div
                      className={`flex gap-3 p-3 rounded-xl border ${
                        r.pct_impact >= 0
                          ? "bg-green-500/5 border-green-500/20"
                          : "bg-red-500/5 border-red-500/20"
                      }`}
                    >
                      {r.pct_impact >= 0 ? (
                        <TrendingUp className="size-4 text-green-500 mt-0.5 shrink-0" />
                      ) : (
                        <TrendingDown className="size-4 text-red-500 mt-0.5 shrink-0" />
                      )}
                      <div>
                        <p className="text-sm font-bold">
                          India exports{" "}
                          <span
                            className={
                              r.pct_impact >= 0
                                ? "text-green-600 dark:text-green-400"
                                : "text-red-500"
                            }
                          >
                            {fmt(displayDelta)} {tradeDir}
                          </span>{" "}
                          to {partnerName}
                        </p>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          {sector === "pharma" ? "Pharma" : sector} exports:{" "}
                          {fmt(displayBaseline)} → {fmt(displayCounterfactual)} per year
                        </p>
                      </div>
                    </div>

                    <div className="flex gap-3 p-3 rounded-xl bg-muted/20 border">
                      <span className="text-base shrink-0 mt-0.5">🌍</span>
                      <div className="space-y-1">
                        <p className="text-xs font-semibold">Global portfolio effect</p>
                        <p className="text-xs text-muted-foreground">
                          {partnerName} accounts for{" "}
                          <span className="font-semibold text-foreground">
                            {(partnerShareFrontend * 100).toFixed(1)}%
                          </span>{" "}
                          of India&apos;s{" "}
                          {sector === "pharma" ? "national pharma" : `forecasted ${sector}`} exports.
                          This scenario shifts that portfolio by{" "}
                          <span className="font-semibold text-foreground">
                            {Math.abs(r.pct_impact * partnerShareFrontend).toFixed(2)}%
                          </span>
                          {partnerShareFrontend < 0.005 && (
                            <span className="text-muted-foreground/70">
                              {" "}— small because {partnerName} is a minor market for India
                            </span>
                          )}
                          .
                        </p>
                      </div>
                    </div>
                  </div>

                  <p className="text-[10px] text-muted-foreground/40 text-center pt-1">
                    Powered by Gravity-Informed GNN · Scenario modelled on 2024 trade graph · Annual projection
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
