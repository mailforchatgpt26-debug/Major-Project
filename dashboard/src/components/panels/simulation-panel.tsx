"use client"

import { useState } from "react"
import { useDashboardStore } from "../dashboard/store"
import { Play, TrendingDown, TrendingUp, Activity, ArrowRight, Zap } from "lucide-react"

const FEATURES = [
  {
    key: "gdp",
    label: "Economy Size",
    icon: "📈",
    description: "What if their economy grows or shrinks?",
  },
  {
    key: "sentiment",
    label: "Trade Relations",
    icon: "🤝",
    description: "What if political/trade relations improve or worsen?",
  },
  {
    key: "tariff",
    label: "Import Barriers",
    icon: "🚧",
    description: "What if they raise or lower trade barriers on Indian goods?",
  },
  {
    key: "fta",
    label: "Trade Deal",
    icon: "📜",
    description: "What if India signs or loses a free trade agreement?",
  },
]

function fmt(v: number) {
  const abs = Math.abs(v)
  if (abs >= 1_000_000_000) return `$${(v / 1_000_000_000).toFixed(2)}B`
  if (abs >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`
  if (abs >= 1_000) return `$${(v / 1_000).toFixed(1)}K`
  return `$${v.toFixed(0)}`
}

function SliderLabel({ change, feature }: { change: number; feature: string }) {
  if (feature === "fta") {
    return (
      <span className="text-xs text-muted-foreground">
        {change >= 0 ? "Sign new deal" : "Remove existing deal"}
      </span>
    )
  }
  if (change === 0) return <span className="text-xs text-muted-foreground">No change</span>
  const dir = change > 0 ? "increase" : "decrease"
  const abs = Math.abs(change)
  const labels: Record<string, string[]> = {
    gdp: ["economy shrinks", "economy grows"],
    sentiment: ["relations worsen", "relations improve"],
    tariff: ["barriers lowered", "barriers raised"],
  }
  const [neg, pos] = labels[feature] ?? ["decreases", "increases"]
  return (
    <span className="text-xs text-muted-foreground">
      Their {dir === "increase" ? pos : neg} by <strong>{abs}%</strong>
    </span>
  )
}

function buildExplanation(
  partnerName: string,
  feature: string,
  change: number,
  baseline: number,
  counterfactual: number,
  pctImpact: number,
  globalImpact: number,
  sector: string,
) {
  const dir = change > 0 ? "increases" : "decreases"
  const abs = Math.abs(change)
  const tradeDir = pctImpact >= 0 ? "more" : "less"
  const absPct = Math.abs(pctImpact)
  const delta = Math.abs(counterfactual - baseline)
  const sectorLabel = sector === "pharma" ? "pharmaceutical" : sector

  const scenarioLine: Record<string, string> = {
    gdp: `If ${partnerName}'s economy ${dir} by ${abs}%`,
    sentiment: `If trade relations with ${partnerName} ${change > 0 ? "improve" : "worsen"} by ${abs}%`,
    tariff: `If ${partnerName} ${change > 0 ? "raises" : "lowers"} import barriers by ${abs}%`,
    fta: change > 0
      ? `If India signs a Free Trade Agreement with ${partnerName}`
      : `If India's trade deal with ${partnerName} is removed`,
  }

  const headline = scenarioLine[feature] ?? `If ${feature} changes by ${change}%`

  return {
    headline,
    summary: `India would export ${fmt(delta)} ${tradeDir} in ${sectorLabel} goods to ${partnerName}.`,
    detail: `That's a ${absPct.toFixed(1)}% ${pctImpact >= 0 ? "increase" : "decrease"} — from ${fmt(baseline)} to ${fmt(counterfactual)} per year.`,
    global: `This would shift India's overall ${sectorLabel} exports by ${Math.abs(globalImpact).toFixed(2)}% globally.`,
  }
}

export function SimulationPanel() {
  const { selectedPartner, predictions, runSimulation, simulationResult, loading, sector } =
    useDashboardStore()

  const [change, setChange] = useState(20)
  const [feature, setFeature] = useState("gdp")

  const partnerName =
    predictions.find((p) => p.partnerCode === selectedPartner)?.partner ??
    selectedPartner ??
    ""

  const handleSimulate = () => {
    if (!selectedPartner) return
    // FTA: use +100 to signal "activate", -100 to signal "remove"
    const effectiveChange = feature === "fta" ? (change >= 0 ? 100 : -100) : change
    runSimulation(selectedPartner, feature, effectiveChange)
  }

  if (!selectedPartner) {
    return (
      <div className="flex flex-col items-center justify-center p-8 text-center h-full text-muted-foreground gap-3">
        <Activity className="size-10 opacity-20" />
        <p className="text-sm font-medium">Select a country from the table</p>
        <p className="text-xs opacity-60">Then run a "what-if" simulation to see how changes affect trade</p>
      </div>
    )
  }

  const featureMeta = FEATURES.find((f) => f.key === feature)!
  const result = simulationResult
  const explanation = result
    ? buildExplanation(
        partnerName,
        feature,
        change,
        result.baseline,
        result.counterfactual,
        result.pct_impact,
        result.global_impact,
        sector,
      )
    : null

  return (
    <div className="flex flex-col h-full p-4 space-y-5 overflow-y-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-0.5">
          <Zap className="size-4 text-primary" />
          <h3 className="text-sm font-bold">What-If Trade Simulator</h3>
        </div>
        <p className="text-xs text-muted-foreground">
          Ask "what happens to India-{partnerName} trade if…"
        </p>
      </div>

      {/* Country badge */}
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-primary/5 border border-primary/20">
        <span className="text-xs text-muted-foreground font-medium">Testing scenario for</span>
        <span className="text-sm font-bold text-primary">{partnerName}</span>
      </div>

      {/* Feature picker */}
      <div className="space-y-2">
        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
          What changes?
        </p>
        <div className="grid grid-cols-2 gap-2">
          {FEATURES.map((f) => (
            <button
              key={f.key}
              onClick={() => setFeature(f.key)}
              className={`px-3 py-2.5 rounded-lg border text-left transition-all ${
                feature === f.key
                  ? "bg-primary text-primary-foreground border-primary shadow-sm"
                  : "bg-background hover:bg-muted border-border"
              }`}
            >
              <div className="text-base leading-none mb-1">{f.icon}</div>
              <div className="text-xs font-semibold">{f.label}</div>
            </button>
          ))}
        </div>
        <p className="text-[11px] text-muted-foreground italic">{featureMeta.description}</p>
      </div>

      {/* Slider — hide for FTA (binary) */}
      {feature !== "fta" ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              By how much?
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
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span>−50%</span>
            <SliderLabel change={change} feature={feature} />
            <span>+50%</span>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            Scenario
          </p>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => setChange(1)}
              className={`py-2 rounded-lg border text-xs font-medium transition-all ${
                change >= 0
                  ? "bg-green-500/10 border-green-500/30 text-green-600 dark:text-green-400"
                  : "bg-background hover:bg-muted"
              }`}
            >
              ✅ Sign new deal
            </button>
            <button
              onClick={() => setChange(-1)}
              className={`py-2 rounded-lg border text-xs font-medium transition-all ${
                change < 0
                  ? "bg-red-500/10 border-red-500/30 text-red-600 dark:text-red-400"
                  : "bg-background hover:bg-muted"
              }`}
            >
              ❌ Remove deal
            </button>
          </div>
        </div>
      )}

      {/* Run button */}
      <button
        onClick={handleSimulate}
        disabled={loading.simulation}
        className="w-full py-3 rounded-xl bg-primary text-primary-foreground font-bold shadow-md hover:scale-[1.02] active:scale-[0.99] transition-transform flex items-center justify-center gap-2 disabled:opacity-50 disabled:scale-100"
      >
        {loading.simulation ? (
          <div className="size-4 border-2 border-primary-foreground/30 border-t-primary-foreground animate-spin rounded-full" />
        ) : (
          <Play className="size-4 fill-current" />
        )}
        {loading.simulation ? "Running simulation…" : "Run Simulation"}
      </button>

      {/* Results */}
      {result && explanation && (
        <div className="space-y-3 pt-3 border-t animate-in fade-in slide-in-from-bottom-2">
          {/* Headline */}
          <p className="text-xs font-bold text-muted-foreground uppercase tracking-wide">
            Simulation Result
          </p>
          <p className="text-sm font-semibold text-foreground leading-snug">
            {explanation.headline}…
          </p>

          {/* Before → After */}
          <div className="flex items-center gap-2">
            <div className="flex-1 p-3 rounded-lg bg-muted/40 border text-center">
              <div className="text-[10px] text-muted-foreground font-bold uppercase mb-1">
                Current forecast
              </div>
              <div className="text-base font-bold">{fmt(result.baseline)}</div>
            </div>
            <ArrowRight
              className={`size-5 shrink-0 ${
                result.pct_impact >= 0 ? "text-green-500" : "text-red-500"
              }`}
            />
            <div
              className={`flex-1 p-3 rounded-lg border text-center ${
                result.pct_impact >= 0
                  ? "bg-green-500/5 border-green-500/30"
                  : "bg-red-500/5 border-red-500/30"
              }`}
            >
              <div
                className={`text-[10px] font-bold uppercase mb-1 ${
                  result.pct_impact >= 0 ? "text-green-600 dark:text-green-400" : "text-red-500"
                }`}
              >
                New forecast
              </div>
              <div className="text-base font-bold">{fmt(result.counterfactual)}</div>
            </div>
          </div>

          {/* Summary cards */}
          <div className="space-y-2">
            <div
              className={`flex items-center gap-3 p-3 rounded-lg border ${
                result.pct_impact >= 0
                  ? "bg-green-500/5 border-green-500/20"
                  : "bg-red-500/5 border-red-500/20"
              }`}
            >
              {result.pct_impact >= 0 ? (
                <TrendingUp className="size-5 text-green-500 shrink-0" />
              ) : (
                <TrendingDown className="size-5 text-red-500 shrink-0" />
              )}
              <div>
                <p className="text-sm font-bold">{explanation.summary}</p>
                <p className="text-xs text-muted-foreground mt-0.5">{explanation.detail}</p>
              </div>
            </div>

            <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/20 border">
              <span className="text-base shrink-0">🌍</span>
              <p className="text-xs text-muted-foreground">{explanation.global}</p>
            </div>
          </div>

          <p className="text-[10px] text-muted-foreground/50 text-center pt-1">
            Powered by Graph Neural Network — scenario tested on live model
          </p>
        </div>
      )}
    </div>
  )
}
