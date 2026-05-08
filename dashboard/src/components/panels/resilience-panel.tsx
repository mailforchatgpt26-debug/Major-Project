"use client"

import type { TradeResilience, ResiliencePartner } from "@/lib/types"

function humaniseFlag(flag: string): string | null {
  if (flag.includes("Low model confidence")) return null
  if (flag.includes("High export dependency"))
    return flag.replace("High export dependency", "Heavy reliance on this market")
  if (flag.includes("Elevated export concentration"))
    return flag.replace("Elevated export concentration", "Significant concentration")
  if (flag.includes("Critical import source"))
    return flag.replace("Critical import source", "Key import source")
  if (flag.includes("Declining trend"))
    return flag.replace("Declining trend", "Exports declining")
  if (flag.includes("Negative geopolitical sentiment"))
    return "Negative trade news"
  if (flag.includes("High network centrality"))
    return "Important regional hub"
  return flag
}

function ResilienceBar({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const color = pct >= 70 ? "bg-green-500" : pct >= 50 ? "bg-amber-500" : "bg-red-500"
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs tabular-nums">{pct}/100</span>
    </div>
  )
}

function ConcentrationBar({ value, label }: { value: number; label: string }) {
  const pct = Math.min(value / 10000, 1)
  const color = label === "competitive" ? "bg-green-500" : label === "moderate" ? "bg-amber-500" : "bg-red-500"
  const plain: Record<string, string> = {
    competitive: "Well diversified",
    moderate: "Moderately concentrated",
    concentrated: "Highly concentrated",
  }
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground capitalize">{plain[label] ?? label}</span>
        <span className="font-medium tabular-nums">{value.toFixed(0)}</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct * 100}%` }} />
      </div>
    </div>
  )
}

function PartnerRow({ p, variant }: { p: ResiliencePartner; variant: "risk" | "opportunity" }) {
  const changePct = p.export_change * 100
  const changeColor = changePct >= 0 ? "text-green-600 dark:text-green-400" : "text-destructive"
  const visibleFlags = (p.flags ?? []).map(humaniseFlag).filter(Boolean) as string[]

  return (
    <div className="rounded-md border border-border bg-card/60 p-2.5 text-xs space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium">{p.partner}</span>
        <span className={`font-semibold tabular-nums ${changeColor}`}>
          {changePct >= 0 ? "+" : ""}{changePct.toFixed(1)}%
        </span>
      </div>

      <div className="text-muted-foreground">
        Stability: <ResilienceBar score={p.resilience_score} />
      </div>

      <div className="text-muted-foreground">
        Export share: <span className="text-foreground font-medium">{(p.export_share * 100).toFixed(1)}%</span>
      </div>

      {visibleFlags.length > 0 && (
        <ul className="space-y-0.5 text-muted-foreground">
          {visibleFlags.map((f) => (
            <li key={f} className="flex items-start gap-1">
              <span className="shrink-0 mt-0.5">{variant === "risk" ? "⚠" : "✓"}</span>
              <span>{f}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function ResiliencePanel({ resilience }: { resilience?: TradeResilience }) {
  if (!resilience) {
    return (
      <div className="p-3">
        <h3 className="text-sm font-semibold mb-2">Trade Resilience</h3>
        <div className="h-48 rounded-md border bg-muted/40 animate-pulse" />
      </div>
    )
  }

  return (
    <div className="p-3 space-y-5">
      <div>
        <h3 className="text-sm font-semibold">Trade Resilience</h3>
        <p className="text-xs text-muted-foreground mt-0.5">{resilience.summary}</p>
      </div>

      {/* Concentration */}
      <div className="space-y-2">
        <p className="text-xs font-medium">Partner concentration</p>
        <div className="space-y-3">
          <div>
            <p className="text-xs text-muted-foreground mb-1">Exports</p>
            <ConcentrationBar value={resilience.export_hhi} label={resilience.export_hhi_label} />
          </div>
          <div>
            <p className="text-xs text-muted-foreground mb-1">Imports</p>
            <ConcentrationBar value={resilience.import_hhi} label={resilience.import_hhi_label} />
          </div>
        </div>
        <p className="text-xs text-muted-foreground">Lower = more diversified (better)</p>
      </div>

      {/* Vulnerable corridors */}
      {resilience.top_risks.length > 0 && (
        <div>
          <p className="text-xs font-medium mb-2 text-destructive">Declining corridors</p>
          <div className="space-y-2">
            {resilience.top_risks.map((p) => (
              <PartnerRow key={p.partnerCode} p={p} variant="risk" />
            ))}
          </div>
        </div>
      )}

      {/* Growth opportunities */}
      {resilience.top_opportunities.length > 0 && (
        <div>
          <p className="text-xs font-medium mb-2 text-green-600 dark:text-green-400">
            Expansion opportunities
          </p>
          <div className="space-y-2">
            {resilience.top_opportunities.map((p) => (
              <PartnerRow key={p.partnerCode} p={p} variant="opportunity" />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
