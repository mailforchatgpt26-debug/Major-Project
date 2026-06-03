"use client"

import Link from "next/link"
import type { AlertItem } from "@/lib/types"
import { isLegacyInsightLine } from "@/lib/corridor-insights"

export function AlertsPanel({
  alerts,
}: {
  alerts: AlertItem[]
}) {
  const positive = alerts.filter((a) => a.type === "opportunity")
  const negative = alerts.filter((a) => a.type === "risk")

  return (
    <div className="p-3">
      <h3 className="text-sm font-semibold mb-2">Alerts & Recommendations</h3>
      <div className="space-y-3">
        <section aria-labelledby="opportunities">
          <h4
            id="opportunities"
            className="text-xs font-medium text-pretty mb-1"
            style={{ color: "var(--color-chart-1)" }}
          >
            Opportunity spikes
          </h4>
          <div className="space-y-2">
            {positive.map((a) => (
              <AlertCard key={a.id} item={a} />
            ))}
            {positive.length === 0 && <EmptyLine label="No opportunity alerts" />}
          </div>
        </section>

        <section aria-labelledby="risks" className="pt-2">
          <h4 id="risks" className="text-xs font-medium mb-1" style={{ color: "var(--destructive)" }}>
            Negative shocks
          </h4>
          <div className="space-y-2">
            {negative.map((a) => (
              <AlertCard key={a.id} item={a} />
            ))}
            {negative.length === 0 && <EmptyLine label="No risk alerts" />}
          </div>
        </section>
      </div>
    </div>
  )
}

function AlertCard({ item }: { item: AlertItem }) {
  // Safely render recommendations regardless of API format
  const renderRecommendations = () => {
    if (!item.recommendations || item.recommendations.length === 0) return null
    return (
      <ul className="mt-2 space-y-1">
        {item.recommendations
          .map((r: { text?: string; action?: string; country_name?: string }) => {
            const text = r.text || r.action || r.country_name || null
            if (!text || isLegacyInsightLine(String(text))) return null
            return text
          })
          .filter((t): t is string => Boolean(t))
          .slice(0, 6)
          .map((text, i) => (
            <li key={i} className="text-xs text-muted-foreground flex gap-1.5">
              <span className="shrink-0 mt-0.5">→</span>
              <span>{text}</span>
            </li>
          ))}
      </ul>
    )
  }

  return (
    <div className="rounded-md border p-2 bg-background">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-medium truncate">{item.title}</p>
          <p className="text-xs text-muted-foreground">
            {item.summary} — {(item.change * 100).toFixed(1)}%
          </p>
          {renderRecommendations()}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Link
            href={`/partners/${item.partnerCode}#news`}
            className="rounded-md border px-2 py-1 text-xs hover:bg-accent"
            aria-label={`View news for ${item.partner}`}
          >
            View News
          </Link>
          <Link
            href={`/partners/${item.partnerCode}`}
            className="rounded-md border px-2 py-1 text-xs bg-primary text-primary-foreground"
            aria-label={`View details for ${item.partner}`}
          >
            Details
          </Link>
        </div>
      </div>
    </div>
  )
}

function EmptyLine({ label }: { label: string }) {
  return <p className="text-xs text-muted-foreground border rounded-md px-2 py-2">{label}</p>
}
