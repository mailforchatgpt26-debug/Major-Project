"use client"

import Link from "next/link"
import { useEffect } from "react"
import { useDashboardStore } from "@/components/dashboard/store"

export default function AlertsPage() {
  const { alerts, loadAlerts } = useDashboardStore()

  useEffect(() => {
    loadAlerts()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <main className="min-h-dvh">
      <div className="mx-auto max-w-[1200px] px-4 lg:px-6 py-6 space-y-6">
        <header className="flex items-center justify-between">
          <h1 className="text-xl font-semibold text-pretty">Alerts & Recommendations</h1>
          <Link href="/" className="text-sm underline underline-offset-4 text-muted-foreground hover:text-foreground">
            Back to Dashboard
          </Link>
        </header>

        <section className="grid gap-4 md:grid-cols-2">
          {alerts.map((a) => (
            <div key={a.id} className="rounded-lg border bg-card/80 shadow-sm p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="text-sm font-semibold truncate">{a.title}</h3>
                  <p className="text-xs text-muted-foreground">
                    {a.summary} — {(a.change * 100).toFixed(1)}%
                  </p>
                  {a.recommendations?.length ? (
                    <ul className="text-xs mt-1 space-y-0.5">
                      {a.recommendations.map((r, i) => {
                        const text = (r as any).text || r.rationale || r.country_name
                        return text ? <li key={i} className="text-muted-foreground">→ {text}</li> : null
                      })}
                    </ul>
                  ) : null}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <Link
                    href={`/partners/${a.partnerCode}#news`}
                    className="rounded-md border px-2 py-1 text-xs hover:bg-accent"
                  >
                    View News
                  </Link>
                  <Link
                    href={`/partners/${a.partnerCode}`}
                    className="rounded-md border px-2 py-1 text-xs bg-primary text-primary-foreground"
                  >
                    Details
                  </Link>
                </div>
              </div>
            </div>
          ))}
          {alerts.length === 0 && <p className="text-sm text-muted-foreground">No alerts available.</p>}
        </section>
      </div>
    </main>
  )
}
