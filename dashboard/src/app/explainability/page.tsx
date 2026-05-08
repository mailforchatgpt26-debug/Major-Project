"use client"

import { useEffect } from "react"
import { useDashboardStore } from "@/components/dashboard/store"
import { ExplainabilityPanel } from "@/components/panels/explainability-panel"
import Link from "next/link"

export default function ExplainabilityPage() {
  const { explainability, selectedPartner, loadExplainability } = useDashboardStore()

  useEffect(() => {
    // Load detailed explainability for the currently selected partner (or global)
    loadExplainability(selectedPartner)
  }, [selectedPartner, loadExplainability])

  return (
    <main className="mx-auto max-w-[1100px] px-4 lg:px-6 py-6 space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-pretty">Explainability</h1>
          <p className="text-sm text-muted-foreground">
            Model insights behind the forecasts: attention to trade neighbors and feature importance.
            {selectedPartner
              ? ` Focused on ${selectedPartner}.`
              : " Select a partner on the dashboard to focus the analysis."}
          </p>
        </div>
        <Link href="/" className="text-sm underline underline-offset-4 text-muted-foreground hover:text-foreground">
          Back to Dashboard
        </Link>
      </header>

      <section className="rounded-xl border bg-card/80 shadow-sm">
        <ExplainabilityPanel explainability={explainability} />
      </section>

      <section className="rounded-xl border bg-card/80 shadow-sm p-4">
        <h2 className="text-sm font-semibold mb-2">Methodology</h2>
        <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
          <li>Attention weights indicate relative influence of partner relationships on the forecast.</li>
          <li>Feature importance summarizes each variable’s contribution to model output.</li>
          <li>Higher bars = stronger influence; compare across partners or features to understand drivers.</li>
        </ul>
      </section>
    </main>
  )
}
