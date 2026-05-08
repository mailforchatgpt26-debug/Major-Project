"use client"

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import type { Explainability } from "@/lib/types"

export function ExplainabilityPanel({ explainability }: { explainability?: Explainability }) {
  return (
    <div className="p-3">
      <h3 className="text-sm font-semibold">Why this prediction?</h3>
      {!explainability ? (
        <div className="h-40 mt-2 rounded-md border bg-muted/40 animate-pulse" />
      ) : (
        <div className="space-y-4">
          <div>
            <p className="text-xs text-muted-foreground mb-1">Feature importance</p>
            <div className="h-40 rounded-md border bg-background">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={explainability.features}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border, rgba(148,163,184,0.3))" />
                  <XAxis
                    dataKey="feature"
                    tick={{ fontSize: 11, fill: "var(--muted-foreground, #94a3b8)" }}
                    stroke="var(--muted-foreground, #94a3b8)"
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "var(--muted-foreground, #94a3b8)" }}
                    stroke="var(--muted-foreground, #94a3b8)"
                  />
                  <Tooltip />
                  <Bar dataKey="importance" fill="var(--chart-2, #22c55e)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <p className="text-xs text-pretty">{explainability.blurb}</p>
        </div>
      )}
    </div>
  )
}
