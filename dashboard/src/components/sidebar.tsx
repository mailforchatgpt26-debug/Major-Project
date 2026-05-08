"use client"

export function Sidebar() {
  return (
    <div className="p-4 lg:p-6 space-y-6">
      <section>
        <h2 className="text-sm font-medium text-muted-foreground mb-2">Product sector</h2>
        <div className="px-3 py-2 rounded-md border bg-primary text-primary-foreground text-sm font-medium">
          Pharmaceuticals (HS Chapter 30)
        </div>
      </section>

      <section>
        <h2 className="text-sm font-medium text-muted-foreground mb-2">Analysis</h2>
        <div className="space-y-1">
          <a href="#overview" className="block text-xs px-3 py-1.5 rounded hover:bg-accent transition-colors">Network Overview</a>
          <a href="#predictions" className="block text-xs px-3 py-1.5 rounded hover:bg-accent transition-colors">Trade Forecasts</a>
          <button
            onClick={() => document.getElementById("open-scenario-engine")?.click()}
            className="block w-full text-left text-xs px-3 py-1.5 rounded hover:bg-accent transition-colors flex items-center gap-2"
          >
            <span className="size-1.5 rounded-full bg-primary" />
            Policy Scenario Engine
          </button>
        </div>
      </section>

      <section className="pt-2">
        <p className="text-xs text-muted-foreground">
          India pharma export forecasts · GNN + gravity model · Real UN Comtrade data
        </p>
      </section>
    </div>
  )
}
