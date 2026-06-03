import type { Metadata } from "next"
import Link from "next/link"
import {
  ArrowRight,
  Brain,
  Check,
  Globe2,
  LineChart,
  Minus,
  Network,
  Shield,
  Sparkles,
  X,
  Zap,
} from "lucide-react"
import { Header } from "@/components/header"
import { ThemeToggle } from "@/components/theme-toggle"
import {
  COMPARISON_MATRIX,
  COMPETITORS,
  DIFFERENTIATORS,
  MATRIX_COLUMNS,
  ONE_LINE_CONCLUSION,
  POSITIONING_STATEMENT,
  type MatrixCell,
} from "@/lib/competitive-comparison"

export const metadata: Metadata = {
  title: "Why PharmaTrade AI",
  description:
    "How PharmaTrade AI compares to PharmaFootpath, PharmInt AI, GTAIC, and TradeInt—predictive, risk-aware, explainable pharma trade decision support.",
}

const ICONS = {
  network: Network,
  news: Sparkles,
  risk: Shield,
  simulate: Brain,
} as const

function MatrixIcon({ value, highlight }: { value: MatrixCell; highlight?: boolean }) {
  if (value === "yes") {
    return (
      <span
        className={`inline-flex items-center justify-center size-6 rounded-full ${
          highlight ? "bg-emerald-500/20 text-emerald-400" : "bg-muted text-muted-foreground"
        }`}
      >
        <Check className="size-3.5" strokeWidth={2.5} />
      </span>
    )
  }
  if (value === "no") {
    return (
      <span className="inline-flex items-center justify-center size-6 rounded-full bg-muted/60 text-muted-foreground/50">
        <X className="size-3.5" />
      </span>
    )
  }
  return (
    <span className="text-[10px] font-medium uppercase tracking-wide text-amber-600 dark:text-amber-400">
      {value === "limited" ? "Limited" : "Partial"}
    </span>
  )
}

export default function WhyUsPage() {
  return (
    <main className="min-h-dvh grid grid-rows-[auto_1fr]">
      <Header rightSlot={<ThemeToggle />} />

      <div className="relative overflow-hidden">
        {/* Ambient background */}
        <div
          className="pointer-events-none absolute inset-0 opacity-40"
          aria-hidden
          style={{
            background:
              "radial-gradient(ellipse 80% 50% at 50% -20%, oklch(0.45 0.12 260 / 0.35), transparent), radial-gradient(ellipse 60% 40% at 100% 50%, oklch(0.55 0.15 150 / 0.12), transparent)",
          }}
        />

        <div className="relative mx-auto max-w-[1200px] px-4 lg:px-6 py-10 lg:py-14 space-y-16 lg:space-y-20">
          {/* Hero */}
          <section className="max-w-3xl">
            <p className="text-xs font-semibold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-3">
              Platform differentiation
            </p>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight text-balance leading-[1.15]">
              Predictive, risk-aware, and explainable pharma trade intelligence
            </h1>
            <p className="mt-4 text-base lg:text-lg text-muted-foreground leading-relaxed">
              {POSITIONING_STATEMENT}
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link
                href="/#predictions"
                className="inline-flex items-center gap-2 rounded-lg bg-primary text-primary-foreground px-4 py-2.5 text-sm font-semibold hover:opacity-90 transition-opacity"
              >
                Explore forecasts
                <ArrowRight className="size-4" />
              </Link>
              <Link
                href="/resilience"
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-card/80 px-4 py-2.5 text-sm font-medium hover:bg-muted/50 transition-colors"
              >
                Risk & resilience
              </Link>
            </div>
          </section>

          {/* Four differentiators */}
          <section>
            <h2 className="text-lg font-semibold mb-1">What makes PharmaTrade AI different</h2>
            <p className="text-sm text-muted-foreground mb-6">
              Four capabilities aligned with project objectives and deliverables—not a single &quot;better model&quot; claim.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {DIFFERENTIATORS.map((d) => {
                const Icon = ICONS[d.icon]
                return (
                  <article
                    key={d.title}
                    className="rounded-xl border border-border/80 bg-card/60 p-5 backdrop-blur-sm hover:border-emerald-500/30 transition-colors"
                  >
                    <div className="flex items-start gap-3">
                      <div className="rounded-lg bg-emerald-500/10 p-2.5 shrink-0">
                        <Icon className="size-5 text-emerald-600 dark:text-emerald-400" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-sm">{d.title}</h3>
                        <p className="mt-1.5 text-sm text-muted-foreground leading-relaxed">{d.description}</p>
                      </div>
                    </div>
                  </article>
                )
              })}
            </div>
          </section>

          {/* Comparison matrix */}
          <section>
            <div className="flex items-end justify-between gap-4 mb-6">
              <div>
                <h2 className="text-lg font-semibold">Feature comparison at a glance</h2>
                <p className="text-sm text-muted-foreground mt-1">
                  Descriptive analytics vs integrated forecasting, sentiment, risk, and simulation.
                </p>
              </div>
              <LineChart className="size-8 text-muted-foreground/30 shrink-0 hidden sm:block" />
            </div>

            <div className="rounded-xl border border-border overflow-hidden bg-card/40">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[720px] text-sm">
                  <thead>
                    <tr className="border-b border-border bg-muted/30">
                      <th className="text-left font-medium px-4 py-3 sticky left-0 bg-muted/30 z-10 min-w-[200px]">
                        Capability
                      </th>
                      {MATRIX_COLUMNS.map((col) => (
                        <th
                          key={col.key}
                          className={`px-3 py-3 text-center font-medium whitespace-nowrap ${
                            col.highlight
                              ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                              : "text-muted-foreground"
                          }`}
                        >
                          {col.label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {COMPARISON_MATRIX.map((row, i) => (
                      <tr
                        key={row.feature}
                        className={i % 2 === 0 ? "bg-background/40" : "bg-muted/10"}
                      >
                        <td className="px-4 py-3 font-medium sticky left-0 bg-inherit z-10 border-r border-border/50">
                          {row.feature}
                        </td>
                        {MATRIX_COLUMNS.map((col) => (
                          <td
                            key={col.key}
                            className={`px-3 py-3 text-center ${
                              col.highlight ? "bg-emerald-500/5" : ""
                            }`}
                          >
                            <MatrixIcon value={row[col.key]} highlight={col.highlight} />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="px-4 py-2.5 border-t border-border flex flex-wrap gap-4 text-[10px] text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <Check className="size-3 text-emerald-500" /> Full support
                </span>
                <span className="flex items-center gap-1.5">
                  <Minus className="size-3" /> Limited / partial
                </span>
                <span className="flex items-center gap-1.5">
                  <X className="size-3 opacity-50" /> Not offered
                </span>
              </div>
            </div>
          </section>

          {/* Competitor deep dives */}
          <section>
            <h2 className="text-lg font-semibold mb-1">How we compare to leading platforms</h2>
            <p className="text-sm text-muted-foreground mb-8">
              Each competitor excels at market intelligence; PharmaTrade AI adds prediction, network learning, and
              decision support in one workflow.
            </p>
            <div className="space-y-8">
              {COMPETITORS.map((c, idx) => (
                <article
                  key={c.id}
                  className="rounded-xl border border-border bg-card/50 overflow-hidden"
                >
                  <div className="px-5 py-4 border-b border-border bg-muted/20 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-muted-foreground">0{idx + 1}</span>
                        <h3 className="text-base font-bold">{c.name}</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mt-0.5">{c.tagline}</p>
                    </div>
                    <ul className="flex flex-wrap gap-1.5">
                      {c.focus.map((f) => (
                        <li
                          key={f}
                          className="text-[10px] rounded-full border border-border px-2 py-0.5 text-muted-foreground"
                        >
                          {f}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="p-5 grid gap-4 md:grid-cols-3">
                    {c.advantages.map((a) => (
                      <div key={a.title} className="space-y-2">
                        <p className="text-xs font-semibold text-foreground flex items-center gap-1.5">
                          <Zap className="size-3.5 text-emerald-500 shrink-0" />
                          {a.title}
                        </p>
                        <div className="rounded-lg bg-muted/30 px-3 py-2 text-[11px] leading-relaxed">
                          <p className="text-muted-foreground">
                            <span className="font-medium text-foreground/80">{c.name}: </span>
                            {a.theirs}
                          </p>
                        </div>
                        <div className="rounded-lg border border-emerald-500/25 bg-emerald-500/5 px-3 py-2 text-[11px] leading-relaxed">
                          <p>
                            <span className="font-medium text-emerald-700 dark:text-emerald-400">
                              PharmaTrade AI:{" "}
                            </span>
                            {a.ours}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          </section>

          {/* Conclusion */}
          <section className="rounded-2xl border border-emerald-500/30 bg-gradient-to-br from-emerald-500/10 via-card to-card p-6 lg:p-8">
            <div className="flex items-start gap-4">
              <div className="hidden sm:flex rounded-xl bg-emerald-500/15 p-3">
                <Globe2 className="size-8 text-emerald-600 dark:text-emerald-400" />
              </div>
              <div className="flex-1">
                <h2 className="text-lg font-semibold">One-line takeaway</h2>
                <p className="mt-3 text-sm lg:text-base text-muted-foreground leading-relaxed italic border-l-2 border-emerald-500/50 pl-4">
                  {ONE_LINE_CONCLUSION}
                </p>
                <Link
                  href="/"
                  className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-emerald-600 dark:text-emerald-400 hover:underline"
                >
                  Open the live dashboard
                  <ArrowRight className="size-4" />
                </Link>
              </div>
            </div>
          </section>

          <p className="text-center text-xs text-muted-foreground pb-4">
            Comparisons reflect publicly described capabilities of reference platforms and this project&apos;s
            documented scope. Competitor offerings may evolve.
          </p>
        </div>
      </div>
    </main>
  )
}
