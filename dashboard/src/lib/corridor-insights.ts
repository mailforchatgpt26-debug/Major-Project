/** Corridor card bullets: max 3 narrative lines; drop legacy label-style flags. */
const DROPPED_PREFIXES = [
  "Model inference:",
  "Likely decline period:",
  "Historical export growth CAGR",
  "Model-suggested alternative corridors:",
]

export function corridorCardInsights(flags: string[] | undefined): string[] {
  return (flags ?? [])
    .map((f) => f?.trim())
    .filter((f): f is string => {
      if (!f) return false
      return !DROPPED_PREFIXES.some((p) => f.startsWith(p))
    })
    .slice(0, 3)
}
