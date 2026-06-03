/** Corridor card bullets: max 2 narrative lines; drop legacy label-style flags. */
const DROPPED_PREFIXES = [
  "Model inference:",
  "Likely decline period:",
  "Historical export growth CAGR",
  "5-yr export CAGR",
  "Model-suggested alternative corridors:",
]

const CARD_INSIGHT_LIMIT = 2
const ALERT_INSIGHT_LIMIT = 6

export function isLegacyInsightLine(text: string): boolean {
  const s = text.trim()
  if (!s) return true
  if (DROPPED_PREFIXES.some((p) => s.startsWith(p))) return true
  if (/export\s+CAGR\s*\(\s*20\d{2}/i.test(s)) return true
  return false
}

export function corridorCardInsights(flags: string[] | undefined): string[] {
  return (flags ?? [])
    .map((f) => f?.trim())
    .filter((f): f is string => Boolean(f) && !isLegacyInsightLine(f))
    .slice(0, CARD_INSIGHT_LIMIT)
}

export function alertDetailInsights(flags: string[] | undefined): string[] {
  return (flags ?? [])
    .map((f) => f?.trim())
    .filter((f): f is string => Boolean(f) && !isLegacyInsightLine(f))
    .slice(0, ALERT_INSIGHT_LIMIT)
}
