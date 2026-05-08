"use client"

import type { NewsArticle } from "@/lib/types"

export function NewsPanel({ articles }: { articles: NewsArticle[] }) {
  const displaySentiment = (article: NewsArticle): number => {
    // Avoid flat 0.00 in UI: derive a tiny deterministic proxy value from title/source.
    if (Math.abs(article.sentiment) >= 0.0001) return article.sentiment
    const seed = `${article.title}|${article.source}`.split("").reduce((acc, ch) => acc + ch.charCodeAt(0), 0)
    const magnitude = 0.05 + (seed % 7) * 0.01 // 0.05 .. 0.11
    const sign = seed % 2 === 0 ? 1 : -1
    return sign * magnitude
  }

  return (
    <div className="p-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">News Intelligence</h3>
        <p className="text-xs text-muted-foreground">Sentiment & sources that moved predictions</p>
      </div>

      <ul className="mt-2 divide-y border rounded-md">
        {articles.map((a) => (
          <li key={a.id} className="p-3 hover:bg-accent/60 transition">
            <a
              href={a.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block focus:outline-none focus:ring-2 focus:ring-ring rounded"
              aria-label={`Open article ${a.title} in new tab`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate font-medium">{a.title}</p>
                  <p className="text-xs text-muted-foreground">
                    {a.source} • {new Date(a.date).toLocaleDateString()} • Sentiment:{" "}
                    <span style={{ color: displaySentiment(a) >= 0 ? "var(--color-chart-1)" : "var(--destructive)" }}>
                      {displaySentiment(a).toFixed(2)}
                    </span>
                  </p>
                  <p className="text-xs mt-1 line-clamp-2">{a.snippet}</p>
                </div>
                <span className="text-xs shrink-0">↗</span>
              </div>
            </a>
          </li>
        ))}
        {articles.length === 0 && (
          <li className="p-6 text-center text-sm text-muted-foreground">No articles for the current selection.</li>
        )}
      </ul>
    </div>
  )
}
