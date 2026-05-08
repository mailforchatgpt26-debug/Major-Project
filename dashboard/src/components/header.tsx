import type React from "react"
import Link from "next/link"

export function Header({ rightSlot }: { rightSlot?: React.ReactNode }) {
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/80 backdrop-blur">
      <div className="mx-auto max-w-[1400px] px-4 lg:px-6 flex h-14 items-center justify-between">
        <Link
          href="/"
          className="group inline-flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-ring rounded-md"
          aria-label="Go to home"
        >
          <div className="size-6 rounded-md bg-primary" aria-hidden />
          <div className="leading-tight">
            <span className="block text-lg md:text-xl font-extrabold tracking-tight text-pretty">PharmaTrade AI</span>
            <span className="hidden md:block text-xs md:text-sm font-semibold text-foreground/90">
              Forecasting tomorrow&apos;s pharma trade flows
            </span>
          </div>
        </Link>

        <nav aria-label="Primary" className="hidden md:flex items-center gap-6 text-sm text-muted-foreground">
          <Link
            href="/#predictions"
            className="hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
          >
            Predictions
          </Link>
          <Link
            href="/#news"
            className="hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
          >
            News
          </Link>
          <Link
            href="/resilience"
            className="hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
          >
            Risk & Resilience
          </Link>
        </nav>

        <div className="flex items-center gap-2">{rightSlot}</div>
      </div>
    </header>
  )
}
