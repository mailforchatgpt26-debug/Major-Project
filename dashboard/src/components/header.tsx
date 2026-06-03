"use client"

import type React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"

const NAV_LINKS = [
  { href: "/#predictions", label: "Predictions", match: (p: string) => p === "/" },
  { href: "/#news", label: "News", match: (p: string) => p === "/" },
  { href: "/resilience", label: "Risk & Resilience", match: (p: string) => p.startsWith("/resilience") },
  { href: "/why-us", label: "Why Us", match: (p: string) => p.startsWith("/why-us") },
] as const

function navClass(active: boolean) {
  return active
    ? "text-foreground font-medium whitespace-nowrap"
    : "hover:text-foreground whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
}

export function Header({ rightSlot }: { rightSlot?: React.ReactNode }) {
  const pathname = usePathname()

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/80 backdrop-blur">
      <div className="mx-auto max-w-[1400px] px-4 lg:px-6 flex h-14 items-center justify-between gap-3">
        <Link
          href="/"
          className="group inline-flex items-center gap-2 shrink-0 focus:outline-none focus:ring-2 focus:ring-ring rounded-md"
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

        <nav
          aria-label="Primary"
          className="flex items-center gap-4 md:gap-6 text-sm text-muted-foreground overflow-x-auto max-w-[min(100vw-12rem,42rem)] md:max-w-none"
        >
          {NAV_LINKS.map((item) => (
            <Link key={item.href} href={item.href} className={navClass(item.match(pathname))}>
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2 shrink-0">{rightSlot}</div>
      </div>
    </header>
  )
}
