"use client"

import { useEffect, useState } from "react"

export function ThemeToggle() {
  const [isDark, setIsDark] = useState(true) // default to dark

  useEffect(() => {
    const root = document.documentElement
    const stored = localStorage.getItem("theme")
    // Default to dark if no preference stored
    const nextDark = stored ? stored === "dark" : true
    setIsDark(nextDark)
    root.classList.toggle("dark", nextDark)
  }, [])

  function toggle() {
    const root = document.documentElement
    const next = !isDark
    setIsDark(next)
    root.classList.toggle("dark", next)
    localStorage.setItem("theme", next ? "dark" : "light")
  }

  return (
    <button
      onClick={toggle}
      aria-pressed={isDark}
      aria-label="Toggle theme"
      className="inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring transition-colors"
      title={isDark ? "Switch to light" : "Switch to dark"}
    >
      <span className="size-2.5 rounded-full" style={{ background: isDark ? "#22c55e" : "var(--color-chart-1)" }} />
      {isDark ? "Dark" : "Light"}
    </button>
  )
}
