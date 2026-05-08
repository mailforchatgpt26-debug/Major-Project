"use client"

import { useMemo, useState } from "react"
import { Calendar } from "@/components/ui/calendar"

function parseMonth(value: string | undefined): Date {
  // value like "2025-09"
  if (!value) return new Date()
  const [y, m] = value.split("-").map((v) => Number.parseInt(v, 10))
  const d = new Date()
  d.setFullYear(y || d.getFullYear(), (m || d.getMonth() + 1) - 1, 1)
  d.setHours(0, 0, 0, 0)
  return d
}

export function MonthYearPicker({
  value,
  onChange,
  fromYear = 2015,
  toYear = 2035,
}: {
  value?: string
  onChange: (next: string) => void
  fromYear?: number
  toYear?: number
}) {
  const initial = useMemo(() => parseMonth(value), [value])
  const [view, setView] = useState<Date>(initial)

  function updateMonth(d: Date) {
    const y = d.getFullYear()
    const m = `${d.getMonth() + 1}`.padStart(2, "0")
    onChange(`${y}-${m}`)
  }

  return (
    <div>
      <Calendar
        mode="single"
        captionLayout="dropdown"
        selected={initial}
        defaultMonth={initial}
        month={view}
        onMonthChange={(d) => {
          setView(d)
          updateMonth(d)
        }}
        fromYear={fromYear}
        toYear={toYear}
        showOutsideDays
        // keep days clickable (optional); selecting a day also updates month
        onSelect={(d) => d && updateMonth(d)}
        className="rounded-md border"
      />
      <p className="mt-2 text-xs text-muted-foreground">Select month and year from the dropdowns above.</p>
    </div>
  )
}
