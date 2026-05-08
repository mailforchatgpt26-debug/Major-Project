import { Suspense } from "react"
import { Header } from "@/components/header"
import { ThemeToggle } from "@/components/theme-toggle"
import { LiveClock } from "@/components/live-clock"
import DashboardClient from "@/components/dashboard/dashboard-client"

export default function Page() {
  return (
    <main className="min-h-dvh grid grid-rows-[auto_1fr]">
      <Header rightSlot={<div className="flex items-center gap-3"><LiveClock /><ThemeToggle /></div>} />
      <section className="p-4 lg:p-6">
        <Suspense fallback={<div className="animate-pulse h-40 rounded-lg bg-muted/50" />}>
          <DashboardClient />
        </Suspense>
      </section>
    </main>
  )
}
