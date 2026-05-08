# AI-Driven Trade Forecasting (Frontend)

An interactive dashboard for visualizing India's bilateral export predictions with transparent explainability and actionable news intelligence.

## Run
- Open the preview in v0 (no commands needed).
- The app uses the Next.js App Router and Tailwind v4 with design tokens.

## Structure
- app/page.tsx — page shell
- components/ — modular UI:
  - header.tsx, sidebar.tsx, theme-toggle.tsx, footer.tsx
  - trade-network.tsx — world map + network overlay
  - predictions-table.tsx — sortable table
  - panels/alerts-panel.tsx — risks/opportunities with CTAs
  - panels/news-panel.tsx — external links only
  - panels/explainability-panel.tsx — Recharts bar charts
- lib/types.ts, lib/mock.ts — types and mock data

## Integrations
// TODO: Integrate with backend via API
- Replace mock loaders in `components/dashboard/store.ts` with fetch calls:
  - `/api/predictions?sector={sector}&month={month}`
  - `/api/alerts?sector={sector}&month={month}`
  - `/api/news?sector={sector}&month={month}&partner={code}`
  - `/api/explainability?sector={sector}&month={month}&partner={code}`

Notes:
- Keep external articles as links (don’t render full bodies).
- Maintain accessibility (ARIA labels, focus styles, keyboard nav).
- The color system uses a deep-navy brand with emerald (positive) and red (risk) accents.
