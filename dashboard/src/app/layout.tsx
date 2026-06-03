import type { Metadata } from 'next'
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'

export const metadata: Metadata = {
  title: {
    default: 'PharmaTrade AI - Global Pharma Trade Forecasting',
    template: '%s | PharmaTrade AI',
  },
  description: 'AI-powered pharma trade forecasting and risk intelligence for India’s bilateral export/import corridors.',
}

const vercelHosted =
  process.env.VERCEL === '1' || process.env.NEXT_PUBLIC_VERCEL === '1'

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable}`} suppressHydrationWarning>
        {children}
        {vercelHosted ? <Analytics /> : null}
      </body>
    </html>
  )
}
