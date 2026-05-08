"use client"

import { memo, useEffect, useMemo, useRef, useState } from "react"
import dynamic from "next/dynamic"
import { scaleLinear } from "d3-scale"
import type { Prediction } from "@/lib/types"
// topojson -> geojson conversion
import { feature } from "topojson-client"
import { useRouter } from "next/navigation" // import router to navigate to partner detail page on click

const Globe = dynamic(() => import("react-globe.gl"), { ssr: false })

type Props = {
  data: Prediction[]
  selectedPartner?: string
  onSelectPartner: (countryCode: string) => void
}

// Approximate partner coords (lon, lat)
const COORDS: Record<string, [number, number]> = {
  // Already present
  USA: [-100, 40],
  DEU: [10, 51],
  ARE: [54, 24],
  CHN: [103, 35],
  AUS: [133, -25],
  GBR: [-2, 54],
  JPN: [138, 36],
  BRA: [-51, -10],
  // Top pharma export partners
  ZAF: [25, -29],
  RUS: [90, 60],
  NGA: [8, 10],
  FRA: [2, 46],
  KEN: [38, -1],
  CAN: [-96, 60],
  NLD: [5, 52],
  PHL: [122, 12],
  BEL: [4, 51],
  LKA: [81, 7],
  TZA: [35, -6],
  NPL: [84, 28],
  MMR: [96, 20],
  VNM: [108, 16],
  UGA: [32, 1],
  GHA: [-2, 8],
  ETH: [40, 9],
  THA: [101, 15],
  MOZ: [35, -18],
  UKR: [32, 49],
  MLT: [14, 36],
  COD: [24, -4],
  CHL: [-71, -30],
  IRQ: [44, 33],
  MEX: [-102, 24],
  ZMB: [28, -14],
  ZWE: [30, -20],
  UZB: [63, 41],
  VEN: [-66, 8],
  KOR: [128, 36],
  SGP: [104, 1],
  HKG: [114, 22],
}
const INDIA: [number, number] = [78.9629, 20.5937]
const WORLD_TOPO_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json"

export const TradeNetwork = memo(function TradeNetwork({ data, selectedPartner, onSelectPartner }: Props) {
  const globeRef = useRef<any>(null)
  const router = useRouter() // router for navigation
  const [polygons, setPolygons] = useState<any[]>([])
  const [palette, setPalette] = useState(() => ({
    destructive: "#ef4444", // fallback red-500
    muted: "#9ca3af", // fallback gray-400
    chart1: "#10b981", // fallback emerald-500
    primary: "#38bdf8", // fallback sky-400
    secondary: "rgba(30,41,59,0.8)", // fallback slate-800 w/ alpha
    unselectedPoint: "#7c8aa0", // neutral for points
  }))

  useEffect(() => {
    try {
      const cs = getComputedStyle(document.documentElement)
      const get = (name: string, fb: string) => cs.getPropertyValue(name)?.trim() || fb
      setPalette({
        destructive: get("--destructive", palette.destructive),
        muted: get("--muted-foreground", palette.muted),
        chart1: get("--color-chart-1", palette.chart1),
        primary: get("--color-primary", palette.primary),
        secondary: get("--secondary", palette.secondary),
        unselectedPoint: get("--border", palette.unselectedPoint), // use a subtle neutral if available
      })
    } catch {
      // no-op: keep fallbacks
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Node size by value; arc thickness by value
  const maxVal = Math.max(1, ...data.map((d) => d.export_forecast))
  const size = scaleLinear().domain([0, maxVal]).range([0.6, 2.6])
  const thickness = scaleLinear().domain([0, maxVal]).range([0.5, 2.2])

  const color = scaleLinear<string>().domain([-0.25, 0, 0.25]).range(["#ef4444", "#9ca3af", "#10b981"])

  const nodes = useMemo(() => {
    return data
      .filter((d) => COORDS[d.partnerCode])
      .map((d) => {
        const [lon, lat] = COORDS[d.partnerCode]
        return {
          code: d.partnerCode,
          name: d.partner,
          lon,
          lat,
          value: d.export_forecast,
          change: d.export_change,
          selected: selectedPartner === d.partnerCode,
        }
      })
  }, [data, selectedPartner])

  const arcs = useMemo(() => {
    const [iLon, iLat] = INDIA
    return nodes.map((n) => ({
      startLat: iLat,
      startLng: iLon,
      endLat: n.lat,
      endLng: n.lon,
      color: color(n.change),
      thickness: thickness(n.value),
      selected: n.selected,
    }))
  }, [nodes, color, thickness])

  // Fetch world polygons (countries), no labels
  useEffect(() => {
    let mounted = true
    ;(async () => {
      try {
        const res = await fetch(WORLD_TOPO_URL)
        const topo = await res.json()
        const geo = feature(topo, topo.objects.countries) as any
        if (mounted) setPolygons(geo.features)
      } catch {}
    })()
    return () => {
      mounted = false
    }
  }, [])

  // Configure globe controls + auto rotate
  useEffect(() => {
    const g = globeRef.current
    if (!g) return
    const controls = g.controls()
    controls.enableZoom = true
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.35
    g.pointOfView({ lat: 0, lng: 0, altitude: 3.6 }, 0)
  }, [])

  useEffect(() => {
    const g = globeRef.current
    if (!g) return
    const fit = () => {
      g.pointOfView({ lat: 0, lng: 0, altitude: 3.6 }, 0)
      g.controls().update?.()
    }
    const id = requestAnimationFrame(fit)
    window.addEventListener("resize", fit)
    return () => {
      cancelAnimationFrame(id)
      window.removeEventListener("resize", fit)
    }
  }, [])

  // Fly to selected partner
  useEffect(() => {
    const g = globeRef.current
    if (!g || !selectedPartner || !COORDS[selectedPartner]) return
    const [lon, lat] = COORDS[selectedPartner]
    g.pointOfView({ lat, lng: lon, altitude: 1.8 }, 1200)
  }, [selectedPartner])

  function handlePointClick(code: string) {
    onSelectPartner(code)
    router.push(`/partners/${code}`)
  }

  return (
    <div className="p-3">
      <div className="flex items-center justify-between px-2 py-2">
        <h3 className="text-sm font-semibold text-pretty">Global Trade Network (3D)</h3>
        <p className="text-xs text-muted-foreground">
          Zoom, drag, and click to explore. Arcs colored by Δ%, point size by value.
        </p>
      </div>

      <div className="relative aspect-[16/9] rounded-lg border bg-background overflow-hidden grid place-items-center">
        {data.length === 0 ? (
          <div className="text-center text-muted-foreground">
            <p className="text-sm">No trade data available</p>
            <p className="text-xs mt-1">Run data preprocessing to see the network</p>
          </div>
        ) : (
          <>
            <Globe
              ref={globeRef}
              backgroundColor="rgba(0,0,0,0)"
              globeImageUrl="https://unpkg.com/three-globe/example/img/earth-dark.jpg"
              bumpImageUrl="https://unpkg.com/three-globe/example/img/earth-topology.png"
              showAtmosphere
              atmosphereColor="rgba(120,170,255,0.4)"
              polygonsData={polygons}
              polygonCapColor={() => "rgba(56, 189, 248, 0.25)"}
              polygonSideColor={() => "rgba(0,0,0,0)"}
              polygonStrokeColor={() => "rgba(120, 120, 140, 0.28)"}
              polygonsTransitionDuration={300}
              // Points for partners (no text labels)
              pointsData={nodes}
              pointLat={(d: any) => d.lat}
              pointLng={(d: any) => d.lon}
              pointAltitude={(d: any) => 0.01 + size(d.value) / 100}
              pointColor={(d: any) => (d.selected ? "#38bdf8" : "#7c8aa0")}
              pointRadius={(d: any) => 0.25 + size(d.value) / 10}
              onPointClick={(d: any) => handlePointClick(d.code)}
              pointsMerge={true}
              // Animated arcs from India to partner
              arcsData={arcs}
              arcColor={(d: any) => [d.color, d.color]}
              arcStroke={(d: any) => (d.selected ? Math.min(3, d.thickness + 0.6) : d.thickness)}
              arcDashLength={0.5}
              arcDashGap={0.2}
              arcDashAnimateTime={1600}
              arcsTransitionDuration={0}
            />
            {/* Legend chip */}
            <div className="pointer-events-none absolute right-3 bottom-3 rounded-md bg-card/80 ring-1 ring-border px-2 py-1 text-xs">
              <span className="inline-block mr-2" style={{ color: "#10b981" }}>
                ▲
              </span>{" "}
              positive Δ
              <span className="inline-block ml-3 mr-2" style={{ color: "#ef4444" }}>
                ▼
              </span>{" "}
              negative Δ
            </div>
          </>
        )}
      </div>

      <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
        {nodes.map((n) => (
          <button
            key={n.code}
            onClick={() => handlePointClick(n.code)}
            className={`group inline-flex w-full items-center justify-between gap-2 rounded-md border px-3 py-2 text-left transition overflow-hidden ${
              n.selected ? "bg-primary text-primary-foreground" : "bg-card hover:bg-accent"
            }`}
            aria-pressed={n.selected}
          >
            <span className="text-sm font-medium truncate min-w-0">{n.name}</span>
            <span
              className="text-xs rounded-full px-2 py-0.5 shrink-0"
              style={{ backgroundColor: "transparent", color: n.change >= 0 ? "#10b981" : "#ef4444" }}
            >
              {(n.change * 100).toFixed(1)}%
            </span>
          </button>
        ))}
      </div>
    </div>
  )
})
