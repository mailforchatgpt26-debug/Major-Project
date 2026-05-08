/**
 * Client-side API base URL.
 * Default: same-origin `/backend` (rewritten by Next to FastAPI) so the UI works
 * whether you open the app at http://localhost:3000 or http://127.0.0.1:3000.
 * Override with NEXT_PUBLIC_API_URL (e.g. http://localhost:8000) if needed.
 */
export function getApiBaseUrl(): string {
  const u = process.env.NEXT_PUBLIC_API_URL?.trim()
  if (u) return u.replace(/\/$/, "")
  return "/backend"
}

/** Avoid stale dashboard data when API responses are cacheable. */
export const apiFetchInit: RequestInit = { cache: "no-store" }
