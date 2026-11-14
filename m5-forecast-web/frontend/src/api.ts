import { useQuery } from '@tanstack/react-query'
const BASE = import.meta.env.VITE_API_BASE_URL || ''
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${BASE}/api${path}`, body === undefined ? undefined : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!response.ok) {
    const data = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
  }
  return response.json()
}
export function useApi<T>(path: string) { return useQuery({ queryKey: [path], queryFn: () => api<T>(path) }) }
export async function downloadForecast(body: unknown) {
  const response = await fetch(`${BASE}/api/forecast/export`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!response.ok) throw new Error('Không thể xuất CSV. Kiểm tra kết nối API.')
  const url = URL.createObjectURL(await response.blob())
  const link = document.createElement('a'); link.href = url; link.download = 'm5-forecast.csv'; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
