import type { ReactNode } from 'react'
import { AlertCircle, LoaderCircle } from 'lucide-react'
import { fmt } from '../lib/utils'
import type { ForecastResult } from '../types'

export function Panel({ title, subtitle, children, action }: { title?: string; subtitle?: string; children: ReactNode; action?: ReactNode }) {
  return <section className="panel">{(title || action) && <div className="panel-heading"><div><h2>{title}</h2>{subtitle && <p className="muted text-sm mt-1">{subtitle}</p>}</div>{action}</div>}{children}</section>
}
export function Loading() { return <div className="loading" role="status"><LoaderCircle className="animate-spin" size={22}/> Đang đọc dữ liệu nghiên cứu…</div> }
export function ErrorBox({ error }: { error: Error | null | undefined }) { return error ? <div role="alert" className="notice error"><AlertCircle size={18}/><span>{error.message}</span></div> : null }
export function Notice({ children }: { children: ReactNode }) { return <div className="notice"><AlertCircle size={18}/><div>{children}</div></div> }
export function Stat({ label, value, note }: { label: string; value: ReactNode; note?: string }) { return <div className="stat"><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div> }
export function ResultStats({ result }: { result: ForecastResult }) {
  const p = result.predictions, total = p.reduce((a,b) => a+b,0)
  return <><div className="stats-grid"><Stat label="Tổng dự báo · 28 ngày" value={fmt(total)}/><Stat label="Trung bình / ngày" value={fmt(total / 28)}/><Stat label="Cao nhất / thấp nhất" value={`${fmt(Math.max(...p))} / ${fmt(Math.min(...p))}`}/><Stat label="Series thành phần" value={fmt(result.metadata.series_count, 0)}/></div>{result.actuals && <div className="stats-grid metrics"><Stat label="RMSE ↓" value={fmt(result.metrics.RMSE)}/><Stat label="MAE ↓" value={fmt(result.metrics.MAE)}/><Stat label="WMAPE ↓" value={result.metrics.WMAPE_pct == null ? '—' : `${fmt(result.metrics.WMAPE_pct)}%`}/><Stat label="Forecast Bias" value={result.metrics.Bias_pct == null ? '—' : `${fmt(result.metrics.Bias_pct)}%`} note="Âm = dự báo thiếu"/></div>}</>
}
