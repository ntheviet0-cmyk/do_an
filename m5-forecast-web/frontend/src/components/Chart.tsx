import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { LineChart, BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, MarkLineComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsCoreOption } from 'echarts/core'
import type { ForecastResult } from '../types'

echarts.use([LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, MarkLineComponent, CanvasRenderer])
export const colors = ['#6554c0', '#2b9d8f', '#e4a044', '#4f87d5', '#e0708a']
export function Chart({ option, height = 350, label = 'Biểu đồ dữ liệu nghiên cứu' }: { option: EChartsCoreOption; height?: number; label?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!ref.current) return
    const chart = echarts.init(ref.current)
    chart.setOption({ animation: false, color: colors, textStyle: { fontFamily: 'Inter, Segoe UI, sans-serif' }, ...option })
    const observer = new ResizeObserver(() => chart.resize()); observer.observe(ref.current)
    return () => { observer.disconnect(); chart.dispose() }
  }, [option])
  return <div ref={ref} style={{ height, width: '100%', minWidth: 0 }} role="img" aria-label={label} data-testid="chart"/>
}
export function ForecastChart({ results, history = true, labels }: { results: ForecastResult[]; history?: boolean; labels?: string[] }) {
  if (!results.length) return null
  const first = results[0], dates = history ? [...first.historical_dates, ...first.dates] : first.dates
  history = history && first.historical_dates.length > 0
  const prefix = history ? first.historical_dates.length : 0
  const series = [
    ...(history ? [{ name: 'Lịch sử thực tế', type: 'line', data: [...first.historical_sales, ...Array(28).fill(null)], itemStyle: { color: '#94a3b8' }, symbol: 'none', lineStyle: { width: 2 } }] : []),
    ...(first.actuals ? [{ name: 'Thực tế kỳ đánh giá', type: 'line', data: [...Array(prefix).fill(null), ...first.actuals], itemStyle: { color: '#25334b' }, symbol: 'none', lineStyle: { width: 2.5 } }] : []),
    ...results.map((r, i) => ({ name: labels?.[i] || `${r.model} · ${r.calibration}`, type: 'line', data: [...Array(prefix).fill(null), ...r.predictions], symbol: 'none', itemStyle: { color: colors[i] }, lineStyle: { width: 2.5, type: 'dashed' },
      ...(history && i === 0 ? { markLine: { silent: true, symbol: 'none', label: { formatter: 'Origin d1913', position: 'insideEndTop' }, lineStyle: { color: '#a39ac5' }, data: [{ xAxis: first.historical_dates.at(-1) }] } } : {}) })),
  ]
  return <Chart label="Lịch sử, dự báo và thực tế 28 ngày" height={390} option={{ tooltip: { trigger: 'axis', valueFormatter: (v: number) => v == null ? '—' : v.toLocaleString('vi-VN', { maximumFractionDigits: 2 }) }, legend: { bottom: 0, type: 'scroll' }, grid: { top: 28, left: 64, right: 24, bottom: 78 }, xAxis: { type: 'category', data: dates, axisLabel: { formatter: (v: string) => v.slice(5) }, boundaryGap: false }, yAxis: { type: 'value', name: 'Đơn vị bán', splitLine: { lineStyle: { color: '#e9edf3' } } }, dataZoom: [{ type: 'inside' }], series }}/>
}
