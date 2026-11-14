import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Download, Database } from 'lucide-react'
import { api, downloadForecast } from '../api'
import type { ForecastRequest, ForecastResult } from '../types'
import { ForecastControls } from '../components/ForecastControls'
import { ErrorBox, Loading, Notice, Panel, ResultStats } from '../components/Common'
import { ForecastChart } from '../components/Chart'
import { Button } from '../components/ui/button'

export default function Forecast() {
  const run = useMutation({ mutationFn: (request: ForecastRequest) => api<ForecastResult>('/forecast', request) })
  const [exportError, setExportError] = useState<Error | null>(null)
  const result = run.data
  return <><ForecastControls onRun={r => { setExportError(null); run.mutate(r) }} pending={run.isPending}/><ErrorBox error={run.error || exportError}/>{run.isPending && <Loading/>}{result && !run.isPending ? <>
    <ResultStats result={result}/>
    {!result.metadata.history_available && <Notice>{result.metadata.history_note}</Notice>}
    <Panel title={`${result.model} · ${result.series_id}`} subtitle={`${result.historical_dates.length} ngày lịch sử khả dụng · 28 ngày dự báo · ${result.calibration} · Bottom-Up`} action={<Button variant="outline" onClick={() => downloadForecast(run.variables).catch(setExportError)}><Download size={16}/>Xuất CSV</Button>}><ForecastChart results={[result]}/></Panel>
    <div className="two-col"><Panel title="Nguồn dự báo"><div className="source-line"><Database size={18}/><code>{result.metadata.prediction_source}</code></div><p className="muted text-sm mt-3">Precomputed Forecast / Historical Replay. Nút chạy đọc đầu ra nghiên cứu đã lưu, không thực hiện inference mới.</p></Panel><Panel title="Quy ước đánh giá"><p className="muted text-sm">WMAPE được tính trên các quan sát gộp của mục tiêu đã chọn. Bias âm là dự báo thiếu. WRMSSE-5 toàn cấp xem tại Model Comparison.</p></Panel></div>
    <Notice>{result.metadata.actual_source}. Actuals chỉ dùng hiển thị và đánh giá, không đưa vào routine dự báo.</Notice>
  </> : !run.isPending && <Panel><div className="empty-state"><Database size={38}/><h2>Sẵn sàng khám phá nhu cầu</h2><p>Chọn sản phẩm hoặc cấp tổng hợp, sau đó chạy dự báo để xem kết quả nghiên cứu thật.</p><span className="badge">56 ngày encoder · 28 ngày horizon</span></div></Panel>}</>
}
