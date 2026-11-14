import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { api, useApi } from '../api'
import type { ArchivedMetrics, Comparison, ForecastRequest, Model, ModelInfo } from '../types'
import { ForecastControls } from '../components/ForecastControls'
import { Chart, colors, ForecastChart } from '../components/Chart'
import { ErrorBox, Loading, Notice, Panel } from '../components/Common'
import { fmt } from '../lib/utils'

export default function ComparisonPage() {
  const [selected,setSelected] = useState<Model[]>(['TFT','LSTM','ARIMA','PROPHET'])
  const [metric,setMetric] = useState<'RMSE'|'MAE'|'WMAPE_pct'|'Bias_pct'>('WMAPE_pct')
  const [ascending,setAscending] = useState(true)
  const models = useApi<ModelInfo[]>('/models'), archive = useApi<ArchivedMetrics>('/metrics/summary')
  const run = useMutation({ mutationFn: (r: ForecastRequest) => api<Comparison>('/compare', {...r, models:selected}) })
  const results = run.data?.results || []
  const sorted = [...results].sort((a,b) => ((a.metrics[metric] == null ? Infinity : metric === 'Bias_pct' ? Math.abs(a.metrics[metric]!) : a.metrics[metric]!) - (b.metrics[metric] == null ? Infinity : metric === 'Bias_pct' ? Math.abs(b.metrics[metric]!) : b.metrics[metric]!)) * (ascending ? 1 : -1))
  const best = [...results].sort((a,b) => Math.abs(a.metrics[metric] ?? Infinity)-Math.abs(b.metrics[metric] ?? Infinity))[0]?.model
  return <><Panel title="Mô hình tham gia" subtitle="Kỳ đánh giá cố định d1914–d1941 · cùng ground truth · cùng scope mục tiêu"><div className="model-checks">{models.data?.map(m => <label key={m.name}><input type="checkbox" checked={selected.includes(m.name)} disabled={!m.available} onChange={e => setSelected(e.target.checked ? [...selected,m.name] : selected.filter(v => v !== m.name))}/><b>{m.name}</b><span className="badge neutral">{m.available ? 'Replay' : 'Unavailable'}</span></label>)}</div></Panel>
    <ForecastControls onRun={r => run.mutate(r)} pending={run.isPending || !selected.length}/><ErrorBox error={run.error || models.error}/>{run.isPending && <Loading/>}{run.data?.unavailable.map(v => <Notice key={v.model}>{v.model}: {v.reason}</Notice>)}
    {!!results.length && !run.isPending && <><Panel title="So sánh dự báo trên cùng mục tiêu" subtitle={`${results[0].level} · ${results[0].series_id} · Historical Replay`}><ForecastChart results={results} history={false}/></Panel>
      {!run.data?.ranking_allowed && <Notice>Không đủ điều kiện xếp hạng do protocol không tương thích.</Notice>}
      <Panel title="Sai số trên mục tiêu đã chọn" subtitle="Bấm tiêu đề cột để sắp xếp. Bias được so sánh theo trị tuyệt đối."><div className="table-wrap"><table><thead><tr><th>Mô hình</th>{(['RMSE','MAE','WMAPE_pct','Bias_pct'] as const).map(k => <th key={k}><button onClick={() => {setAscending(k === metric ? !ascending : true);setMetric(k)}}>{k.replace('_pct',' %')} {metric === k ? ascending ? '↑' : '↓' : '↕'}</button></th>)}</tr></thead><tbody>{sorted.map(r => <tr key={r.model} className={run.data?.ranking_allowed && r.model === best ? 'best-row' : ''}><td><b>{r.model}</b> <small>{r.calibration}</small>{r.model === best && run.data?.ranking_allowed && <span className="badge">Tốt nhất · {metric.replace('_pct','')}</span>}</td>{(['RMSE','MAE','WMAPE_pct','Bias_pct'] as const).map(k => <td key={k}>{fmt(r.metrics[k])}</td>)}</tr>)}</tbody></table></div></Panel>
      <div className="two-col"><Panel title="Sai số tuyệt đối theo ngày"><Chart height={280} option={{ tooltip:{trigger:'axis'}, legend:{bottom:0},grid:{left:50,right:20,top:20,bottom:65},xAxis:{type:'category',data:results[0].dates.map(d=>d.slice(5))},yAxis:{type:'value'},series:results.map(r=>({name:r.model,type:'line',symbol:'none',data:r.predictions.map((v,i)=>Math.abs(v-r.actuals![i]))})) }}/></Panel><Panel title="Forecast Bias (%)"><Chart height={280} option={{tooltip:{trigger:'axis'},grid:{left:60,right:20,top:20,bottom:45},xAxis:{type:'category',data:results.map(r=>r.model)},yAxis:{type:'value'},series:[{type:'bar',data:results.map((r,i)=>({value:r.metrics.Bias_pct,itemStyle:{color:colors[i]}})),barMaxWidth:42}]}}/></Panel></div>
      <Panel title="Độ lớn sai số" subtitle={`${metric.replace('_pct',' %')} · mục tiêu đã chọn`}><Chart height={240} option={{tooltip:{trigger:'axis'},grid:{left:60,right:20,top:15,bottom:40},xAxis:{type:'category',data:results.map(r=>r.model)},yAxis:{type:'value'},series:[{type:'bar',barMaxWidth:45,data:results.map((r,i)=>({value:r.metrics[metric],itemStyle:{color:colors[i]}}))}]}}/></Panel>
      <Panel title="WRMSSE-5 · Toàn cấp (artifact nghiên cứu)" subtitle="Điểm này bao phủ toàn bộ series của cấp, không phải WRMSSE của mục tiêu đang chọn."><ErrorBox error={archive.error}/><div className="table-wrap"><table><thead><tr><th>Mô hình nghiên cứu</th><th>Cấp</th><th>WRMSSE</th></tr></thead><tbody>{archive.data?.rows.filter(r => r.level === results[0].level && ['TFT_SELECTED_STORE_CAL','LSTM','PROPHET','ARIMA_ROBUST'].includes(r.model)).map(r=><tr key={r.model}><td>{r.model}</td><td>{r.level}</td><td>{fmt(r.WRMSSE,5)}</td></tr>)}</tbody></table></div><p className="muted text-xs mt-3">Nguồn: {archive.data?.source} · test d1914–d1941 · full_M5 · {archive.data?.definition} Không tính lại runtime, không dùng bảng này để xếp hạng mục tiêu riêng.</p></Panel>
    </>}
  </>
}
