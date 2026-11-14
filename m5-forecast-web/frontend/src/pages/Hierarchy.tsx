import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { ArrowRight, CheckCircle2, RotateCcw } from 'lucide-react'
import { api } from '../api'
import type { ForecastRequest, HierarchyResult, Level } from '../types'
import { defaultRequest, ForecastControls } from '../components/ForecastControls'
import { Chart, ForecastChart } from '../components/Chart'
import { ErrorBox, Loading, Notice, Panel, ResultStats, Stat } from '../components/Common'
import { Button } from '../components/ui/button'
import { fmt } from '../lib/utils'

export default function Hierarchy() {
  const [params] = useSearchParams()
  const requested = params.get('level') as Level
  const [trail,setTrail] = useState<ForecastRequest[]>([])
  const [filter,setFilter] = useState('')
  const run = useMutation({mutationFn:(r:ForecastRequest)=>api<HierarchyResult>('/hierarchy/analyze',r)})
  const result = run.data
  const drill = (series: string) => { if (!result?.child_level || !run.variables) return; const req = {...run.variables,level:result.child_level,series_id:series};setTrail([...trail,run.variables]);setFilter('');run.mutate(req) }
  return <><ForecastControls initialLevel={['L1','L2','L3','L9','L12'].includes(requested) ? requested : 'L1'} onRun={r=>{setTrail([]);setFilter('');run.mutate(r)}} pending={run.isPending}/>
    <div className="breadcrumb"><Button variant="outline" size="sm" onClick={()=>{setTrail([]);run.mutate(defaultRequest)}}><RotateCcw size={14}/>Toàn hệ thống</Button>{trail.map((r,i)=><Button key={`${r.level}-${i}`} variant="ghost" size="sm" onClick={()=>{setTrail(trail.slice(0,i));run.mutate(r)}}>{r.series_id}<ArrowRight size={14}/></Button>)}{result && <span className="badge">{result.parent.series_id}</span>}</div>
    <ErrorBox error={run.error}/>{run.isPending && <Loading/>}{result && !run.isPending && <><ResultStats result={result.parent}/><Panel title={`${result.parent.level} · ${result.parent.series_id}`} subtitle="Cùng một dự báo L12 được cộng lên mọi cấp"><ForecastChart results={[result.parent]}/></Panel>
      {!result.leaf ? <><div className="stats-grid"><Stat label="Parent forecast total" value={fmt(result.parent.predictions.reduce((a,b)=>a+b,0))}/><Stat label="Sum of children" value={fmt(result.sum_children?.reduce((a,b)=>a+b,0))}/><Stat label="Max daily difference" value={result.max_difference.toExponential(3)}/><Stat label="Coherence" value={<span className="coherence"><CheckCircle2 size={20}/>{result.max_difference < 1e-6 ? 'Đạt' : 'Cần kiểm tra'}</span>}/></div>
      <Panel title={`Đóng góp của các series con · ${result.child_level}`} subtitle="Chọn một hàng để drill-down. Biểu đồ hiển thị tối đa 15 đóng góp lớn nhất."><Chart height={260} option={{tooltip:{trigger:'axis'},grid:{left:60,right:20,top:15,bottom:85},xAxis:{type:'category',data:[...result.children].sort((a,b)=>b.forecast_total-a.forecast_total).slice(0,15).map(c=>c.series_id),axisLabel:{rotate:25}},yAxis:{type:'value'},series:[{type:'bar',barMaxWidth:45,data:[...result.children].sort((a,b)=>b.forecast_total-a.forecast_total).slice(0,15).map(c=>c.forecast_total)}]}}/><input className="search-input" placeholder="Lọc series con…" aria-label="Lọc series con" value={filter} onChange={e=>setFilter(e.target.value)}/><div className="table-wrap bounded"><table><thead><tr><th>Series con</th><th>Tổng dự báo</th><th>WMAPE %</th><th>RMSE</th><th/></tr></thead><tbody>{result.children.filter(c=>c.series_id.toLowerCase().includes(filter.toLowerCase())).map(c=><tr key={c.series_id}><td><button className="text-link" onClick={()=>drill(c.series_id)}>{c.series_id}</button></td><td>{fmt(c.forecast_total)}</td><td>{fmt(c.metrics.WMAPE_pct)}</td><td>{fmt(c.metrics.RMSE)}</td><td><Button variant="ghost" size="sm" aria-label={`Drill ${c.series_id}`} onClick={()=>drill(c.series_id)}><ArrowRight size={15}/></Button></td></tr>)}</tbody></table></div></Panel></> : <Notice>Đây là cấp lá Item × Store. Không có series con.</Notice>}
      <Notice>Direct Forecasting chưa có artifact tương ứng. Tất cả kết quả ở đây là Bottom-Up từ cùng run L12; không thay thế dự báo Direct bằng tổng hợp.</Notice>
    </>}{!result && !run.isPending && <Panel><div className="empty-state"><h2>Đi từ tổng thể đến từng sản phẩm</h2><p>Chạy L1 hoặc chọn cấp bắt đầu. Sau đó bấm series con để khám phá cấu trúc nhu cầu.</p></div></Panel>}</>
}
