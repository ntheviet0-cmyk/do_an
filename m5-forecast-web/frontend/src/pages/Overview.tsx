import { Link } from 'react-router-dom'
import { ArrowRight, Layers3, ArrowDown, BrainCircuit, GitMerge, BarChart3 } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { api, useApi } from '../api'
import type { DatasetInfo, ForecastResult, ModelInfo } from '../types'
import { defaultRequest } from '../components/ForecastControls'
import { ErrorBox, Loading, Notice, Panel, Stat } from '../components/Common'
import { ForecastChart } from '../components/Chart'
import { fmt } from '../lib/utils'
import { Button } from '../components/ui/button'

const names = ['Toàn hệ thống', 'Bang', 'Cửa hàng', 'Cửa hàng × Ngành hàng', 'Sản phẩm × Cửa hàng']
export default function Overview() {
  const info = useApi<DatasetInfo>('/dataset/info'), models = useApi<ModelInfo[]>('/models')
  const total = useQuery({ queryKey: ['overview-forecast'], queryFn: () => api<ForecastResult>('/forecast', defaultRequest) })
  return <><section className="hero"><div><span className="eyebrow">M5 FORECASTING · RESEARCH PLATFORM</span><h2>Từ từng sản phẩm<br/>đến toàn hệ thống.</h2><p>Dự báo nhu cầu tồn kho đa cấp độ bằng Temporal Fusion Transformer có xét đến các yếu tố ngoại sinh.</p><Button asChild><Link to="/forecast">Khám phá dự báo <ArrowRight size={17}/></Link></Button></div><div className="hero-visual"><Layers3 size={80} strokeWidth={1}/><span>HIERARCHICAL<br/><b>DEMAND INTELLIGENCE</b></span><div className="hero-dot"/></div></section>
    <ErrorBox error={info.error || models.error}/>{info.isPending ? <Loading/> : info.data && <div className="stats-grid six"><Stat label="Item × Store series" value={fmt(info.data.series, 0)} note="Dữ liệu M5 thực tế"/><Stat label="Cửa hàng" value={info.data.stores}/><Stat label="Bang" value={info.data.states}/><Stat label="Horizon" value={`${info.data.horizon} ngày`}/><Stat label="Cấp nghiên cứu" value={info.data.levels.length}/><Stat label="Mô hình khả dụng" value={`${models.data?.filter(m => m.available).length ?? '—'} / ${models.data?.length ?? '—'}`}/></div>}
    <div className="overview-grid"><Panel title="Một hệ thống · Năm cấp độ" subtitle="Cấu trúc tổng hợp của nghiên cứu"><div className="hierarchy-flow">{info.data?.levels.map((v,i) => <div key={v.level}><Link to={`/hierarchy?level=${v.level}`} className={`hierarchy-node node-${i}`}><span className="badge neutral">{v.level}</span><div><strong>{fmt(v.count,0)}</strong><span>{names[i]}</span></div><ArrowRight size={16}/></Link>{i < 4 && <ArrowDown size={16} className="flow-arrow"/>}</div>)}</div></Panel>
      <Panel title="Nhu cầu toàn hệ thống" subtitle="Dự báo TFT RAW và test actuals · origin d1913"><ErrorBox error={total.error}/>{total.isPending ? <Loading/> : total.data && <><ForecastChart results={[total.data]}/>{!total.data.metadata.history_available && <Notice>{total.data.metadata.history_note}</Notice>}</>}</Panel></div>
    <div className="three-col">{[{icon: BrainCircuit,title:'Học từ ngữ cảnh',text:'TFT-Lite kết hợp lịch sử bán hàng, đặc trưng tĩnh và các biến tương lai đã biết theo pipeline nghiên cứu.'},{icon:GitMerge,title:'Nhất quán phân cấp',text:'Dự báo L12 được cộng lên L9, L3, L2 và L1 từ cùng một run, giữ nguyên kỳ dự báo.'},{icon:BarChart3,title:'Đánh giá minh bạch',text:'So sánh TFT và các baseline trên cùng mục tiêu, cùng ground truth và cùng 28 ngày đánh giá.'}].map(v => <Panel key={v.title}><v.icon size={22} className="accent"/><h3 className="mt-4">{v.title}</h3><p className="muted text-sm mt-2 leading-relaxed">{v.text}</p></Panel>)}</div>
  </>
}
