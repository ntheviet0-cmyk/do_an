import { CheckCircle2, CircleOff, RefreshCw } from 'lucide-react'
import { useApi } from '../api'
import type { SystemStatus } from '../types'
import { ErrorBox, Loading, Notice, Panel, Stat } from '../components/Common'
import { Button } from '../components/ui/button'

export default function SystemInfo() {
  const status=useApi<SystemStatus>('/system/status'), health=useApi<{status:string}>('/health')
  const s=status.data
  return <><ErrorBox error={status.error || health.error}/>{status.isPending ? <Loading/> : s && <>
    <div className="stats-grid"><Stat label="Backend API" value={health.data?.status==='ok'?'Online':'—'}/><Stat label="Artifact registry" value={s.status}/><Stat label="Execution mode" value="Replay"/><Stat label="Inference device" value={s.device}/></div>
    <Panel title="Model & Artifact Registry" subtitle="Checkpoint hiện diện không đồng nghĩa đã được load hoặc hỗ trợ live inference." action={<Button variant="outline" onClick={()=>status.refetch()}><RefreshCw size={15}/>Kiểm tra lại</Button>}><div className="table-wrap"><table><thead><tr><th>Model</th><th>Trạng thái</th><th>Artifact</th><th>Origin</th><th>Live inference</th></tr></thead><tbody>{s.models.map(m=><tr key={m.name}><td><b>{m.name}</b><small className="block">{m.version}</small></td><td><span className={`badge ${m.available?'success':'neutral'}`}>{m.available?<CheckCircle2 size={12}/>:<CircleOff size={12}/>} {m.available?'Replay ready':'Unavailable'}</span>{m.reason && <p className="text-xs mt-2">{m.reason}</p>}</td><td><code>{m.artifact}</code></td><td>{m.origins.map(v=>`d${v}`).join(', ') || '—'}</td><td>{m.live_inference?'Enabled':'Unavailable'}</td></tr>)}</tbody></table></div></Panel>
    <div className="two-col"><Panel title="Dữ liệu & protocol"><dl className="details"><dt>Dataset</dt><dd>{s.dataset}</dd><dt>Training target</dt><dd>Đến d1885</dd><dt>Validation</dt><dd>d1886–d1913</dd><dt>Test</dt><dd>d1914–d1941</dd><dt>Scope</dt><dd>full_M5 · 30,490 item-store series</dd><dt>Last array validation</dt><dd>{s.last_validation ? new Date(s.last_validation).toLocaleString('vi-VN'):'Chưa xác minh'}</dd></dl></Panel><Panel title="Checkpoint inventory"><div className="stack">{s.checkpoints.map(c=><div className="checkpoint" key={c.file}><code>{c.file}</code><span className="badge neutral">{c.present?'Present':'Missing'} · {c.loaded?'Loaded':'Not loaded'}</span></div>)}</div><a className="text-link block mt-5" href="http://localhost:8000/docs" target="_blank" rel="noreferrer">Mở FastAPI OpenAPI documentation ↗</a></Panel></div>
    <Panel title="Giới hạn đã xác minh"><div className="stack">{s.limitations.map(l=><Notice key={l}>{l}</Notice>)}</div></Panel>
  </>}</>
}
