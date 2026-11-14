import { useState } from 'react'
import { useApi } from '../api'
import type { Explainability } from '../types'
import { Chart } from '../components/Chart'
import { ErrorBox, Loading, Notice, Panel } from '../components/Common'
import { fmt } from '../lib/utils'

export default function ExplainabilityPage() {
  const data=useApi<Explainability>('/explainability'), [kind,setKind]=useState('past')
  const rows=data.data?.rows.filter(r=>r.kind===kind).sort((a,b)=>b.weight-a.weight) || []
  return <><ErrorBox error={data.error}/>{data.isPending ? <Loading/> : data.data?.available ? <>
    <Notice>{data.data.note}</Notice><Panel title="Variable Selection · Đặc trưng được mô hình ưu tiên" subtitle={data.data.method} action={<select aria-label="Nhóm đặc trưng" value={kind} onChange={e=>setKind(e.target.value)}><option value="past">Historical features</option><option value="future">Known-future features</option></select>}><Chart height={430} option={{tooltip:{trigger:'axis'},grid:{left:175,right:35,top:15,bottom:40},xAxis:{type:'value',name:'Trọng số VSN'},yAxis:{type:'category',inverse:true,data:rows.slice(0,15).map(r=>r.feature)},series:[{type:'bar',barMaxWidth:18,data:rows.slice(0,15).map(r=>r.weight)}]}}/></Panel>
    <div className="two-col"><Panel title="Nguồn giải thích"><dl className="details"><dt>Model version</dt><dd>{data.data.model_version}</dd><dt>Artifact</dt><dd><code>{data.data.source}</code></dd><dt>Phạm vi</dt><dd>Global research export · validation samples</dd></dl></Panel><Panel title="Toàn bộ trọng số"><div className="table-wrap bounded"><table><thead><tr><th>Feature</th><th>Weight</th></tr></thead><tbody>{rows.map(r=><tr key={r.feature}><td>{r.feature}</td><td>{fmt(r.weight,6)}</td></tr>)}</tbody></table></div></Panel></div>
    <Notice>Chưa có attention tensor hay SHAP artifact đã xác minh cho run này. Trọng số VSN không phải bằng chứng nhân quả và không giải thích riêng forecast vừa chọn.</Notice>
  </> : <Notice>{data.data?.reason || 'Explainability chưa khả dụng.'}</Notice>}</>
}
