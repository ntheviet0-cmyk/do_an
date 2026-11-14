import { useEffect, useState } from 'react'
import { Play, Search } from 'lucide-react'
import { useApi } from '../api'
import type { Calibration, ForecastRequest, Level, Model, ModelInfo, Selectors } from '../types'
import { Button } from './ui/button'
import { ErrorBox } from './Common'

const levelNames: Record<Level, string> = { L1: 'L1 · Toàn hệ thống', L2: 'L2 · Bang', L3: 'L3 · Cửa hàng', L9: 'L9 · Cửa hàng × Ngành hàng', L12: 'L12 · Sản phẩm × Cửa hàng' }
export const defaultRequest: ForecastRequest = { model: 'TFT', level: 'L1', series_id: 'TOTAL', origin_day: 1913, horizon: 28, strategy: 'bottom_up', calibration: 'RAW' }
export function ForecastControls({ onRun, pending = false, tftOnly = false, initialLevel = 'L12' }: { onRun: (r: ForecastRequest) => void; pending?: boolean; tftOnly?: boolean; initialLevel?: Level }) {
  const models = useApi<ModelInfo[]>('/models')
  const [chosenModel, setModel] = useState<Model>('TFT')
  const [level, setLevel] = useState<Level>(initialLevel)
  const [state, setState] = useState('TX'), [chosenStore, setStore] = useState('TX_1'), [chosenDept, setDept] = useState('FOODS_3'), [chosenItem, setItem] = useState('FOODS_3_586')
  const [search, setSearch] = useState(''), [debounced, setDebounced] = useState('')
  const [chosenCal, setCal] = useState<Calibration>('RAW')
  useEffect(() => { const id = setTimeout(() => setDebounced(search), 250); return () => clearTimeout(id) }, [search])
  const states = useApi<Selectors>('/hierarchy/states')
  const stores = useApi<Selectors>(`/hierarchy/stores?state=${state}`)
  const store = stores.data?.values.includes(chosenStore) ? chosenStore : stores.data?.values[0] || ''
  const depts = useApi<Selectors>(`/hierarchy/departments?store=${store}&state=${state}`)
  const department = depts.data?.values.includes(chosenDept) ? chosenDept : depts.data?.values[0] || ''
  const items = useApi<Selectors>(`/hierarchy/items?store=${store}&department=${department}&state=${state}&q=${encodeURIComponent(debounced)}&limit=100`)
  const item = chosenItem || items.data?.values[0] || ''
  const model = tftOnly ? 'TFT' : chosenModel
  const info = models.data?.find(m => m.name === model)
  const calibration = info?.calibrations.includes(chosenCal) ? chosenCal : 'RAW'
  const series = level === 'L1' ? 'TOTAL' : level === 'L2' ? state : level === 'L3' ? store : level === 'L9' ? `${store}|${department}` : `${item}_${store}`
  const incomplete = !info?.available || !info.origins.length || (level !== 'L1' && !state) || (['L3','L9','L12'].includes(level) && !store) || (['L9','L12'].includes(level) && !department) || (level === 'L12' && (!item || items.isFetching || depts.isFetching || stores.isFetching))
  const select = (label: string, value: string, options: string[], change: (v: string) => void) => <label className="field"><span>{label}</span><select aria-label={label} value={value} onChange={e => change(e.target.value)} disabled={!options.length}>{!options.length && <option value="">Chưa có dữ liệu</option>}{options.map(v => <option key={v} value={v}>{v}</option>)}</select></label>
  return <form className="control-panel" onSubmit={e => { e.preventDefault(); if (!incomplete) onRun({ model, level, series_id: series, origin_day: info!.origins[0], horizon: 28, strategy: 'bottom_up', calibration }) }}>
    <div className="controls-grid">
      {select('Mô hình', model, (models.data || []).filter(m => m.available && (!tftOnly || m.name === 'TFT')).map(m => m.name), v => { setModel(v as Model); setCal('RAW') })}
      <label className="field"><span>Cấp dự báo</span><select aria-label="Cấp dự báo" value={level} onChange={e => setLevel(e.target.value as Level)}>{Object.entries(levelNames).map(([v,label]) => <option value={v} key={v}>{label}</option>)}</select></label>
      {select('Chiến lược', 'Bottom-Up', info?.strategies.includes('bottom_up') ? ['Bottom-Up'] : [], () => {})}
      {select('Calibration', calibration, info?.calibrations || [], v => setCal(v as Calibration))}
      {select('Forecast origin', info?.origins[0] ? `d${info.origins[0]}` : '', info?.origins.map(v => `d${v}`) || [], () => {})}
    </div>
    {level !== 'L1' && <div className="controls-grid target-grid">
      {select('Bang', state, states.data?.values || [], v => { setState(v); setStore(''); setDept(''); setItem(''); setSearch('') })}
      {['L3','L9','L12'].includes(level) && select('Cửa hàng', store, stores.data?.values || [], v => { setStore(v); setDept(''); setItem(''); setSearch('') })}
      {['L9','L12'].includes(level) && select('Ngành hàng', department, depts.data?.values || [], v => { setDept(v); setItem(''); setSearch('') })}
      {level === 'L12' && <><label className="field"><span><Search size={13}/> Tìm mã sản phẩm</span><input aria-label="Tìm mã sản phẩm" placeholder="Ví dụ: FOODS_3_586" value={search} onChange={e => { setSearch(e.target.value); setItem('') }}/></label>{select(`Sản phẩm (${items.data?.total ?? 0})`, item, Array.from(new Set([...(item ? [item] : []), ...(items.data?.values || [])])), setItem)}</>}
    </div>}
    <div className="control-footer"><div><span className="badge neutral">Historical Replay</span><span className="text-sm muted ml-3 break-all">{series}</span></div><Button type="submit" disabled={pending || incomplete}><Play size={15}/>{pending ? 'Đang xử lý…' : 'Chạy dự báo'}</Button></div>
    <ErrorBox error={models.error || states.error || stores.error || depts.error || items.error}/>
  </form>
}
