export type Level = 'L1' | 'L2' | 'L3' | 'L9' | 'L12'
export type Model = 'TFT' | 'LSTM' | 'ARIMA' | 'PROPHET'
export type Calibration = 'RAW' | 'STORE_CAL' | 'GLOBAL_CAL' | 'SERIES_CAL'
export interface ForecastRequest { model: Model; level: Level; series_id: string; origin_day: number; horizon: number; strategy: 'bottom_up'; calibration: Calibration }
export interface Metrics { RMSE: number; MAE: number; WMAPE_pct: number | null; Bias_pct: number | null; WRMSSE: number | null }
export interface ForecastResult extends ForecastRequest { execution_mode: string; dates: string[]; historical_dates: string[]; historical_sales: number[]; predictions: number[]; actuals: number[] | null; metrics: Metrics; metadata: { series_count: number; prediction_source: string; actual_source: string; bias_convention: string; wrmsse_note: string; history_available: boolean; history_note: string } }
export interface ModelInfo { name: Model; available: boolean; reason: string | null; execution_mode: string; live_inference: boolean; live_reason: string; strategies: string[]; calibrations: Calibration[]; origins: number[]; levels: Level[]; artifact: string; version: string }
export interface DatasetInfo { name: string; series: number; items: number; stores: number; states: number; horizon: number; levels: { level: Level; count: number }[]; history_window: number[]; raw_evaluation_available: boolean }
export interface Selectors { values: string[]; total: number; series: { series_id: string; item_id: string; store_id: string; dept_id: string; state_id: string }[] }
export interface Comparison { results: ForecastResult[]; unavailable: { model: string; reason: string }[]; ranking_allowed: boolean }
export interface HierarchyResult { parent: ForecastResult; child_level: Level | null; children: { series_id: string; forecast_total: number; predictions: number[]; metrics: Metrics }[]; max_difference: number; sum_children?: number[]; leaf: boolean }
export interface FactorInfo { factors: { store: string; factor: number; max_residual: number }[]; source_period: number[]; source: string; provenance: string; factor_method: string }
export interface LevelMetrics extends Metrics { model: Model; calibration: Calibration; level: Level; scope: string; period: number[] }
export interface ArchivedMetrics { rows: { model: string; level: Level; WRMSSE: number }[]; source: string | null; definition: string; runtime_recomputed: boolean }
export interface SystemStatus { status: string; models: ModelInfo[]; execution_mode: string; device: string; last_validation: string | null; dataset: string; checkpoints: { file: string; present: boolean; loaded: boolean }[]; limitations: string[] }
export interface Explainability { available: boolean; reason?: string; rows: { feature: string; weight: number; kind: string }[]; source: string; model_version: string; method: string; note: string }
