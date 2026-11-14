import csv
import io
import os
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import numpy as np

from .schemas import CompareRequest, ForecastRequest, ForecastResponse, Level
from .services import FILES, LEVELS, PROTOCOL, RAW, RESULT, service

app = FastAPI(title='M5 Hierarchical Demand Forecasting', version='1.0.0',
              description='Read-only, genuine historical forecast replay. No training or synthetic predictions.')
app.add_middleware(CORSMiddleware, allow_origins=os.getenv('M5_CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(','),
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


@app.exception_handler(ValueError)
async def invalid_request(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={'detail': str(exc)})


@app.exception_handler(FileNotFoundError)
async def missing_artifact(request: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=503, content={'detail': f'Required data/artifact is missing: {os.path.basename(exc.filename or "unknown")}'})


@app.get('/api/health')
def health():
    return {'status': 'ok', 'service': 'm5-forecast-api'}


@app.get('/api/models')
def models():
    return service.models()


@app.get('/api/models/{name}')
def model(name: str):
    found = next((m for m in service.models() if m['name'] == name.upper()), None)
    if found is None:
        raise HTTPException(404, 'Unknown model')
    return found


@app.get('/api/system/status')
def status():
    model_status = service.models()
    return {'status': 'ready' if all(m['available'] for m in model_status) else 'degraded',
            'models': model_status, 'protocol': PROTOCOL, 'execution_mode': 'artifact_replay', 'device': 'CPU / NumPy',
            'last_validation': service.data.validated_at,
            'dataset': 'M5 Forecasting Accuracy / sales_train_validation + exported test actuals',
            'checkpoints': [{'file': name, 'present': (RESULT / name).exists(), 'loaded': False} for name in ['model_test (2).pt', 'lstm_baseline.pt']],
            'limitations': ['Live checkpoint inference is not enabled: preprocessing compatibility unverified.',
                            'Direct multilevel forecast artifacts are unavailable.',
                            'Test actuals come from the TFT artifact; sales_train_evaluation.csv is absent.',
                            'Local raw sales has only 100 series. History is hidden for targets without complete raw coverage.',
                            'WRMSSE research scores are archived, not recalculated from missing evaluator contexts.',
                            'Calibration fit period is source-documented; original validation TFT forecasts are unavailable.']}


@app.get('/api/dataset/info')
def dataset():
    m = service.data.metadata()
    return {'name': 'M5 Forecasting – Accuracy', 'series': len(m), 'items': int(m.item_id.nunique()),
            'stores': int(m.store_id.nunique()), 'states': int(m.state_id.nunique()), 'horizon': 28,
            'levels': [{'level': level, 'count': len(service.data.groups(level)[0])} for level in LEVELS],
            'metadata_source': 'Full forecast artifact IDs', 'raw_history_series': int(np.isfinite(service.data.history()).all(axis=1).sum()),
            'protocol': PROTOCOL, 'history_window': [1858, 1913], 'raw_evaluation_available': (RAW / 'sales_train_evaluation.csv').exists()}


@app.get('/api/hierarchy/{kind}')
def selectors(kind: str, state: str | None = None, store: str | None = None, department: str | None = None,
              q: str = '', limit: Annotated[int, Query(ge=1, le=500)] = 100):
    if kind not in ['states', 'stores', 'departments', 'items']:
        raise HTTPException(404, 'Unknown selector')
    return service.data.selectors(kind, state, store, department, q, limit)


@app.get('/api/hierarchy')
def hierarchy_get():
    return dataset()['levels']


@app.post('/api/hierarchy/analyze')
def hierarchy_analyze(request: ForecastRequest):
    return service.hierarchy(request)


@app.get('/api/series/{series_id}')
def series(series_id: str, level: Level = 'L12'):
    _, mask = service.data.index(level, series_id)
    return {'series_id': series_id, 'level': level, 'series_count': int(mask.sum()),
            'members': service.data.metadata()[mask].head(20).to_dict('records')}


@app.get('/api/series/{series_id}/history')
def history(series_id: str, level: Level = 'L12'):
    import numpy as np
    _, mask = service.data.index(level, series_id)
    if not np.isfinite(service.data.history()[mask]).all():
        raise HTTPException(409, 'Complete historical sales are unavailable for this target in the 100-series local raw subset.')
    return {'dates': service.data.dates(1858, 1913), 'sales': service.data.history()[mask].sum(axis=0).tolist()}


@app.post('/api/forecast', response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    return service.forecast(request)


@app.post('/api/evaluate')
def evaluate(request: ForecastRequest):
    f = service.forecast(request)
    return {'metrics': f['metrics'], 'metadata': f['metadata']}


@app.post('/api/compare')
def compare(request: CompareRequest):
    results, unavailable = [], []
    for name in dict.fromkeys(request.models):
        try:
            payload = request.model_dump(exclude={'models'})
            payload.update(model=name, calibration=request.calibration if name == 'TFT' else 'RAW')
            results.append(service.forecast(ForecastRequest(**payload)))
        except (ValueError, OSError, KeyError) as exc:
            unavailable.append({'model': name, 'reason': str(exc)})
    compatible = bool(results) and all(r['dates'] == results[0]['dates'] and r['actuals'] == results[0]['actuals']
                                       and r['series_id'] == results[0]['series_id'] for r in results)
    return {'results': results, 'unavailable': unavailable, 'ranking_allowed': compatible, 'protocol': PROTOCOL}


@app.get('/api/calibration/methods')
def calibration_methods():
    return {'models': {m['name']: m['calibrations'] for m in service.models()}, **service.calibration_info()}


@app.get('/api/calibration/factors')
def factors():
    return service.factors()


@app.get('/api/metrics/summary')
def metrics_summary():
    return service.archived_metrics()


@app.get('/api/metrics/by-level')
def by_level():
    return service.level_metrics()


@app.get('/api/explainability')
def explanations():
    import pandas as pd
    path = RESULT / 'feature_importance_test (2).csv'
    if not path.exists():
        return {'available': False, 'reason': 'Feature importance artifact missing.', 'rows': []}
    return {'available': True, 'rows': pd.read_csv(path).to_dict('records'), 'source': path.name,
            'model_version': 'TFT V2 / full_5level (co-located config)',
            'method': 'Static-context Variable Selection Network weights, averaged over validation samples (research export).',
            'note': 'Global research export, not a per-request explanation or causal importance. Artifact/checkpoint hash linkage was not exported.'}


@app.post('/api/forecast/export')
def export(request: ForecastRequest):
    f = service.forecast(request)
    out = io.StringIO(newline='')
    writer = csv.writer(out)
    writer.writerow(['date', 'day', 'prediction', 'actual', 'model', 'calibration', 'level', 'series_id', 'execution_mode', 'protocol'])
    for i, date in enumerate(f['dates']):
        writer.writerow([date, 1914 + i, f['predictions'][i], f['actuals'][i], f['model'], f['calibration'], f['level'], f['series_id'], f['execution_mode'], PROTOCOL['id']])
    return Response(content='\ufeff' + out.getvalue(), media_type='text/csv; charset=utf-8',
                    headers={'Content-Disposition': 'attachment; filename="m5-forecast.csv"'})
