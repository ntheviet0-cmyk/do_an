"""Real artifact integration tests plus isolated mathematical edge cases."""
import csv
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ForecastRequest
from app.services import ArtifactAdapter, DataService, FILES, RESULT, aggregate, metrics, service


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


def test_startup_and_openapi(client):
    assert client.get('/api/health').json()['status'] == 'ok'
    assert '/api/forecast' in client.get('/openapi.json').json()['paths']


def test_full_metadata_from_artifact_not_raw_subset(client):
    data = client.get('/api/dataset/info').json()
    assert data['series'] == 30490
    assert [v['count'] for v in data['levels']] == [1, 3, 10, 70, 30490]
    assert data['raw_history_series'] == 100


def test_hierarchy_selectors(client):
    stores = client.get('/api/hierarchy/stores?state=TX').json()['values']
    assert stores == ['TX_1', 'TX_2', 'TX_3']
    items = client.get('/api/hierarchy/items?store=TX_1&department=FOODS_3&q=586').json()
    assert 'FOODS_3_586' in items['values']
    assert all(r['series_id'].endswith('_TX_1') for r in items['series'])
    assert client.get('/api/hierarchy/items?state=CA&store=TX_1').json()['total'] == 0


@pytest.mark.parametrize('name', list(FILES))
def test_available_models_and_real_predictions(client, name):
    info = client.get(f'/api/models/{name}').json()
    assert info['available'] and not info['live_inference']
    response = client.post('/api/forecast', json={'model': name, 'level': 'L12', 'series_id': 'FOODS_3_586_TX_1'})
    assert response.status_code == 200, response.text
    f = response.json()
    assert len(f['predictions']) == len(f['actuals']) == len(f['dates']) == 28
    assert f['dates'][0] == '2016-04-25' and f['dates'][-1] == '2016-05-22'
    with np.load(RESULT / FILES[name], allow_pickle=False) as a:
        index = np.flatnonzero(a['ids'] == 'FOODS_3_586_TX_1_evaluation')[0]
        np.testing.assert_array_equal(f['predictions'], a['y_pred_bu' if name == 'TFT' else 'test_pred'][index])
    assert f['execution_mode'] == 'artifact_replay'
    assert f['metrics'] == metrics(f['actuals'], f['predictions'])


@pytest.mark.parametrize('payload', [
    {'horizon': 27}, {'origin_day': 1885}, {'strategy': 'direct'},
    {'model': 'MISSING'}, {'level': 'L4'}, {'series_id': 'BOGUS'},
    {'model': 'LSTM', 'calibration': 'STORE_CAL'}, {'future_actuals': [1]}
])
def test_invalid_requests(client, payload):
    assert client.post('/api/forecast', json=payload).status_code == 422


def test_history_never_partial(client):
    assert client.get('/api/series/TOTAL/history?level=L1').status_code == 409
    f = client.post('/api/forecast', json={}).json()
    assert f['historical_sales'] == [] and not f['metadata']['history_available']
    f = client.post('/api/forecast', json={'level': 'L12', 'series_id': 'FOODS_3_001_CA_1'}).json()
    assert len(f['historical_sales']) == 56


@pytest.mark.parametrize('level,target', [('L1', 'TOTAL'), ('L2', 'TX'), ('L3', 'TX_1'), ('L9', 'TX_1|FOODS_3')])
def test_bottom_up_coherence(client, level, target):
    h = client.post('/api/hierarchy/analyze', json={'level': level, 'series_id': target, 'calibration': 'STORE_CAL'}).json()
    assert h['max_difference'] < 1e-6
    np.testing.assert_allclose(h['parent']['predictions'], np.sum([c['predictions'] for c in h['children']], axis=0))


def test_calibration_does_not_fit_on_test_labels(client, monkeypatch):
    service.factors.cache_clear()
    monkeypatch.setattr(service, 'truth', lambda *args: pytest.fail('Calibration must not access test truth'))
    factors = client.get('/api/calibration/factors').json()
    assert factors['source_period'][1] < 1914
    assert len(factors['factors']) == 10
    assert all(0.8 <= f['factor'] <= 1.2 and f['max_residual'] < 1e-5 for f in factors['factors'])


def test_metrics_pooled_and_zero_demand():
    a, p = np.array([[100., 100.], [1., 1.]]), np.array([[90., 90.], [0., 0.]])
    m = metrics(a, p)
    assert m['WMAPE_pct'] == pytest.approx(22 / 202 * 100)
    assert m['Bias_pct'] == pytest.approx(-22 / 202 * 100)
    assert metrics([0, 0], [0, 1])['WMAPE_pct'] is None
    assert metrics([0, 0], [0, 1])['Bias_pct'] is None
    assert metrics([1, 2], [1, 2])['RMSE'] == 0


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_metric_rejected(bad):
    with pytest.raises(ValueError):
        metrics([1, 2], [1, bad])


def test_vectorized_aggregation():
    np.testing.assert_array_equal(aggregate(np.array([[1., 2.], [3., 4.], [5., 6.]]), np.array([0, 1, 0]), 2), [[6, 8], [3, 4]])


def test_missing_artifact_reporting(monkeypatch, tmp_path):
    import app.services as module
    monkeypatch.setattr(module, 'RESULT', tmp_path)
    adapter = ArtifactAdapter('LSTM', DataService())
    with pytest.raises(FileNotFoundError):
        adapter.load()


@pytest.mark.parametrize('change', ['negative', 'nan', 'short'])
def test_corrupt_forecast_rejected(monkeypatch, tmp_path, change):
    import app.services as module
    data = service.data
    ids = data.metadata().id.to_numpy(dtype=str)
    values = np.zeros((len(ids), 27 if change == 'short' else 28), dtype=np.float32)
    if change == 'negative':
        values[0, 0] = -1
    if change == 'nan':
        values[0, 0] = np.nan
    np.savez(tmp_path / FILES['LSTM'], ids=ids, test_pred=values)
    monkeypatch.setattr(module, 'RESULT', tmp_path)
    with pytest.raises(ValueError):
        ArtifactAdapter('LSTM', data).load()


def test_comparison_and_csv_end_to_end(client):
    payload = {'level': 'L12', 'series_id': 'FOODS_3_586_TX_1'}
    comparison = client.post('/api/compare', json=payload).json()
    assert comparison['ranking_allowed'] and len(comparison['results']) == 4
    assert not comparison['unavailable']
    exported = client.post('/api/forecast/export', json=payload)
    assert exported.status_code == 200
    rows = list(csv.DictReader(io.StringIO(exported.content.decode('utf-8-sig'))))
    assert len(rows) == 28
    forecast = comparison['results'][0]
    assert float(rows[0]['prediction']) == forecast['predictions'][0]
    assert rows[0]['execution_mode'] == 'artifact_replay'


def test_runtime_level_metrics_and_archived_scope(client):
    rows = client.get('/api/metrics/by-level').json()
    assert len(rows) == 25
    assert all(r['WRMSSE'] is None for r in rows)
    archive = client.get('/api/metrics/summary').json()
    assert archive['runtime_recomputed'] is False
    assert all(r['split'] == 'test' and r['aggregation_scope'] == 'full_M5' for r in archive['rows'])


def test_missing_checkpoints_do_not_claim_inference(client):
    status = client.get('/api/system/status').json()
    assert all(not c['loaded'] for c in status['checkpoints'])
    assert all(not m['live_inference'] for m in status['models'])
