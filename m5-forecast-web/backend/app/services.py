"""Read-only artifact integration. No training, fitting, or checkpoint execution."""
import json
import os
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .schemas import ForecastRequest

WEB_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(WEB_ROOT / '.env')
ROOT = Path(os.getenv('M5_REPO_ROOT', str(WEB_ROOT.parent))).resolve()
RESULT = Path(os.getenv('M5_RESULT_DIR', str(ROOT / 'result')))
RAW = Path(os.getenv('M5_DATA_DIR', str(ROOT / 'data' / 'raw')))
LEVELS = ['L1', 'L2', 'L3', 'L9', 'L12']
KEYS = {'RAW': 'y_pred_bu', 'STORE_CAL': 'y_pred_store_cal',
        'GLOBAL_CAL': 'y_pred_global_cal', 'SERIES_CAL': 'y_pred_series_cal'}
FILES = {'TFT': 'predictions_test (2).npz', 'LSTM': 'lstm_predictions.npz',
         'ARIMA': 'arima_predictions.npz', 'PROPHET': 'prophet_predictions.npz'}
PROTOCOL = {'id': 'full_m5_test_d1914_1941', 'train_end': 1885,
            'validation': [1886, 1913], 'test': [1914, 1941], 'origin_day': 1913,
            'horizon': 28, 'scope': 'full_M5', 'encoder_days': 56}


def metrics(actual, prediction):
    a, p = np.asarray(actual, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if a.shape != p.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError('Metrics require matching, finite, nonempty observations.')
    error = p - a
    denominator = float(np.abs(a).sum())
    return {'RMSE': float(np.sqrt(np.mean(error ** 2))), 'MAE': float(np.abs(error).mean()),
            'WMAPE_pct': float(np.abs(error).sum() / denominator * 100) if denominator else None,
            'Bias_pct': float(error.sum() / denominator * 100) if denominator else None,
            'WRMSSE': None}


def aggregate(values, codes, count):
    out = np.zeros((count, values.shape[1]), dtype=np.float64)
    np.add.at(out, codes, values)
    return out


class DataService:
    def __init__(self):
        self.lock = threading.RLock()
        self.validated_at = None

    @lru_cache(maxsize=1)
    def metadata(self):
        # The local raw CSV is a 100-series subset. Full hierarchy comes from
        # the forecast run itself; never label partial raw sums as full M5.
        with np.load(RESULT / FILES['TFT'], allow_pickle=False) as archive:
            ids = archive['ids'].astype(str)
        records = []
        for original in ids:
            canonical = original.removesuffix('_evaluation').removesuffix('_validation')
            parts = canonical.split('_')
            if len(parts) != 5 or parts[0] not in ['FOODS', 'HOBBIES', 'HOUSEHOLD'] or parts[3] not in ['CA', 'TX', 'WI']:
                raise ValueError(f'Unrecognized M5 item-store ID: {canonical}')
            records.append({'id': original, 'series_id': canonical, 'item_id': '_'.join(parts[:3]),
                            'dept_id': '_'.join(parts[:2]), 'cat_id': parts[0],
                            'state_id': parts[3], 'store_id': '_'.join(parts[3:])})
        frame = pd.DataFrame(records)
        if frame.series_id.duplicated().any():
            raise ValueError('Duplicate item-store IDs in sales metadata.')
        return frame.set_index('series_id', drop=False)

    @lru_cache(maxsize=1)
    def calendar(self):
        frame = pd.read_csv(RAW / 'calendar.csv')
        frame['day'] = frame.d.str.removeprefix('d_').astype(int)
        return frame.set_index('day')

    def dates(self, start, end):
        calendar = self.calendar()
        first_date = pd.Timestamp(calendar.iloc[0]['date'])
        first_day = int(calendar.index[0])
        expected = first_date + pd.to_timedelta(calendar.index.to_numpy() - first_day, unit='D')
        if not np.array_equal(expected.to_numpy(), pd.to_datetime(calendar.date).to_numpy()):
            raise ValueError('Calendar contains inconsistent daily indices.')
        # Dates alone may be extended deterministically. No future covariates
        # or sales values are created for absent calendar rows.
        return [(first_date + pd.Timedelta(days=d - first_day)).strftime('%Y-%m-%d') for d in range(start, end + 1)]

    @lru_cache(maxsize=1)
    def history(self):
        # Only the presentation encoder window is read, not the full sales panel.
        days = [f'd_{d}' for d in range(1858, 1914)]
        frame = pd.read_csv(RAW / 'sales_train_validation.csv', usecols=['id'] + days)
        frame.index = frame.id.str.replace(r'_(validation|evaluation)$', '', regex=True)
        result = frame.reindex(self.metadata().index)[days].to_numpy(dtype=np.float64)
        if np.isinf(result).any() or (result < 0).any():
            raise ValueError('Invalid historical sales.')
        return result

    @lru_cache(maxsize=5)
    def groups(self, level):
        m = self.metadata()
        keys = {'L1': np.full(len(m), 'TOTAL'), 'L2': m.state_id.to_numpy(),
                'L3': m.store_id.to_numpy(), 'L9': (m.store_id + '|' + m.dept_id).to_numpy(),
                'L12': m.series_id.to_numpy()}[level]
        labels, codes = np.unique(keys, return_inverse=True)
        return labels, codes

    def index(self, level, series_id):
        series_id = series_id.removesuffix('_evaluation').removesuffix('_validation')
        labels, codes = self.groups(level)
        found = np.flatnonzero(labels == series_id)
        if not len(found):
            raise ValueError(f'Unknown {level} series: {series_id}')
        return int(found[0]), codes == found[0]

    def selectors(self, kind, state=None, store=None, department=None, q='', limit=100):
        frame = self.metadata()
        for column, value in [('state_id', state), ('store_id', store), ('dept_id', department)]:
            if value:
                frame = frame[frame[column] == value]
        column = {'states': 'state_id', 'stores': 'store_id', 'departments': 'dept_id', 'items': 'item_id'}[kind]
        if q:
            frame = frame[frame[column].str.contains(q, case=False, regex=False)]
        values = sorted(frame[column].unique().tolist())
        return {'values': values[:limit], 'total': len(values), 'limit': limit,
                'series': frame[['series_id', 'item_id', 'store_id', 'dept_id', 'state_id']].head(limit).to_dict('records') if kind == 'items' else []}


class ArtifactAdapter:
    def __init__(self, name, data):
        self.name, self.data = name, data

    @lru_cache(maxsize=1)
    def load(self):
        with np.load(RESULT / FILES[self.name], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        ids = np.array([str(i).removesuffix('_evaluation').removesuffix('_validation') for i in arrays['ids']])
        expected = self.data.metadata().index
        if len(ids) != len(expected) or len(np.unique(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError(f'{self.name}: incomplete or duplicate series coverage.')
        order = pd.Index(ids).get_indexer(expected)
        arrays = {k: v[order] if v.ndim and len(v) == len(ids) else v for k, v in arrays.items()}
        for key, a in arrays.items():
            if a.dtype.kind in 'fiu':
                if not np.isfinite(a).all():
                    raise ValueError(f'{self.name}/{key}: NaN or Inf.')
                if a.ndim == 2 and a.shape != (len(expected), 28):
                    raise ValueError(f'{self.name}/{key}: incomplete horizon.')
                if ('pred' in key or key == 'y_true') and (a < 0).any():
                    raise ValueError(f'{self.name}/{key}: negative values; no silent clipping.')
        self.data.validated_at = datetime.now(timezone.utc).isoformat()
        return arrays

    def predictions(self, calibration):
        if self.name != 'TFT' and calibration != 'RAW':
            raise ValueError('Calibration artifacts are only available for TFT.')
        key = KEYS[calibration] if self.name == 'TFT' else 'test_pred'
        if key not in self.load():
            raise ValueError(f'Calibration {calibration} unavailable.')
        return self.load()[key]


class ForecastService:
    def __init__(self):
        self.data = DataService()
        self.adapters = {name: ArtifactAdapter(name, self.data) for name in FILES}

    @lru_cache(maxsize=1)
    def verify_protocol(self):
        manifest = json.loads((RESULT / 'baseline_manifest.json').read_text(encoding='utf-8'))
        config = json.loads((RESULT / 'config_test (2).json').read_text(encoding='utf-8'))
        expected = {'train_end': 1885, 'val_start': 1886, 'val_end': 1913, 'test_start': 1914,
                    'test_end': 1941, 'horizon': 28, 'seq': 56, 'n_series': 30490,
                    'aggregation_scope': 'full_M5'}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError('Baseline manifest does not match the registered protocol; comparison is disabled.')
        if any(config.get(k) != v for k, v in {'n_series': 30490, 'seq': 56, 'horizon': 28,
                                              'forecast_split': 'evaluation', 'profile': 'full_5level'}.items()):
            raise ValueError('TFT config does not match the registered full-M5 evaluation protocol.')
        return manifest

    def models(self):
        result = []
        for name, adapter in self.adapters.items():
            try:
                self.verify_protocol()
                a = adapter.load()
                adapter.predictions('RAW')
                available, reason = True, None
            except (OSError, ValueError, KeyError) as exc:
                a, available, reason = {}, False, str(exc).replace(str(ROOT), '<repository>')
            result.append({'name': name, 'available': available, 'reason': reason,
                           'execution_mode': 'artifact_replay', 'live_inference': False,
                           'live_reason': 'Checkpoint/preprocessing compatibility has not been validated.',
                           'strategies': ['bottom_up'] if available else [],
                           'calibrations': [k for k, v in KEYS.items() if v in a] if name == 'TFT' else (['RAW'] if available else []),
                           'origins': [1913] if available else [], 'levels': LEVELS if available else [],
                           'artifact': FILES[name], 'version': 'full_5level', 'protocol': PROTOCOL})
        return result

    def validate(self, request):
        self.verify_protocol()
        if request.origin_day != 1913:
            raise ValueError('This registry supports only origin d1913 (test d1914–d1941).')
        if request.strategy != 'bottom_up':
            raise ValueError('Direct multilevel artifacts are not available in this repository.')

    @lru_cache(maxsize=100)
    def grouped(self, model, calibration, level):
        labels, codes = self.data.groups(level)
        return aggregate(self.adapters[model].predictions(calibration), codes, len(labels))

    @lru_cache(maxsize=5)
    def truth(self, level):
        labels, codes = self.data.groups(level)
        return aggregate(self.adapters['TFT'].load()['y_true'], codes, len(labels))

    def forecast(self, request: ForecastRequest):
        self.validate(request)
        with self.data.lock:
            index, mask = self.data.index(request.level, request.series_id)
            p = self.grouped(request.model, request.calibration, request.level)[index]
            a = self.truth(request.level)[index]
            history_rows = self.data.history()[mask]
            history_complete = bool(np.isfinite(history_rows).all())
            h = history_rows.sum(axis=0) if history_complete else np.array([])
        return {'model': request.model, 'execution_mode': 'artifact_replay', 'strategy': 'bottom_up',
                'calibration': request.calibration, 'level': request.level, 'series_id': str(self.data.groups(request.level)[0][index]),
                'origin_day': 1913, 'horizon': 28, 'dates': self.data.dates(1914, 1941),
                'historical_dates': self.data.dates(1858, 1913) if history_complete else [], 'historical_sales': h.tolist(),
                'predictions': p.tolist(), 'actuals': a.tolist(), 'metrics': metrics(a, p),
                'metadata': {'protocol': PROTOCOL, 'prediction_source': FILES[request.model],
                             'actual_source': 'predictions_test (2).npz:y_true; raw evaluation CSV unavailable',
                             'series_count': int(mask.sum()), 'uncertainty': None,
                             'history_available': history_complete,
                             'history_note': '56-day raw CSV history.' if history_complete else 'Local raw sales has only 100 series. Full target history is unavailable; no partial aggregate is shown.',
                             'dates_note': 'Daily dates extended from verified M5 calendar anchor; future covariates not synthesized.',
                             'wrmsse_note': 'Target metrics omit WRMSSE: research score is level-wide, not this target.',
                             'bias_convention': '100 * sum(predicted - actual) / sum(abs(actual)); negative = underforecast',
                             'calibration': self.calibration_info() if request.calibration != 'RAW' else None}}

    def calibration_info(self):
        return {'source_period': [1886, 1913], 'source': 'kaggle/tft_m5_zero_test_src.py: refit on all validation days',
                'provenance': 'Period is source-code documented; validation TFT predictions were not exported, so fitting cannot be independently replayed.',
                'factor_method': 'Recovered from existing calibrated/raw predictions, verified pointwise; no labels used.'}

    @lru_cache(maxsize=1)
    def factors(self):
        self.verify_protocol()
        raw = self.adapters['TFT'].predictions('RAW').astype(np.float64)
        cal = self.adapters['TFT'].predictions('STORE_CAL').astype(np.float64)
        labels, codes = self.data.groups('L3')
        result = []
        for i, label in enumerate(labels):
            r, c = raw[codes == i], cal[codes == i]
            factor = float(c.sum() / r.sum()) if r.sum() else 1.0
            residual = float(np.max(np.abs(c - r * factor)))
            if not 0.8 - 1e-6 <= factor <= 1.2 + 1e-6 or not np.allclose(c, r * factor, atol=1e-5, rtol=1e-6):
                raise ValueError('Store calibration is not a consistent clipped multiplicative scaling.')
            result.append({'store': str(label), 'factor': factor, 'max_residual': residual})
        return {'factors': result, **self.calibration_info()}

    def hierarchy(self, request):
        parent = self.forecast(request)
        position = LEVELS.index(request.level)
        if position == 4:
            return {'parent': parent, 'child_level': None, 'children': [], 'max_difference': 0.0, 'leaf': True}
        child_level = LEVELS[position + 1]
        _, mask = self.data.index(request.level, request.series_id)
        labels, codes = self.data.groups(child_level)
        child_indices = np.unique(codes[mask])
        predictions = self.grouped(request.model, request.calibration, child_level)
        actual = self.truth(child_level)
        children = [{'series_id': str(labels[i]), 'forecast_total': float(predictions[i].sum()),
                     'predictions': predictions[i].tolist(), 'metrics': metrics(actual[i], predictions[i])} for i in child_indices]
        difference = np.asarray(parent['predictions']) - predictions[child_indices].sum(axis=0)
        return {'parent': parent, 'child_level': child_level, 'children': children,
                'max_difference': float(np.abs(difference).max()), 'difference': difference.tolist(),
                'sum_children': predictions[child_indices].sum(axis=0).tolist(), 'leaf': False}

    def level_metrics(self):
        result = []
        for model in self.models():
            if not model['available']:
                continue
            for calibration in (['RAW', 'STORE_CAL'] if model['name'] == 'TFT' else ['RAW']):
                for level in LEVELS:
                    p = self.grouped(model['name'], calibration, level)
                    a = self.truth(level)
                    result.append({'model': model['name'], 'calibration': calibration, 'level': level,
                                   'scope': 'full_M5', 'period': [1914, 1941], **metrics(a, p)})
        # Archived WRMSSE remains explicitly separated from runtime computed metrics.
        return result

    def archived_metrics(self):
        path = RESULT / 'comparison_all_levels.csv'
        if not path.exists():
            return {'rows': [], 'source': None}
        frame = pd.read_csv(path)
        frame = frame[(frame.split == 'test') & (frame.aggregation_scope == 'full_M5')]
        return {'rows': json.loads(frame.to_json(orient='records')), 'source': path.name,
                'period': [1914, 1941], 'definition': 'Research WRMSSE-5 (L1/L2/L3/L9/L12), not official 12-level M5.',
                'runtime_recomputed': False}


service = ForecastService()
