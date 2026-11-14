"""Experiment cells embedded in m5_tft_window_length_experiment.ipynb.

The notebook includes a frozen copy of TFT improved helpers; this file supplies
the experiment-specific cells. Run the notebook rather than this template.
"""

# %% config
import os
import sys
import json
import math
import random
import time
import gc
import hashlib
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

SMOKE = os.getenv('WINDOW_SMOKE', '0') == '1'
WINDOWS = sorted(set(int(x) for x in os.getenv('WINDOW_LENGTHS', '14,28,56,84,112').split(',')))
SEEDS = [int(x) for x in os.getenv('WINDOW_SEEDS', '42' if SMOKE else '42,123,2026').split(',')]
RUN_SEEDS = [int(x) for x in os.getenv('WINDOW_RUN_SEEDS', ','.join(map(str,SEEDS))).split(',')]
assert 56 in WINDOWS and min(WINDOWS) > 0 and len(SEEDS) == len(set(SEEDS))
assert RUN_SEEDS and set(RUN_SEEDS).issubset(SEEDS) and len(RUN_SEEDS)==len(set(RUN_SEEDS))
MAX_SEQ = max(WINDOWS)
N_SERIES = int(os.getenv('WINDOW_N_SERIES', '32' if SMOKE else '30490'))
MAX_EPOCHS = int(os.getenv('WINDOW_MAX_EPOCHS', '1' if SMOKE else '20'))
MAX_TRAIN_ORIGINS = int(os.getenv('WINDOW_MAX_ORIGINS', '2' if SMOKE else '160'))
BATCH = int(os.getenv('WINDOW_BATCH', '16' if SMOKE else '256'))
D_MODEL = int(os.getenv('WINDOW_D_MODEL', '16' if SMOKE else '64'))
HEADS = int(os.getenv('WINDOW_HEADS', '2' if SMOKE else '4'))
PATIENCE = int(os.getenv('WINDOW_PATIENCE', '3'))
H = 28
INNER_END, INNER_VAL_END = 1857, 1885
OUTER_END = 1913
# Optional additional refit cutoffs: e.g. 1801,1829; each forecasts next 28 days.
CONFIRM_CUTOFFS = sorted(set(int(x) for x in os.getenv('WINDOW_CONFIRM_CUTOFFS', '').split(',') if x))
assert all(MAX_SEQ + 2*H <= c < INNER_END and c + H <= OUTER_END for c in CONFIRM_CUTOFFS)
assert MAX_SEQ + H <= INNER_END
assert min(N_SERIES, MAX_EPOCHS, BATCH, D_MODEL, HEADS, PATIENCE) > 0 and D_MODEL % HEADS == 0
SEQ, SEED = 56, 42  # fixed data-selection seed, independent of training seeds
TRAIN_END, VAL_START, VAL_END = INNER_END, INNER_END+1, INNER_VAL_END
TEST_START, TEST_END = INNER_VAL_END+1, OUTER_END  # compatibility names; this is OUTER VALIDATION
CALIBRATION_DAYS = 14
SALES_FILE = 'sales_train_validation.csv'  # only d_1..d_1913; never competition test targets
PROFILE = 'window_experiment'
DROPOUT, LR, WEIGHT_DECAY = .1, .001, .0001
LOSS, TWEEDIE_P = 'tweedie', 1.3
SERIES_WEIGHT_POWER = .5
HIERARCHICAL_LOSS_WEIGHT, ZERO_LOSS_WEIGHT = .2, .1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'
NUM_WORKERS = 0
OUT_DIR = Path(os.getenv('WINDOW_OUT_DIR', 'window_experiment_outputs'))
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR = OUT_DIR / 'runs'
RUN_DIR.mkdir(exist_ok=True)

def safe_str(value):
    return str(value).encode('ascii', 'backslashreplace').decode('ascii')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
print('Device:', DEVICE, '| windows:', WINDOWS, '| seeds:', SEEDS, '| smoke:', SMOKE)
print('Outer validation: d_1886-d_1913. Competition test d_1914-d_1941 is not read.')

# %% experiment
assert len(MODEL_PAST_COLS) == 51 and len(FUT_COLS) == 10 and len(STATIC_COLS) == 5
if not SMOKE and N_SERIES == 30490:
    assert n_series == 30490, 'Full experiment requires all 30,490 series; use WINDOW_SMOKE=1 for local samples.'

# Recompute per-cutoff evaluator contexts explicitly. No calibration.
RAW_PRICE = PRICE_OBS
del PRICE, PRICE_REL, PROMO, SNAP, PRICE_AVAILABLE

def common_origins(cutoff):
    # Zero-based origin inclusive; next H targets end no later than cutoff-1.
    result = np.arange(MAX_SEQ-1, cutoff-H, dtype=np.int64)
    if MAX_TRAIN_ORIGINS > 0 and len(result) > MAX_TRAIN_ORIGINS:
        result = result[np.linspace(0, len(result)-1, MAX_TRAIN_ORIGINS).round().astype(int)]
    assert len(result) and result[0] >= MAX_SEQ-1 and result[-1]+H < cutoff
    return result.tolist()

def moments(values, cutoff):
    # Accumulate in float64, in series chunks to avoid a full panel copy.
    sums = np.zeros(values.shape[-1], np.float64)
    squares = np.zeros_like(sums)
    count = 0
    for start in range(0, len(values), 128):
        part = values[start:start+128, :cutoff].astype(np.float64)
        sums += part.sum(axis=(0,1))
        squares += np.square(part).sum(axis=(0,1))
        count += part.shape[0]*part.shape[1]
    mean = sums/count
    std = np.sqrt(np.maximum(squares/count-mean**2, 0))
    std[std < 1e-6] = 1
    return mean.astype(np.float32), std.astype(np.float32)

def cutoff_state(cutoff):
    base = moments(FEAT, cutoff)
    local = moments(LOCAL_STATE, cutoff)
    hierarchy = {level: moments(HIER_CONTEXT[level], cutoff) for level in HIER_LEVELS}
    revenue = (SALES[:, cutoff-28:cutoff].astype(np.float64)
               * np.nan_to_num(RAW_PRICE[:, cutoff-28:cutoff], nan=0).astype(np.float64)).sum(axis=1)
    weights = np.power(np.maximum(revenue, 0), SERIES_WEIGHT_POWER)
    weights = np.clip(weights/max(weights.mean(), 1e-9), .25, 4) if weights.sum()>0 else np.ones(n_series)
    contexts = build_contexts(GROUPS, SALES[:, :cutoff], RAW_PRICE[:, :cutoff])
    return {'base':base, 'local':local, 'hierarchy':hierarchy,
            'weights':weights.astype(np.float32), 'contexts':contexts}

class WindowDataset(Dataset):
    def __init__(self, seq, origins, state, training=False):
        self.seq, self.origins, self.state, self.training = seq, origins, state, training

    def __len__(self):
        return n_series*len(self.origins)

    def __getitem__(self, index):
        series, position = divmod(index, len(self.origins))
        origin = self.origins[position]
        sl = slice(origin-self.seq+1, origin+1)
        future_sl = slice(origin+1, origin+H+1)
        mean, std = self.state['base']
        past = (FEAT[series, sl][:, PAST_IDX]-mean[PAST_IDX])/std[PAST_IDX]
        future = (FEAT[series, future_sl][:, FUT_IDX]-mean[FUT_IDX])/std[FUT_IDX]
        mean, std = self.state['local']
        local = (LOCAL_STATE[series, sl].astype(np.float32)-mean)/std
        parent = []
        for level in HIER_LEVELS:
            mean, std = self.state['hierarchy'][level]
            parent.append((HIER_CONTEXT[level][GROUPS[level][series], sl]-mean)/std)
        past = np.concatenate([past, local]+parent, axis=1).astype(np.float32)
        target = SALES[series, future_sl]
        assert past.shape == (self.seq, 51) and future.shape == (H,10) and target.shape == (H,)
        weight = float(self.state['weights'][series]) if self.training else 1.
        return (torch.from_numpy(np.ascontiguousarray(past)),
                torch.from_numpy(np.ascontiguousarray(future)),
                torch.tensor(STATIC[series],dtype=torch.long),
                torch.from_numpy(np.ascontiguousarray(target)),
                torch.full((H,),weight,dtype=torch.float32),
                torch.tensor(series),torch.tensor(origin))

class SharedOriginSampler:
    def __init__(self, n_origins, seed):
        self.n_origins, self.seed, self.epoch = n_origins, seed, 0

    def __len__(self):
        return self.n_origins*math.ceil(n_series/BATCH)

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        self.epoch += 1
        for position in rng.permutation(self.n_origins):
            order = rng.permutation(n_series)
            for start in range(0, n_series, BATCH):
                yield [int(s)*self.n_origins+int(position) for s in order[start:start+BATCH]]

def new_model(seed):
    seed_everything(seed)
    return TFTLite(51,10,STATIC_CARD,d_model=D_MODEL,heads=HEADS,
                   dropout=DROPOUT,horizon=H).to(DEVICE)

def train_loader(dataset, seed):
    return DataLoader(dataset,batch_sampler=SharedOriginSampler(len(dataset.origins),seed),
                      num_workers=0,pin_memory=USE_AMP)

all_cutoffs = sorted(set([INNER_END, INNER_VAL_END]+CONFIRM_CUTOFFS+[c-H for c in CONFIRM_CUTOFFS]))
origin_rows = [{'cutoff':c, 'origin_index':o, 'last_observed_day':o+1,
                'target_start_day':o+2,'target_end_day':o+H+1}
               for c in all_cutoffs for o in common_origins(c)]
origin_frame = pd.DataFrame(origin_rows)
assert (origin_frame.target_end_day <= origin_frame.cutoff).all()

def array_digest(values):
    digest = hashlib.sha256()
    for start in range(0,len(values),128):
        digest.update(np.ascontiguousarray(values[start:start+128]).tobytes())
    return digest.hexdigest()

manifest = {'version':1,'source_sha256':SOURCE_HASH,'smoke':SMOKE,'windows':WINDOWS,'seeds':SEEDS,
    'n_series':n_series,'series_sha256':hashlib.sha256(json.dumps(series_ids).encode()).hexdigest(),
    'sales_sha256':array_digest(SALES),'features_sha256':array_digest(FEAT),
    'price_sha256':array_digest(RAW_PRICE),
    'inner_cutoff':INNER_END,'refit_cutoff':INNER_VAL_END,'outer_end':OUTER_END,
    'confirmation_cutoffs':CONFIRM_CUTOFFS,'max_epochs':MAX_EPOCHS,'patience':PATIENCE,
    'max_origins':MAX_TRAIN_ORIGINS,'batch':BATCH,'d_model':D_MODEL,'heads':HEADS,
    'lr':LR,'dropout':DROPOUT,'weight_decay':WEIGHT_DECAY,'loss':LOSS,'tweedie_p':TWEEDIE_P,
    'hier_loss_weight':HIERARCHICAL_LOSS_WEIGHT,'zero_loss_weight':ZERO_LOSS_WEIGHT,
    'series_weight_power':SERIES_WEIGHT_POWER,'past_features':MODEL_PAST_COLS,'future_features':FUT_COLS,
    'torch':torch.__version__,'numpy':np.__version__,'device':str(DEVICE),
    'origins_sha256':hashlib.sha256(origin_frame.to_csv(index=False).encode()).hexdigest()}
manifest_path = OUT_DIR/'experiment_manifest.json'
if manifest_path.exists():
    if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Resume manifest mismatch. Use a new WINDOW_OUT_DIR for changed experiments.')
else:
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
runtime = {'torch':torch.__version__,'device':str(DEVICE),
           'gpu_name':torch.cuda.get_device_name(0) if USE_AMP else None,
           'amp_init_scale':1024.,'requested_run_seeds':RUN_SEEDS}
(OUT_DIR/'latest_runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
origin_frame.to_csv(OUT_DIR/'shared_origins.csv',index=False)
pd.DataFrame([{'kind':'past','feature':x} for x in MODEL_PAST_COLS]
             +[{'kind':'future','feature':x} for x in FUT_COLS]).to_csv(OUT_DIR/'feature_manifest.csv',index=False)
print('Shared origins per cutoff:',origin_frame.groupby('cutoff').size().to_dict())

# %% preflight
inner_state = cutoff_state(INNER_END)
outer_state = cutoff_state(INNER_VAL_END)
# Check exact target equality and encoder suffix equality across all windows.
datasets = [WindowDataset(seq,common_origins(INNER_END),inner_state,True) for seq in WINDOWS]
reference = datasets[-1][0]
for seq, dataset in zip(WINDOWS,datasets):
    sample = dataset[0]
    assert torch.equal(sample[3],reference[3])
    assert torch.equal(sample[0],reference[0][-seq:])
    assert torch.equal(sample[1],reference[1])
    assert sample[6].item()+H < INNER_END

# One real training step at largest window, using the SAME batch size as every run.
probe_model = new_model(SEEDS[0])
probe_loader = train_loader(datasets[-1],SEEDS[0])
probe_opt = torch.optim.AdamW(probe_model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
probe_scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP,init_scale=1024.)
if USE_AMP:
    torch.cuda.reset_peak_memory_stats()
probe_stats = run_epoch(probe_model,[next(iter(probe_loader))],optimizer=probe_opt,scaler=probe_scaler)
assert np.isfinite(probe_stats['loss'])
print('Preflight passed:',probe_stats,'| parameters:',sum(p.numel() for p in probe_model.parameters()))
if USE_AMP:
    print('Largest-window peak allocated GB:',torch.cuda.max_memory_allocated()/2**30)
del probe_model,probe_loader,probe_opt,probe_scaler,datasets,reference
gc.collect()
if USE_AMP:
    torch.cuda.empty_cache()

# %% run
def atomic_json(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)

def fit_phase(seq, seed, cutoff, state, directory, fixed_epochs=None):
    directory.mkdir(parents=True,exist_ok=True)
    selection = fixed_epochs is None
    model = new_model(seed)
    dataset = WindowDataset(seq,common_origins(cutoff),state,True)
    loader = train_loader(dataset,seed)
    optimizer = torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    # Same LR trajectory for selection/refit, even when refit uses fewer epochs.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=LR,
        total_steps=MAX_EPOCHS*len(loader),pct_start=.3,anneal_strategy='cos')
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP,init_scale=1024.)
    eval_ds = WindowDataset(seq,[cutoff-1],state)
    best, selected, stale = float('inf'), 0, 0
    history = []
    started = time.perf_counter()
    if USE_AMP:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,(MAX_EPOCHS if selection else fixed_epochs)+1):
        epoch_start = time.perf_counter()
        stats = run_epoch(model,loader,optimizer=optimizer,scaler=scaler,scheduler=scheduler)
        if not np.isfinite(stats['loss']):
            raise FloatingPointError('Nonfinite training loss')
        score = None
        if selection:
            pred, actual, _ = predict_ds(model,eval_ds)
            _, score = evaluate_wrmsse5(actual,pred,state['contexts'])
            if not np.isfinite(score):
                raise FloatingPointError('Nonfinite inner-validation metric')
            if score < best-1e-4:
                best,selected,stale = score,epoch,0
            else:
                stale += 1
        if USE_AMP:
            torch.cuda.synchronize()
        history.append({'seq':seq,'seed':seed,'cutoff':cutoff,'epoch':epoch,
                        **stats,'inner_wrmsse5':score,'seconds':time.perf_counter()-epoch_start})
        pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
        print(f'seq={seq} seed={seed} cutoff={cutoff} epoch={epoch} loss={stats["loss"]:.4f} inner={score}',flush=True)
        if selection and stale>=PATIENCE:
            break
    elapsed = time.perf_counter()-started
    usage = {'seconds':elapsed,'epochs_run':len(history),
             'peak_allocated_gb':torch.cuda.max_memory_allocated()/2**30 if USE_AMP else None,
             'parameters':sum(p.numel() for p in model.parameters())}
    del loader,optimizer,scheduler,scaler
    return model,selected,usage

def evaluate_run(model,seq,seed,cutoff,state):
    dataset = WindowDataset(seq,[cutoff-1],state)
    pred,actual,_ = predict_ds(model,dataset)
    assert pred.shape == (n_series,H) and np.isfinite(pred).all() and (pred>=0).all()
    np.testing.assert_array_equal(actual,SALES[:,cutoff:cutoff+H])
    rows,overall = evaluate_wrmsse5(actual,pred,state['contexts'])
    metrics = [{'seq':seq,'seed':seed,'cutoff':cutoff,'level':r['level'],
                **{k:v for k,v in r.items() if k!='level'},'WRMSSE_5levels':overall} for r in rows]
    horizon = []
    for week in range(4):
        sl = slice(week*7,(week+1)*7)
        _,score = evaluate_wrmsse5(actual[:,sl],pred[:,sl],state['contexts'])
        horizon.append({'seq':seq,'seed':seed,'cutoff':cutoff,'forecast_week':week+1,'WRMSSE_5levels':score})
    zeros = (SALES[:,:cutoff]==0).mean(axis=1)
    buckets = pd.cut(zeros,[-.01,.25,.5,.75,.99,1.],labels=['<25%','25-50%','50-75%','75-99%','99-100%'])
    bucket_rows = []
    for label in buckets.categories:
        mask = buckets==label
        if mask.any():
            error = pred[mask]-actual[mask]
            denominator = max(float(actual[mask].sum()),1e-9)
            bucket_rows.append({'seq':seq,'seed':seed,'cutoff':cutoff,'zero_bucket':str(label),
                'n_series':int(mask.sum()),'MAE':float(np.abs(error).mean()),
                'WMAPE':float(np.abs(error).sum()/denominator),'Bias':float(error.sum()/denominator)})
    return pred,metrics,horizon,bucket_rows

def run_status_table():
    rows = []
    for path in sorted(RUN_DIR.glob('*/status.json')):
        rows.append(json.loads(path.read_text(encoding='utf-8')))
    pd.DataFrame(rows).to_csv(OUT_DIR/'run_status.csv',index=False)

# Stage-level resume: completed inner selection/refit stages are reused;
# interrupted stages restart from scratch, never reuse partially trained models.
for seed in RUN_SEEDS:
    for seq in WINDOWS:
        key = f'seq{seq}_seed{seed}'
        inner_dir = RUN_DIR/(key+'_inner')
        inner_dir.mkdir(exist_ok=True)
        selection_path = inner_dir/'selection.json'
        if selection_path.exists():
            selected = json.loads(selection_path.read_text())['selected_epoch']
        else:
            model,selected,usage = fit_phase(seq,seed,INNER_END,inner_state,inner_dir)
            del model
            atomic_json(selection_path,{'selected_epoch':selected,**usage})
        selection_usage = json.loads(selection_path.read_text())
        for cutoff in [INNER_VAL_END]+CONFIRM_CUTOFFS:
            directory = RUN_DIR/(key+f'_cutoff{cutoff}')
            directory.mkdir(exist_ok=True)
            status_path = directory/'status.json'
            if status_path.exists() and json.loads(status_path.read_text())['status']=='completed':
                print('Resume completed:',directory.name)
                continue
            # Confirmations choose epoch only before their own forecast cutoff.
            state = outer_state if cutoff==INNER_VAL_END else cutoff_state(cutoff)
            local_selected = selected
            try:
                atomic_json(status_path,{'seq':seq,'seed':seed,'cutoff':cutoff,'status':'running'})
                if cutoff!=INNER_VAL_END:
                    nested = cutoff-H
                    model,local_selected,confirmation_usage = fit_phase(seq,seed,nested,cutoff_state(nested),directory/'inner')
                    del model
                model,_,usage = fit_phase(seq,seed,cutoff,state,directory,fixed_epochs=local_selected)
                pred,metrics,horizon,buckets = evaluate_run(model,seq,seed,cutoff,state)
                np.savez_compressed(directory/'predictions.npz',ids=np.array(series_ids),y_pred=pred,
                                    y_true=SALES[:,cutoff:cutoff+H])
                pd.DataFrame(metrics).to_csv(directory/'metrics.csv',index=False)
                pd.DataFrame(horizon).to_csv(directory/'horizon.csv',index=False)
                pd.DataFrame(buckets).to_csv(directory/'zero_buckets.csv',index=False)
                atomic_json(status_path,{'seq':seq,'seed':seed,'cutoff':cutoff,'status':'completed',
                    'selected_epoch':local_selected,
                    'inner_seconds':selection_usage['seconds'] if cutoff==INNER_VAL_END else confirmation_usage['seconds'],
                    **usage})
                del model,pred
            except Exception as exc:
                atomic_json(status_path,{'seq':seq,'seed':seed,'cutoff':cutoff,'status':'failed',
                                        'error':f'{type(exc).__name__}: {exc}'})
                raise
            finally:
                run_status_table()
                gc.collect()
                if USE_AMP:
                    torch.cuda.empty_cache()

# %% summary
def collect(filename):
    frames = []
    for path in sorted(RUN_DIR.glob('*/status.json')):
        status = json.loads(path.read_text())
        if status['status']=='completed':
            frames.append(pd.read_csv(path.parent/filename))
    if not frames:
        raise RuntimeError('No completed runs')
    return pd.concat(frames,ignore_index=True)

metrics = collect('metrics.csv')
metrics.to_csv(OUT_DIR/'validation_metrics.csv',index=False)
collect('horizon.csv').to_csv(OUT_DIR/'horizon_metrics.csv',index=False)
collect('zero_buckets.csv').to_csv(OUT_DIR/'zero_bucket_metrics.csv',index=False)
statuses = pd.read_csv(OUT_DIR/'run_status.csv')
statuses['total_training_seconds'] = statuses.seconds+statuses.inner_seconds
statuses.to_csv(OUT_DIR/'resource_usage.csv',index=False)
statuses[['seq','seed','cutoff','selected_epoch']].to_csv(OUT_DIR/'selected_epochs.csv',index=False)
histories = [pd.read_csv(p).assign(stage=p.parent.name) for p in RUN_DIR.glob('*/history.csv')]
pd.concat(histories,ignore_index=True).to_csv(OUT_DIR/'training_history.csv',index=False)
scores = metrics[metrics.level.eq('L1')][['seq','seed','cutoff','WRMSSE_5levels']].copy()
reference = scores[scores.seq.eq(56)][['seed','cutoff','WRMSSE_5levels']].rename(columns={'WRMSSE_5levels':'reference_56'})
paired = scores.merge(reference,on=['seed','cutoff'],validate='many_to_one')
paired['delta_vs_56'] = paired.WRMSSE_5levels-paired.reference_56
paired['improvement_pct'] = 100*(paired.reference_56-paired.WRMSSE_5levels)/paired.reference_56
paired['wins_vs_56'] = paired.delta_vs_56 < 0
paired.to_csv(OUT_DIR/'paired_comparisons_vs_56.csv',index=False)
main = paired[paired.cutoff.eq(INNER_VAL_END)]
summary = main.groupby('seq').agg(mean_wrmsse5=('WRMSSE_5levels','mean'),
    std_wrmsse5=('WRMSSE_5levels','std'),mean_improvement_pct=('improvement_pct','mean'),
    wins_vs_56=('wins_vs_56','sum'),n_seeds=('seed','nunique')).reset_index()
l12 = metrics[(metrics.cutoff==INNER_VAL_END)&metrics.level.eq('L12')].groupby('seq').WRMSSE.mean()
summary['mean_l12_wrmsse'] = summary.seq.map(l12)
usage = statuses[statuses.cutoff.eq(INNER_VAL_END)].groupby('seq').agg(
    mean_refit_seconds=('seconds','mean'),mean_selected_epochs=('selected_epoch','mean'),
    mean_total_training_seconds=('total_training_seconds','mean'),
    peak_allocated_gb=('peak_allocated_gb','max'))
summary = summary.merge(usage,on='seq')
summary.to_csv(OUT_DIR/'window_summary.csv',index=False)
levels = metrics[metrics.cutoff.eq(INNER_VAL_END)].groupby(['seq','level']).WRMSSE.agg(['mean','std'])
levels.to_csv(OUT_DIR/'level_summary.csv')
print(summary.round(4).to_string(index=False))
fig,axes = plt.subplots(1,2,figsize=(12,4))
axes[0].errorbar(summary.seq,summary.mean_wrmsse5,yerr=summary.std_wrmsse5.fillna(0),marker='o',capsize=4)
axes[0].set(xlabel='Encoder days',ylabel='Validation WRMSSE-5',title='Mean ± seed standard deviation')
axes[1].plot(summary.seq,summary.mean_refit_seconds,marker='o')
axes[1].set(xlabel='Encoder days',ylabel='Seconds',title='Mean refit training time')
fig.tight_layout();fig.savefig(OUT_DIR/'window_accuracy_cost.png',dpi=140);plt.show()
fig,ax = plt.subplots(figsize=(9,5))
for level in LEVELS:
    part = levels.xs(level,level='level')['mean']
    ax.plot(part.index,part.values,marker='o',label=level)
ax.set(xlabel='Encoder days',ylabel='WRMSSE',title='Validation error by hierarchy level')
ax.legend();fig.tight_layout();fig.savefig(OUT_DIR/'window_level_metrics.png',dpi=140);plt.show()
complete = len(main)==len(WINDOWS)*len(SEEDS)
best = summary.sort_values('mean_wrmsse5').iloc[0]
report = ['# TFT encoder-window experiment',
    '**SMOKE: pipeline verification, not model-selection evidence.**' if SMOKE else 'Full experiment; compare matched seeds and forecast origins.',
    f'Completed main grid: {complete}. Outer validation: d_1886-d_1913; BU without calibration.',
    '```text'+chr(10)+summary.round(5).to_string(index=False)+chr(10)+'```',
    f'Lowest observed mean WRMSSE-5: SEQ={int(best.seq)}. This is a candidate, not automatic proof of superiority.',
    'Use paired deltas, per-level tradeoffs, seed stability and additional origins before freezing the choice. Seed standard deviation is not a confidence interval.',
    'The common feature bank retains lag_364/rolling history even for short encoders. This experiment measures input sequence length conditional on that bank.',
    'Inner validation selects epoch; independent refit predicts outer validation. Confirmations have their own nested epoch selection. No competition test targets are loaded.',
    'Additional origins are temporal sensitivity checks, not a fresh untouched holdout. Shorter windows may be preferred when differences are practically small.',
    'Resume checks code/data/config hashes. Interrupted stages restart; completed stages are reused.']
(OUT_DIR/'window_length_report.md').write_text((chr(10)*2).join(report),encoding='utf-8')
print('Saved:',safe_str(OUT_DIR.resolve()))
