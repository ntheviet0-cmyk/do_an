# TFT M5 five-level training

Pipeline nay chi danh gia hierarchy `L1/L2/L3/L9/L12`. `WRMSSE_5levels` la
trung binh WRMSSE cua nam cap, khong phai official M5 WRMSSE 12 cap.

Encoder dung feature lifecycle/occurrence rieng cho L12 va context qua khu compact
tu L1/L2/L3/L9. Gia/promotion 28 ngay tuong lai duoc xem la khong biet truoc;
decoder chi nhan calendar, SNAP va event. Parent context duoc gather theo batch,
khong broadcast xuong 30.490 series.

Notebook mac dinh chay `full_5level`: 30.490 series, evaluation split va toi da
20 epoch. Pipeline se bao loi neu dataset hoac hierarchy khong du so series.

## Smoke test

```powershell
$env:PROFILE="smoke"
$env:SMOKE="1"
python kaggle/tft_m5_zero_test_src.py
```

## Full 30,490 series

Can co `sales_train_evaluation.csv`, `calendar.csv` va `sell_prices.csv` trong
`M5_DATA_ROOT`:

```powershell
$env:M5_DATA_ROOT="path/to/m5"
$env:PROFILE="full_5level"
$env:FORECAST_SPLIT="evaluation"
$env:OUT_DIR="outputs/full_5level"
python kaggle/tft_m5_zero_test_src.py
```

Tren Kaggle, chon **Add Input**, tim **M5 Forecasting Accuracy** va them dataset
truoc khi Run All. Pipeline tu dong tim file trong cac thu muc long ben duoi
`/kaggle/input`. Neu dung dataset upload rieng, dat trong cell dau:

```python
%env M5_DATA_ROOT=/kaggle/input/ten-dataset/thu-muc-chua-csv
```

Split evaluation dung:

- Train targets den `d_1885`.
- Validation `d_1886-d_1913`.
- Locked test `d_1914-d_1941`.

Calibration duoc fit tren 14 ngay dau validation va chon tren 14 ngay con lai.
Sau khi chon strategy, factor duoc refit tren toan bo validation de forecast test.

## Cau hinh chinh

| Bien | Mac dinh | Y nghia |
|---|---:|---|
| `FULL_EPOCHS` | `20` | So epoch toi da cua full profile; early stopping van co the dung som |
| `HIERARCHICAL_LOSS_WEIGHT` | `0.2` | Trong so auxiliary loss L1/L2/L3/L9; L12 dung base loss |
| `ZERO_LOSS_WEIGHT` | `0.1` | Trong so BCE cho head xac suat co ban |
| `BEST_METRIC` | `wrmsse5` | Chon checkpoint theo trung binh nam cap |
| `CALIBRATION_DAYS` | `14` | So ngay validation de fit calibration khi chon strategy |
| `SERIES_WEIGHT_POWER` | `0.5` | Luy thua recent dollar-sales weight |
| `MAX_TRAIN_ORIGINS` | `160` voi full profile | So forecast origins tren moi series |

## Output quan trong

- `history_test.csv`: base loss, hierarchical loss, learning rate va validation score.
- `metrics_test.csv`: metric model theo tung cap.
- `baseline_metrics_test.csv`: metric seasonal baseline theo tung cap.
- `summary_test.csv`: ket qua cac calibration strategy va gate baseline.
- `val_summary_test.csv`: strategy duoc chon ma khong cham tren ngay fit calibration.
- `config_test.json`: split, evaluator version va cau hinh de tai lap.

Chay unit test evaluator:

```powershell
python -m unittest kaggle/test_wrmsse5.py
```

Dong bo notebook tu source sau khi sua code:

```powershell
python kaggle/build_zero_test_notebook.py
```
