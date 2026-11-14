# Seasonality evidence report

**Train-only M5 analysis.**

Cutoff d_1885; series 30490; subset request 0.

L12 is a stratified sample. Subset aggregates are not full M5 totals.

## Summary

```text
       n_analyzed  median_acf7  median_weekly_strength  median_annual_gain
level                                                                     
L1              1       0.7327                  0.7610              0.0161
L12            30       0.1077                  0.0987             -0.0378
L2              3       0.6890                  0.7064              0.0165
L3             10       0.6837                  0.6903              0.0104
L9             70       0.5252                  0.5092             -0.0021
```

## Interpretation

Read CSV evidence and representative plots together. Labels are descriptive heuristics, not significance tests.

Weekly candidates: sin_wday/cos_wday/is_weekend and lag_7/14/28/35/56. Annual candidates: sin_month/cos_month, sin_year/cos_year and lag_364.

Annual evidence is provisional: few years, trend/event confounding; lag 364 also preserves weekday. Monthly profiles do not prove a 30-day cycle.

Price coverage is not inventory availability. TFT improved uses 51 past, 10 future and 5 static inputs.

## Next experiment

Compare full TFT against removing weekly calendar, weekly lags and annual features separately. Match series, origins, seeds, loss, budget and early stopping. Evaluate uncalibrated BU WRMSSE-5 and per-level metrics on validation.

Rolling/hierarchy features retain seasonal information: ablations measure incremental feature contributions. Do not select features on test.