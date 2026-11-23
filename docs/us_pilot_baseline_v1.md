# US Pilot Baseline v1 audit

Audit date: 2026-11-23. This is a predictive cross-sectional IC check, not a trading backtest.

## Baseline identity

- The generated JSON records the Git HEAD, working-tree state, and SHA-256 fingerprints for the relevant source files. The commit that preserves this baseline is recorded with the Phase 2 US research configuration.
- Frozen panel: `data/us/alpaca_daily_2023-01-01_2025-12-31.parquet`
- Panel SHA-256: `9985f004a46d67b73d3e20478456d5f0e4d671061f45d1abda771478dbc8839b`
- Coverage: 7,520 rows, 752 trading dates, 10 symbols, 2023-01-03 through 2025-12-31; Alpaca SIP daily bars with split adjustment.
- Factor: `alpha101.alpha003 = -1 * correlation(rank(open), rank(volume), 10)`.
- Label: `close[t+5] / close[t] - 1`; minimum five valid stocks per date.
- Metric: daily cross-sectional Spearman Rank IC; `ICIR = mean(IC) / std(IC, ddof=1)`.

Reproduce the audit with:

```bash
python -m scripts.us_data_pilot audit
```

The command writes the machine-readable result, including every corrected exclusion date, to `data/us/alpaca_daily_2023-01-01_2025-12-31.baseline_v1.json`.

## Why the previous result had 722 observations

The previous count was `752 - 25 - 5 = 722`.

The factor was constant across the ten stocks on 25 dates, so Spearman correlation was undefined:

```text
2023-01-03, 2023-01-04, 2023-01-05, 2023-01-06, 2023-01-09,
2023-01-10, 2023-01-11, 2023-01-12, 2023-01-13, 2024-06-26,
2024-07-12, 2024-07-15, 2024-07-16, 2024-07-17, 2024-07-18,
2024-07-19, 2024-07-22, 2024-12-02, 2024-12-03, 2024-12-23,
2024-12-24, 2024-12-27, 2024-12-30, 2024-12-31, 2025-01-02
```

The first nine are the 10-day factor warm-up, filled with zero by the public factor implementation. The remaining 16 also had one unique factor value across the universe.

The final five trading dates had no complete 5-day forward label:

```text
2025-12-24, 2025-12-26, 2025-12-29, 2025-12-30, 2025-12-31
```

The audit found that 722 was not a valid final baseline. The rolling-correlation operator left tiny floating residuals for zero-variance windows and split mathematical ties through a denominator epsilon. Spearman ranking treated those residuals as signal on 158 additional dates. The minimal fix explicitly returns zero for zero-variance windows, divides normally elsewhere, and normalizes at the operator's existing `1e-12` tolerance. No factor formula or window changed.

## Validated result

After the numerical correctness fix, 183 dates have a constant/undefined factor and five dates lack the label, leaving 564 valid daily observations.

| Metric | Framework | Independent pandas calculation |
|---|---:|---:|
| Rank IC observations | 564 | 564 |
| Mean Rank IC | 0.0193171167 | 0.0193171167 |
| Rank IC sample standard deviation | 0.3336829295 | 0.3336829295 |
| ICIR | 0.0578906350 | 0.0578906350 |
| Annualized ICIR (`sqrt(252)`) | 0.9189853414 | 0.9189853414 |
| Positive rate | 0.5141843972 | 0.5141843972 |

The independent path uses pandas cross-sectional minimum ranks, pandas 10-day rolling correlation, an independently shifted close matrix, explicit average ranks followed by Pearson correlation for Spearman IC, and a separate sample-standard-deviation ICIR. Factor values agree within `1.01e-12`; labels agree exactly; daily IC values agree within `1.12e-16`.

## Alignment, data, and loader checks

- The factor at date `t` uses only `open` and `volume` from `t-9` through `t`. Because full daily volume is required, the factor is available only after the close of date `t`. The label uses closes at `t` and `t+5`; no label value enters factor calculation.
- The panel has a unique sorted `(datetime, instrument)` index, no missing or non-finite field values, positive OHLCV, valid OHLC bounds, and VWAP inside the daily low/high range. Maximum absolute close-to-close return is 24.36%, with no unadjusted split-sized close jump.
- NVDA on 2024-06-10 has `high=195.95` while open/close/VWAP are near 121. A read-only SIP re-fetch returned the same source bar. It does not affect `alpha003` or the close-based label, but remains a provider-data anomaly.
- The US entry point calls `build_data(..., apply_filters=False, infer_cap_fields=False)`. The loaded inputs contain no market-cap field. BaoStock-specific filters and synthetic market-cap inference are absent from this evaluation.
- The panel checksum and modification time remained unchanged throughout the audit.

## Limitations and validation

The ten-symbol universe is an engineering sample with survivorship bias. Provider split adjustment is an ex-post corporate-action convention. Daily-volume timing must be respected in any later execution study. The NVDA high anomaly remains unresolved. These metrics do not establish a tradable strategy or publication-quality evidence.

`python -m pytest -q` passes all 10 tests. Calling the standalone `pytest` executable in this shell fails during collection because that entry point omits the repository root from `sys.path`; no test executes in that invocation.
