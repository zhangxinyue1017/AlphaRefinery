# US Research Pipeline Pilot

This engineering pilot reconnects the existing AlphaRefinery research pipeline to the frozen ten-stock Alpaca panel. It does not call an LLM, promote factors, or execute a trading strategy.

## Reproducible inputs

- Phase 1/1.5 baseline commit: `6ba6930`
- Dataset: `data/us/alpaca_daily_2023-01-01_2025-12-31.parquet`
- Dataset SHA-256: `9985f004a46d67b73d3e20478456d5f0e4d671061f45d1abda771478dbc8839b`
- Configuration: `config/us_research_pilot.yaml`
- Fixed candidate: `neg(ts_corr(cs_rank(div(open, delay(close, 1))), cs_rank(volume), 10))`
- Forward-return horizon: 5 trading days
- Groups / minimum stocks: 5 / 10
- Cost assumption: 10 bps
- Search window: 2023-01-03 through 2023-12-29
- Selection window: 2024-01-02 through 2025-12-31
- Final OOS window: unconfigured

The US configuration disables BaoStock filters, synthetic capitalization fields, proxy style exposures, neutralization, and automatic promotion. It uses an isolated SQLite archive under `artifacts/us_research_pilot/`.

## Commands

```bash
MPLCONFIGDIR=.matplotlib_cache python -m scripts.us_research_pilot
MPLCONFIGDIR=.matplotlib_cache python -m pytest -q
```

Each run writes a summary, evaluation CSV files, reports, search state, and archive records below `artifacts/us_research_pilot/`. These generated artifacts and the market data remain outside Git.

## Verified run

Run `run::open_volume_correlation::r1::a244e87939dd` completed the path:

`Seed -> Fixed Candidate -> Full Evaluation -> Research Decision -> Archive`

The selection-stage results were:

| Factor | Rank IC | Rank ICIR | Net annualized return | Net excess annualized return | Net Sharpe | Turnover | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `alpha101.alpha003` | 0.014178 | 0.041770 | unavailable | unavailable | unavailable | unavailable | parent |
| `us_pilot.open_gap_volume_corr_10` | 0.054368 | 0.175687 | 2.216573 | -0.321413 | 2.632111 | 0.623326 | `research_keep` |

The candidate-parent absolute correlation was 0.060460 in selection. No decorrelation target set was configured, so target-relative redundancy fields are unavailable. The candidate did not pass the existing winner gate: Rank ICIR was 0.176 versus the required 0.30. It was retained by the unchanged broad research gate because it beat the parent on Rank IC and Rank ICIR and supplied all four core metrics. The search state contains the retained child, and the isolated archive contains one run, one candidate, two evaluation rows, and one lineage edge. The pending promotion manifest has no entries, and no factor module was changed.

The search-window diagnostic reached a different result: candidate Rank IC was -0.067543, Rank ICIR was -0.209027, and the decision was `research_drop`. Only the configured selection window supplies the formal keep/drop decision; the final OOS stage remains absent.

## Ten-stock limitation

The universe is sufficient to exercise candidate computation and the full pipeline, but it is not sufficient for a reliable five-quantile research evaluation. Five groups contain only two stocks each, and `min_stocks=10` leaves no tolerance for a missing factor or label.

The parent factor is especially affected by mathematical ties. In the search window it had 236 dates with ten aligned observations, only 1–6 distinct values per date, and zero successful five-quantile assignments. In selection it had 488 aligned dates, again only 1–6 distinct values, and zero successful assignments. Pandas rejected every parent assignment because duplicate quantile edges were incompatible with five fixed labels. This is why parent Rank IC exists while its portfolio return, Sharpe, and turnover metrics are unavailable.

The fixed candidate had 227 successful group assignments out of 236 aligned search dates and 473 out of 488 aligned selection dates. Its full metrics therefore exist, but each group still has roughly two stocks and remains highly unstable. The reported annualized portfolio fields compound overlapping five-day forward returns as the legacy evaluator currently defines them. They are pipeline diagnostics, not executable backtest returns or publication-quality estimates.

## Remaining work for one real LLM round

The minimum remaining requirements are valid provider credentials outside Git, one explicitly selected provider/model, and a strict one-call run using this US seed configuration and isolated archive. The generated proposal must still pass the existing parser, expression validation, fixed search/selection protocol, redundancy checks, research gates, and archive flow. A broader and historically defensible US universe is required before treating portfolio metrics or selection outcomes as research evidence; no gate or search policy should be weakened to accommodate the ten-stock pilot.
