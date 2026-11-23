# US daily-data pilot

This pilot downloads Alpaca historical US equity daily bars into the existing
AlphaRefinery panel format. It is an engineering baseline for ten current US
symbols. It is not a survivorship-bias-free universe or a publication-quality
research dataset.

## Setup

Install the project with the US data dependency:

```bash
python -m pip install -e '.[us-data]'
```

Store credentials in the ignored project-root file `api_keys.env`:

```bash
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

The script loads only these two variables. It does not print them. Existing
environment variables take precedence over values in the file.

## Download and validate

The default command requests AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM,
JNJ, and WMT from 2023-01-01 through 2025-12-31, using daily SIP bars with
split adjustment:

```bash
python -m scripts.us_data_pilot download
```

The default outputs are kept under the Git-ignored `data/us/` directory:

- `alpaca_daily_2023-01-01_2025-12-31.parquet`
- `alpaca_daily_2023-01-01_2025-12-31.metadata.json`

The official `alpaca-py` SDK follows API pagination tokens and merges all pages.
If SIP access is denied, the command fails and reports the requested feed. It
never silently retries with IEX. To explicitly run an IEX engineering sample:

```bash
python -m scripts.us_data_pilot download --feed iex
```

## Fixed-factor evaluation

Run the existing public `alpha101.alpha003` factor, the seed used by the
`open_volume_correlation` family, and evaluate its cross-sectional Rank IC:

```bash
python -m scripts.us_data_pilot evaluate
```

The label is the five-trading-day close-to-close forward return:
`close[t+5] / close[t] - 1`. The output JSON records mean Rank IC, its standard
deviation, ICIR, observation count, and the panel checksum. This is a predictive
statistical IC evaluation. It is not an executable trading backtest and does not
model orders, liquidity, slippage, or portfolio construction.

## Panel format

The Parquet file uses a unique, sorted `MultiIndex(datetime, instrument)` and
contains `open`, `high`, `low`, `close`, `volume`, and Alpaca's reported `vwap`.
Daily timestamps are converted to their `America/New_York` trade date and saved
as timezone-naive midnight values. No VWAP or market-cap field is fabricated.
BaoStock filters and synthetic market-cap inference are disabled for this pilot.

The metadata JSON records provider, requested feed and adjustment, requested and
observed date ranges, symbols, row count, missing observations, and the Parquet
SHA-256 checksum.

## Known limitations

- The ten symbols are a current hand-picked engineering universe, so results
  have survivorship and selection bias.
- Split adjustment does not include dividend adjustment.
- Missing bars are reported relative to the union of observed trading dates;
  the pilot does not repair or fill them.
- The small universe makes cross-sectional Rank IC noisy and unsuitable as a
  standalone investment conclusion.
- Data entitlements and SIP availability depend on the Alpaca account.
