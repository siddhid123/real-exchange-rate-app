# Real Exchange Rate App

A Streamlit app that downloads monthly CPI and exchange-rate data from the IMF
(SDMX 3.0 API) and computes a bilateral real exchange rate between any two IMF
countries or the Euro area.

**Live app:** _paste your streamlit.app link here_

## Economics

| Symbol | Definition |
|---|---|
| S(t) | Units of Country 2 currency per 1 unit of Country 1 currency |
| P1(t), P2(t) | Monthly CPI of Country 1 and Country 2 |
| RER(t) | S(t) × P1(t) / P2(t) |
| RER* | Average of RER(t) over valid months in the selected sample |
| Real Index(t) | 100 × RER(t) / RER* (averages 100 over the sample) |

IMF exchange rates are local currency units per USD (X). The app converts them:
S(t) = X₂(t) / X₁(t). For the US, X = 1. Example: Country 1 = United Kingdom,
Country 2 = United States gives S = USD per GBP.

Nominal S(t) is plotted in actual currency units (not normalized). RER* is
recalculated whenever the sample dates change.

## Features

- Choose Country 1 and Country 2 (all ISO countries plus Euro area, IMF code `G163`)
- Monthly data from January 1976 where available
- Select the sample start and end month
- Chart view: Real, Nominal, or Both (separate y-axes)
- CSV download of all observations and calculations in the sample
- Built-in panel showing one RER calculation step by step

## Data

- API: `https://api.imf.org/external/sdmx/3.0`
- CPI: dataflow `CPI`, key `{ISO3}.CPI._T.IX.M`
- Exchange rate: dataflow `ER`, key `{ISO3}.XDC_USD.PA_RT.M`

IMF coverage varies by country and series may have gaps; months where any input
is missing are excluded from RER*. Live downloads can occasionally fail; reload
or use "Clear cached IMF data".

## Files

- `app.py` – the Streamlit app
- `requirements.txt` – Python dependencies (Python 3.12)

## Run locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```
