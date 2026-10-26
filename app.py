"""
Real Exchange Rate App
----------------------
Streamlit app that downloads monthly CPI and exchange-rate data from the
IMF SDMX 3.0 API and computes a bilateral real exchange rate.

Definitions
    S(t)   = units of Country 2 currency per 1 unit of Country 1 currency
    P1(t)  = CPI of Country 1,  P2(t) = CPI of Country 2
    RER(t) = S(t) * P1(t) / P2(t)
    RER*   = average of RER(t) over valid months in the selected sample
    Real Index(t) = 100 * RER(t) / RER*

IMF exchange rates are local currency units per USD (X_i). Therefore
    S(t) = X_2(t) / X_1(t)
e.g. Country 1 = UK, Country 2 = US:  S = 1 / (GBP per USD) = USD per GBP.
"""

import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_URL = "https://api.imf.org/external/sdmx/3.0"
START_MONTH = pd.Timestamp("1976-01-01")
CPI_FLOW = "CPI"
FX_FLOW = "ER"
TIMEOUT = 60

st.set_page_config(page_title="Real Exchange Rate App", page_icon="💱", layout="wide")

# Currency codes for labels (anything missing falls back to "<ISO3> currency")
CURRENCY = {
    "USA": "USD", "GBR": "GBP", "G163": "EUR", "JPN": "JPY", "CHN": "CNY",
    "CAN": "CAD", "AUS": "AUD", "NZL": "NZD", "CHE": "CHF", "SWE": "SEK",
    "NOR": "NOK", "DNK": "DKK", "ISL": "ISK", "KOR": "KRW", "IND": "INR",
    "IDN": "IDR", "MYS": "MYR", "SGP": "SGD", "THA": "THB", "PHL": "PHP",
    "VNM": "VND", "PAK": "PKR", "BGD": "BDT", "LKA": "LKR", "HKG": "HKD",
    "MEX": "MXN", "BRA": "BRL", "ARG": "ARS", "CHL": "CLP", "COL": "COP",
    "PER": "PEN", "URY": "UYU", "ZAF": "ZAR", "NGA": "NGN", "EGY": "EGP",
    "KEN": "KES", "GHA": "GHS", "MAR": "MAD", "TUR": "TRY", "ISR": "ILS",
    "SAU": "SAR", "ARE": "AED", "RUS": "RUB", "POL": "PLN", "CZE": "CZK",
    "HUN": "HUF", "ROU": "RON", "BGR": "BGN", "UKR": "UAH", "KAZ": "KZT",
    "DEU": "EUR", "FRA": "EUR", "ITA": "EUR", "ESP": "EUR", "NLD": "EUR",
    "BEL": "EUR", "AUT": "EUR", "PRT": "EUR", "IRL": "EUR", "FIN": "EUR",
    "GRC": "EUR",
}

FALLBACK_COUNTRIES = {
    "USA": "United States", "GBR": "United Kingdom", "JPN": "Japan",
    "CHN": "China", "CAN": "Canada", "AUS": "Australia", "CHE": "Switzerland",
    "DEU": "Germany", "FRA": "France", "ITA": "Italy", "ESP": "Spain",
    "KOR": "Korea", "IND": "India", "MEX": "Mexico", "BRA": "Brazil",
    "ZAF": "South Africa", "TUR": "Turkey", "SWE": "Sweden", "NOR": "Norway",
}


@st.cache_data(show_spinner=False)
def country_options() -> dict:
    """ISO3 -> display name, plus the Euro area (IMF code G163)."""
    try:
        import pycountry

        opts = {c.alpha_3: getattr(c, "common_name", c.name) for c in pycountry.countries}
    except Exception:
        opts = dict(FALLBACK_COUNTRIES)
    opts.update({"USA": "United States", "GBR": "United Kingdom"})
    opts["G163"] = "Euro area"
    return dict(sorted(opts.items(), key=lambda kv: kv[1]))


def currency_label(iso3: str, names: dict) -> str:
    return CURRENCY.get(iso3, f"{names.get(iso3, iso3)} currency")


# --------------------------------------------------------------------------
# IMF data access
# --------------------------------------------------------------------------
_PERIOD_RE = re.compile(r"^\s*(\d{4})\s*-?\s*M?\s*(\d{1,2})")


def parse_period(value) -> pd.Timestamp | None:
    """Parse IMF monthly periods such as '1976-M01', '1976M01', '1976-01'."""
    m = _PERIOD_RE.match(str(value))
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        return None
    return pd.Timestamp(year=year, month=month, day=1)


def _to_series(periods, values) -> pd.Series:
    idx = [parse_period(p) for p in periods]
    s = pd.Series(pd.to_numeric(pd.Series(list(values)), errors="coerce").values, index=idx)
    s = s[[i is not None for i in s.index]]
    s.index = pd.DatetimeIndex(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.dropna()


def _parse_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text), dtype=str)
    cols = {c.upper(): c for c in df.columns}
    if "TIME_PERIOD" not in cols or "OBS_VALUE" not in cols:
        raise ValueError("CSV response lacks TIME_PERIOD / OBS_VALUE columns")
    return _to_series(df[cols["TIME_PERIOD"]], df[cols["OBS_VALUE"]])


def _parse_json(payload: dict) -> pd.Series:
    data = payload.get("data", payload)
    structure = (data.get("structures") or [data.get("structure")])[0]
    obs_dims = structure["dimensions"]["observation"]
    time_dim = next((d for d in obs_dims if d.get("id") == "TIME_PERIOD"), obs_dims[0])
    time_values = [v.get("id", v.get("value")) for v in time_dim["values"]]

    datasets = data.get("dataSets") or []
    if not datasets or not datasets[0].get("series"):
        raise ValueError("JSON response contains no series")
    series = next(iter(datasets[0]["series"].values()))
    periods, values = [], []
    for k, obs in series.get("observations", {}).items():
        periods.append(time_values[int(k.split(":")[0])])
        values.append(obs[0] if isinstance(obs, list) and obs else None)
    return _to_series(periods, values)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_imf_series(dataflow: str, key: str) -> pd.Series:
    """Download one monthly series from the IMF SDMX 3.0 API (tries several URL/format variants)."""
    errors = []
    attempts = []
    for agency in ("IMF.STA", "*"):
        for version in ("+", "~"):
            url = f"{BASE_URL}/data/dataflow/{agency}/{dataflow}/{version}/{key}"
            attempts.append((url, "csv"))
            attempts.append((url, "json"))

    for url, fmt in attempts:
        if fmt == "csv":
            headers = {"Accept": "application/vnd.sdmx.data+csv;version=2.0.0"}
            params = {}
        else:
            headers = {"Accept": "application/vnd.sdmx.data+json;version=2.0.0"}
            params = {"dimensionAtObservation": "TIME_PERIOD", "attributes": "dsd", "measures": "all"}
        try:
            r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
            if r.status_code == 404:
                errors.append(f"{url} [{fmt}] -> 404 (no data)")
                continue
            r.raise_for_status()
            s = _parse_csv(r.text) if fmt == "csv" else _parse_json(r.json())
            s = s[s.index >= START_MONTH]
            if not s.empty:
                return s
            errors.append(f"{url} [{fmt}] -> empty series")
        except Exception as exc:  # network, HTTP or parsing problems
            errors.append(f"{url} [{fmt}] -> {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not retrieve {dataflow} {key}.\n" + "\n".join(errors[-4:]))


def get_cpi(iso3: str) -> pd.Series:
    return fetch_imf_series(CPI_FLOW, f"{iso3}.CPI._T.IX.M")


def get_fx_per_usd(iso3: str) -> pd.Series | None:
    """Local currency units per USD. Returns None for the US (identically 1)."""
    if iso3 == "USA":
        return None
    return fetch_imf_series(FX_FLOW, f"{iso3}.XDC_USD.PA_RT.M")


# --------------------------------------------------------------------------
# Economics
# --------------------------------------------------------------------------
def build_dataset(fx1, fx2, p1, p2) -> pd.DataFrame:
    """Merge inputs on a monthly index and compute S(t) and RER(t)."""
    pieces = {k: v for k, v in {"FX1": fx1, "FX2": fx2, "P1": p1, "P2": p2}.items() if v is not None}
    df = pd.concat(pieces, axis=1).sort_index()
    df = df[df.index >= START_MONTH]
    if fx1 is None:
        df["FX1"] = 1.0
    if fx2 is None:
        df["FX2"] = 1.0
    df = df[["FX1", "FX2", "P1", "P2"]]
    df["S"] = df["FX2"] / df["FX1"]               # Country-2 units per 1 Country-1 unit
    df["RER"] = df["S"] * df["P1"] / df["P2"]
    df["valid"] = np.isfinite(df["RER"]) & (df["RER"] > 0)
    return df


def apply_sample(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    """Restrict to [start, end], compute RER* and the real index."""
    out = df.loc[(df.index >= start) & (df.index <= end)].copy()
    rer_star = out.loc[out["valid"], "RER"].mean()
    out["RER_star"] = rer_star
    out["Real_Index"] = np.where(out["valid"], 100 * out["RER"] / rer_star, np.nan)
    return out, rer_star


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("💱 Real Exchange Rate App")
st.caption(
    "Monthly IMF CPI and exchange-rate data (from January 1976 where available). "
    "S(t) = Country 2 currency per 1 unit of Country 1 currency; "
    "RER(t) = S(t) × P1(t) / P2(t); Real Index(t) = 100 × RER(t) / RER*."
)

names = country_options()
codes = list(names.keys())

with st.sidebar:
    st.header("Countries")
    c1 = st.selectbox("Country 1", codes, index=codes.index("GBR"),
                      format_func=lambda c: f"{names[c]} ({c})")
    c2 = st.selectbox("Country 2", codes, index=codes.index("USA"),
                      format_func=lambda c: f"{names[c]} ({c})")
    if st.button("Clear cached IMF data"):
        st.cache_data.clear()

if c1 == c2:
    st.warning("Country 1 and Country 2 are the same; S(t) = 1 by construction.")

cur1, cur2 = currency_label(c1, names), currency_label(c2, names)

with st.spinner("Downloading IMF data…"):
    problems = []
    results = {}
    for label, fn, iso in [("FX1", get_fx_per_usd, c1), ("FX2", get_fx_per_usd, c2),
                           ("P1", get_cpi, c1), ("P2", get_cpi, c2)]:
        try:
            results[label] = fn(iso)
        except Exception as exc:
            problems.append(f"**{label} ({names[iso]})**: {exc}")

if problems:
    st.error("Some IMF series could not be downloaded. The IMF may not publish them for "
             "this country, or the API may be temporarily unavailable.")
    for p in problems:
        st.code(p.replace("**", ""), language=None)
    st.stop()

data = build_dataset(results["FX1"], results["FX2"], results["P1"], results["P2"])
valid_months = data.index[data["valid"]]
if len(valid_months) < 2:
    st.error("The CPI and exchange-rate series for these countries do not overlap.")
    st.stop()

with st.expander("Data coverage"):
    cov = []
    for label, desc in [("FX1", f"{cur1} per USD"), ("FX2", f"{cur2} per USD"),
                        ("P1", f"CPI {names[c1]}"), ("P2", f"CPI {names[c2]}")]:
        s = data[label].dropna()
        cov.append({"Series": desc, "First": s.index.min().strftime("%Y-%m"),
                    "Last": s.index.max().strftime("%Y-%m"), "Obs": len(s)})
    st.table(pd.DataFrame(cov))

month_labels = [d.strftime("%Y-%m") for d in valid_months]
with st.sidebar:
    st.header("Sample period")
    start_lbl, end_lbl = st.select_slider("Start and end month", options=month_labels,
                                          value=(month_labels[0], month_labels[-1]))
    st.header("Chart")
    view = st.radio("Show", ["Real", "Nominal", "Both"], index=2, horizontal=True)

start, end = pd.Timestamp(start_lbl + "-01"), pd.Timestamp(end_lbl + "-01")
sample, rer_star = apply_sample(data, start, end)
n_valid = int(sample["valid"].sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("RER* (sample mean of RER)", f"{rer_star:,.6g}")
m2.metric("Mean of Real Index", f"{sample['Real_Index'].mean():.4f}")
m3.metric("Valid months", f"{n_valid}")
m4.metric(f"Latest S ({cur2} per {cur1})", f"{sample.loc[sample['valid'], 'S'].iloc[-1]:,.6g}")

# ---- Chart ---------------------------------------------------------------
plot = sample[sample["valid"]]
real_trace = go.Scatter(x=plot.index, y=plot["Real_Index"], name="Real index (RER* = 100)",
                        line=dict(color="#1f77b4", width=2))
nom_trace = go.Scatter(x=plot.index, y=plot["S"], name=f"Nominal S ({cur2} per {cur1})",
                       line=dict(color="#d62728", width=2))

if view == "Both":
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(real_trace, secondary_y=False)
    fig.add_trace(nom_trace, secondary_y=True)
    fig.update_yaxes(title_text="Real index (sample mean = 100)", secondary_y=False)
    fig.update_yaxes(title_text=f"{cur2} per 1 {cur1}", secondary_y=True)
    fig.add_hline(y=100, line_dash="dot", line_color="gray", secondary_y=False)
elif view == "Real":
    fig = go.Figure(real_trace)
    fig.update_yaxes(title_text="Real index (sample mean = 100)")
    fig.add_hline(y=100, line_dash="dot", line_color="gray")
else:
    fig = go.Figure(nom_trace)
    fig.update_yaxes(title_text=f"{cur2} per 1 {cur1}")

fig.update_layout(title=f"{names[c1]} vs {names[c2]}: {start_lbl} to {end_lbl}",
                  hovermode="x unified", height=520,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
fig.update_xaxes(rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True)

st.caption("A rise in the real index means Country 1's goods become more expensive relative to "
           "Country 2's (real appreciation of Country 1's currency).")

# ---- Export --------------------------------------------------------------
export = pd.DataFrame({
    "month": sample.index.strftime("%Y-%m"),
    f"FX1_{cur1}_per_USD": sample["FX1"].values,
    f"FX2_{cur2}_per_USD": sample["FX2"].values,
    f"S_{cur2}_per_{cur1}": sample["S"].values,
    f"P1_CPI_{c1}": sample["P1"].values,
    f"P2_CPI_{c2}": sample["P2"].values,
    "RER": sample["RER"].values,
    "valid": sample["valid"].values,
    "RER_star": sample["RER_star"].values,
    "Real_Index": sample["Real_Index"].values,
})
st.download_button("⬇️ Download CSV (selected sample)",
                   data=export.to_csv(index=False).encode("utf-8"),
                   file_name=f"rer_{c1}_{c2}_{start_lbl}_{end_lbl}.csv", mime="text/csv")

# ---- Manual verification helper -----------------------------------------
with st.expander("Verify one RER calculation"):
    pick = st.selectbox("Month", list(plot.index.strftime("%Y-%m"))[::-1])
    row = sample.loc[pd.Timestamp(pick + "-01")]
    st.markdown(
        f"""
- FX1 = {row['FX1']:.6g} {cur1} per USD, FX2 = {row['FX2']:.6g} {cur2} per USD
- S = FX2 / FX1 = {row['FX2']:.6g} / {row['FX1']:.6g} = **{row['S']:.6g} {cur2} per {cur1}**
- P1 = {row['P1']:.6g}, P2 = {row['P2']:.6g}
- RER = S × P1 / P2 = {row['S']:.6g} × {row['P1']:.6g} / {row['P2']:.6g} = **{row['RER']:.6g}**
- Real Index = 100 × RER / RER* = 100 × {row['RER']:.6g} / {rer_star:.6g} = **{row['Real_Index']:.4f}**
"""
    )

with st.expander("Show data table"):
    st.dataframe(export, use_container_width=True, hide_index=True)

st.caption("Source: IMF SDMX 3.0 API (dataflows CPI and ER). Data may have gaps or limited coverage.")
