"""
data_sources.py -- loaders for prices, chokepoint transits and conflict events.

Every loader reads a local snapshot in data/raw first and only goes to the
network when asked (refresh=True), so results are reproducible offline.

Status of each network path (see README "What has been run"):
    oil prices, GitHub mirror of EIA   run on 2026-11-10
    FRED CSV endpoint                  not run here (host not reachable)
    IMF PortWatch ArcGIS endpoint      not run here; URL and query pattern
                                       match a public replication package
                                       that used them on 2026-10-28
    ACLED API                          not run here; needs your credentials
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REF = ROOT / "data" / "reference"

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
GITHUB_OIL = ("https://raw.githubusercontent.com/datasets/oil-prices/main/"
              "data/{name}-daily.csv")
PORTWATCH_QUERY = ("https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/"
                   "services/Daily_Chokepoints_Data/FeatureServer/0/query")
# PortWatch chokepoint ids
CHOKEPOINTS = {"suez": "chokepoint1", "bab_el_mandeb": "chokepoint4",
               "malacca": "chokepoint5", "hormuz": "chokepoint6",
               "cape": "chokepoint7"}

# FRED series worth pulling for the full study
FRED_SERIES = {
    "brent": "DCOILBRENTEU",      # Brent spot, USD/bbl, daily (EIA)
    "wti": "DCOILWTICO",          # WTI spot, USD/bbl, daily (EIA)
    "henry_hub": "DHHNGSP",       # Henry Hub gas, USD/MMBtu, daily
    "ovx": "OVXCLS",              # CBOE crude oil volatility index
    "vix": "VIXCLS",
    "usd_broad": "DTWEXBGS",      # broad dollar index
    "ttf_monthly": "PNGASEUUSDM", # EU gas, monthly (daily TTF needs ICE/LSEG)
    "lng_asia_monthly": "PNGASJPUSDM",
}


# -------------------------------------------------------------------- prices
def load_fred(sid: str, timeout=60) -> pd.Series:
    """One FRED series through the public CSV endpoint (no API key)."""
    r = requests.get(FRED_CSV.format(sid=sid), timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", sid]
    s = pd.to_numeric(df[sid], errors="coerce")
    s.index = pd.to_datetime(df["date"])
    return s.dropna().rename(sid)


def _read_price_csv(path_or_buf) -> pd.Series:
    df = pd.read_csv(path_or_buf)
    df.columns = [c.strip().lower() for c in df.columns]
    s = pd.to_numeric(df["price"], errors="coerce")
    s.index = pd.to_datetime(df["date"])
    return s.dropna()


def load_oil_prices(refresh: bool = False, source: str = "github") -> pd.DataFrame:
    """Daily Brent and WTI spot prices (USD/bbl), columns ['brent', 'wti'].

    refresh=False reads data/raw/{brent,wti}-daily.csv.
    refresh=True downloads from `source` ('github' mirror of EIA, or 'fred')
    and overwrites the local snapshot.
    """
    out = {}
    for name in ("brent", "wti"):
        path = RAW / f"{name}-daily.csv"
        if refresh or not path.exists():
            if source == "fred":
                s = load_fred(FRED_SERIES[name])
            else:
                r = requests.get(GITHUB_OIL.format(name=name), timeout=60)
                r.raise_for_status()
                s = _read_price_csv(io.StringIO(r.text))
            RAW.mkdir(parents=True, exist_ok=True)
            s.rename("Price").rename_axis("Date").to_csv(path)
        out[name] = _read_price_csv(path)
    df = pd.DataFrame(out).sort_index()
    df.index.name = "date"
    return df


# ----------------------------------------------------------------- PortWatch
def fetch_portwatch(portid: str = "chokepoint6", start: str = "2019-01-01",
                    page: int = 1000, timeout=120) -> pd.DataFrame:
    """Daily transit calls for one chokepoint from the IMF PortWatch ArcGIS
    layer 'Daily_Chokepoints_Data'. Pages through the result set."""
    where = f"portid='{portid}' AND date >= DATE '{start}'"
    rows, offset = [], 0
    while True:
        params = {"f": "json", "where": where, "outFields": "*",
                  "orderByFields": "ObjectId", "returnGeometry": "false",
                  "resultOffset": offset, "resultRecordCount": page}
        r = requests.get(PORTWATCH_QUERY, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        feats = data.get("features", [])
        rows += [f["attributes"] for f in feats]
        if len(feats) < page:
            break
        offset += page
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], unit="ms").dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def load_portwatch(chokepoint: str = "hormuz", refresh: bool = False) -> pd.DataFrame:
    """Daily PortWatch transits for one chokepoint, indexed by date.

    Columns include n_tanker, n_container, n_dry_bulk, n_total and the
    matching capacity_* fields (deadweight tonnes).

    refresh=False reads the snapshot shipped in data/raw (IMF API responses of
    2026-10-28, 2023-01-01 to 2026-10-25, taken from a public replication
    package). refresh=True pulls the full history from the IMF.
    """
    portid = CHOKEPOINTS.get(chokepoint, chokepoint)
    fresh = RAW / f"portwatch_{portid}.csv"
    if refresh:
        df = fetch_portwatch(portid)
        RAW.mkdir(parents=True, exist_ok=True)
        df.to_csv(fresh, index=False)
    elif fresh.exists():
        df = pd.read_csv(fresh, parse_dates=["date"])
    else:
        df = pd.read_csv(RAW / "portwatch_daily_2023_2026_mirror.csv",
                         parse_dates=["date"])
        df = df[df["portid"] == portid]
    return df.set_index("date").sort_index()


# --------------------------------------------------------------------- ACLED
def load_acled_csv(path, *, event_type="Explosions/Remote violence",
                   sub_event_types=("Shelling/artillery/missile attack",
                                    "Air/drone strike"),
                   countries=None, max_geo_precision=2,
                   keyword_regex=None) -> pd.DataFrame:
    """Filter an ACLED export (CSV from the Data Export Tool) to strike events.

    Keeps rows with the requested event and sub-event types, optionally a
    country list, a geo-precision ceiling (1 = exact place, 3 = region only)
    and a regular expression on the notes field, for example
    r"missile|ballistic|drone|UAV|rocket".
    """
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df["event_date"] = pd.to_datetime(df["event_date"], format="mixed", dayfirst=True)
    keep = (df["event_type"] == event_type) & df["sub_event_type"].isin(list(sub_event_types))
    if countries is not None:
        keep &= df["country"].isin(list(countries))
    if max_geo_precision is not None and "geo_precision" in df:
        keep &= pd.to_numeric(df["geo_precision"], errors="coerce") <= max_geo_precision
    if keyword_regex:
        keep &= df["notes"].fillna("").str.contains(keyword_regex, case=False, regex=True)
    out = df.loc[keep].copy()
    for c in ("latitude", "longitude", "fatalities"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.rename(columns={"event_date": "date", "latitude": "lat",
                               "longitude": "lon"}).reset_index(drop=True)


def fetch_acled(countries, start, end, *, username=None, password=None,
                page_size=5000, timeout=120) -> pd.DataFrame:
    """ACLED API client. NOT RUN in the build environment.

    Written for the OAuth-based API that ACLED introduced in 2025, as the
    author recalls it. ACLED has changed its access system more than once, so
    read the current API documentation before relying on this function; the
    CSV route (load_acled_csv) does not depend on it.

    Credentials default to the ACLED_USERNAME / ACLED_PASSWORD environment
    variables. Never hard-code them.
    """
    username = username or os.environ["ACLED_USERNAME"]
    password = password or os.environ["ACLED_PASSWORD"]
    tok = requests.post("https://acleddata.com/oauth/token",
                        data={"username": username, "password": password,
                              "grant_type": "password", "client_id": "acled"},
                        timeout=timeout)
    tok.raise_for_status()
    headers = {"Authorization": f"Bearer {tok.json()['access_token']}"}
    frames, page = [], 1
    while True:
        params = {"_format": "json", "country": "|".join(countries),
                  "event_date": f"{start}|{end}", "event_date_where": "BETWEEN",
                  "limit": page_size, "page": page}
        r = requests.get("https://acleddata.com/api/acled/read",
                         params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        frames.append(pd.DataFrame(data))
        if len(data) < page_size:
            break
        page += 1
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ----------------------------------------------------------- reference files
def load_reference(name: str) -> pd.DataFrame:
    """Read one of the small reference tables in data/reference."""
    parse = {"strike_events_reference": ["date"],
             "episode_onsets": ["calendar_date", "day0"]}.get(name)
    return pd.read_csv(REF / f"{name}.csv", parse_dates=parse)
