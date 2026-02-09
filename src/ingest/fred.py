from __future__ import annotations
import os
import requests
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def _get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_fred_series(series_id: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("Falta FRED_API_KEY en tu .env (FRED requiere API key).")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start: params["observation_start"] = start
    if end: params["observation_end"] = end

    js = _get(url, params=params)
    obs = js.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return df

    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("date")
    df["series_id"] = series_id
    return df
