from __future__ import annotations

import pandas as pd
import requests

# Endpoints SDMX REST (SDMX-JSON) más comunes
BASE_URLS = {
    "ECB": "https://data-api.ecb.europa.eu/service/data",   # ECB Data Portal API
    "BIS": "https://stats.bis.org/api/v1/data",             # BIS SDMX (v1)
    "IMF": "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData",  # IMF SDMX-JSON
}

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def fetch_sdmx_series_http(
    source: str,
    flow: str,
    key: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Descarga una serie SDMX y la normaliza a columnas:
      date | value | source | flow | key

    source: "ECB" | "BIS" | "IMF"
    flow: dataset/dataflow id
    key:  clave SDMX (dim1.dim2....)
    start/end: YYYY or YYYY-MM or YYYY-MM-DD (según proveedor)
    """
    source = source.upper()
    if source not in BASE_URLS:
        raise ValueError(f"source inválido: {source}. Usa: {list(BASE_URLS)}")

    # Construcción de URL según proveedor
    if source in ("ECB", "BIS"):
        # ECB/BIS: .../flow/key
        url = f"{BASE_URLS[source]}/{flow}/{key}"
        params = {}
        if start:
            params["startPeriod"] = start
        if end:
            params["endPeriod"] = end

        # Pedimos SDMX-JSON
        headers = {"Accept": "application/vnd.sdmx.data+json;version=1.0.0-wd"}
        r = requests.get(url, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        js = r.json()

        # Parseo SDMX-JSON (estructura típica: dataSets[0].series -> observations)
        datasets = js.get("dataSets", [])
        if not datasets:
            return pd.DataFrame(columns=["date", "value", "source", "flow", "key"])

        # Time dimension
        # En SDMX-JSON suele venir en structure.dimensions.observation
        obs_dims = js.get("structure", {}).get("dimensions", {}).get("observation", [])
        # Busca dimensión TIME_PERIOD
        time_pos = None
        time_values = None
        for i, d in enumerate(obs_dims):
            if d.get("id") in ("TIME_PERIOD", "TIME", "PERIOD"):
                time_pos = i
                time_values = [v.get("id") for v in d.get("values", [])]
                break
        if time_values is None:
            return pd.DataFrame(columns=["date", "value", "source", "flow", "key"])

        series_dict = datasets[0].get("series", {})
        # muchas veces hay una sola serie -> clave "0:0:0..."
        if not series_dict:
            return pd.DataFrame(columns=["date", "value", "source", "flow", "key"])

        # toma todas las series y concatena (normalmente 1)
        rows = []
        for _, s in series_dict.items():
            obs = s.get("observations", {})
            for t_idx, val_arr in obs.items():
                # val_arr suele ser [value] (o [value, flags...])
                v = _to_float(val_arr[0] if isinstance(val_arr, list) and val_arr else val_arr)
                if v is None:
                    continue
                t = time_values[int(t_idx)]
                rows.append((t, v))

        df = pd.DataFrame(rows, columns=["date", "value"])
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values("date")
        df["source"] = source
        df["flow"] = flow
        df["key"] = key
        return df.reset_index(drop=True)

    else:
        # IMF CompactData: .../flow/key
        url = f"{BASE_URLS[source]}/{flow}/{key}"
        params = {}
        if start:
            params["startPeriod"] = start
        if end:
            params["endPeriod"] = end

        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()

        # IMF CompactData suele devolver Series -> Obs con @TIME_PERIOD y @OBS_VALUE
        series = (
            js.get("CompactData", {})
              .get("DataSet", {})
              .get("Series", None)
        )
        if series is None:
            return pd.DataFrame(columns=["date", "value", "source", "flow", "key"])

        # Series puede ser dict o list
        if isinstance(series, dict):
            series_list = [series]
        else:
            series_list = series

        rows = []
        for s in series_list:
            obs = s.get("Obs", [])
            if isinstance(obs, dict):
                obs = [obs]
            for o in obs:
                t = o.get("@TIME_PERIOD") or o.get("TIME_PERIOD")
                v = o.get("@OBS_VALUE") or o.get("OBS_VALUE")
                v = _to_float(v)
                if t is None or v is None:
                    continue
                rows.append((t, v))

        df = pd.DataFrame(rows, columns=["date", "value"])
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values("date")
        df["source"] = source
        df["flow"] = flow
        df["key"] = key
        return df.reset_index(drop=True)
