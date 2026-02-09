from __future__ import annotations

import os
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")



ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data" / "bronze" / "fred"
SILVER = ROOT / "data" / "silver"


FRED_SERIES_ENDPOINT = "https://api.stlouisfed.org/fred/series"


def _read_series(path: Path, series_id: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[ERROR] No existe: {path}")

    df = pd.read_parquet(path)

    if "date" not in df.columns or "value" not in df.columns:
        raise SystemExit(f"[ERROR] Formato inesperado en {path}. Columnas: {list(df.columns)}")

    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    df = df.rename(columns={"value": series_id})
    return df


def _fred_series_units(series_id: str) -> str | None:
    """
    Devuelve el campo 'units' de la metadata de FRED, por ejemplo:
      "Billions of Dollars", "Millions of Dollars", etc.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] Falta FRED_API_KEY en tu .env para consultar metadata de unidades.")

    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    r = requests.get(FRED_SERIES_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    series = js.get("seriess", [])
    if not series:
        return None
    return series[0].get("units")


def _to_millions_multiplier(units: str | None) -> float:
    """
    Convierte a MILLIONS como unidad común:
    - Billions -> * 1000
    - Millions -> * 1
    - Thousands -> / 1000 (si apareciera)
    Si no reconoce, devuelve 1 y avisa.
    """
    if not units:
        return 1.0

    u = units.lower()
    if "billions" in u:
        return 1000.0
    if "millions" in u:
        return 1.0
    if "thousands" in u:
        return 0.001

    # fallback
    print(f"[WARN] Unidades no reconocidas ('{units}'). No convierto (mult=1).")
    return 1.0


def main() -> None:
    SILVER.mkdir(parents=True, exist_ok=True)

    # 1) Leer series
    walcl = _read_series(BRONZE / "WALCL.parquet", "WALCL")         # Fed total assets
    rrp = _read_series(BRONZE / "RRPONTSYD.parquet", "RRPONTSYD")   # Reverse Repo ON
    tga = _read_series(BRONZE / "WTREGEN.parquet", "WTREGEN")       # TGA weekly avg

    # 2) Detectar unidades en FRED y convertir todo a MILLIONS
    units_walcl = _fred_series_units("WALCL")
    units_rrp = _fred_series_units("RRPONTSYD")
    units_tga = _fred_series_units("WTREGEN")

    mult_walcl = _to_millions_multiplier(units_walcl)
    mult_rrp = _to_millions_multiplier(units_rrp)
    mult_tga = _to_millions_multiplier(units_tga)

    print("[INFO] Unidades FRED:")
    print(f"  WALCL     units='{units_walcl}'   mult_to_millions={mult_walcl}")
    print(f"  RRPONTSYD units='{units_rrp}'    mult_to_millions={mult_rrp}")
    print(f"  WTREGEN   units='{units_tga}'    mult_to_millions={mult_tga}")

    walcl["WALCL"] = walcl["WALCL"] * mult_walcl
    rrp["RRPONTSYD"] = rrp["RRPONTSYD"] * mult_rrp
    tga["WTREGEN"] = tga["WTREGEN"] * mult_tga

    # 3) Merge + ffill (alineación)
    df = walcl.merge(rrp, on="date", how="outer").merge(tga, on="date", how="outer").sort_values("date")
    df[["WALCL", "RRPONTSYD", "WTREGEN"]] = df[["WALCL", "RRPONTSYD", "WTREGEN"]].ffill()
    df = df.dropna(subset=["WALCL", "RRPONTSYD", "WTREGEN"])

    # 4) Net Liquidity (en MILLIONS)
    df["net_liquidity_usa_millions"] = df["WALCL"] - df["RRPONTSYD"] - df["WTREGEN"]
    df["net_liquidity_usa_millions_d1"] = df["net_liquidity_usa_millions"].diff(1)
    df["net_liquidity_usa_millions_w1"] = df["net_liquidity_usa_millions"].diff(7)

    out_parquet = SILVER / "net_liquidity_usa_fixed.parquet"
    out_csv = SILVER / "net_liquidity_usa_fixed.csv"

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print("\n[OK] Net Liquidity (USA) FIXED (unidades consistentes) construido.")
    print(f"     Parquet: {out_parquet}")
    print(f"     CSV:     {out_csv}")
    print("\nÚltimas filas:")
    cols = [
        "date", "WALCL", "RRPONTSYD", "WTREGEN",
        "net_liquidity_usa_millions", "net_liquidity_usa_millions_d1", "net_liquidity_usa_millions_w1"
    ]
    print(df[cols].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()



## python 03_build_net_liquidity_fixed_units.py
