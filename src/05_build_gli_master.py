from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver"
FEATURES = ROOT / "data" / "features"


def zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / sd


def pct_rank(s: pd.Series, window: int) -> pd.Series:
    # percentil rolling: valor actual dentro de la ventana
    return s.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def main() -> None:
    FEATURES.mkdir(parents=True, exist_ok=True)

    p_net = SILVER / "net_liquidity_usa_fixed.parquet"
    p_gcb = SILVER / "global_cb_assets_usd.parquet"

    if not p_net.exists():
        raise SystemExit(f"[ERROR] Falta: {p_net}")
    if not p_gcb.exists():
        raise SystemExit(f"[ERROR] Falta: {p_gcb}")

    net = pd.read_parquet(p_net)
    gcb = pd.read_parquet(p_gcb)

    # Normaliza fechas
    net["date"] = pd.to_datetime(net["date"], utc=True, errors="coerce")
    gcb["date"] = pd.to_datetime(gcb["date"], utc=True, errors="coerce")

    # Selección de columnas clave
    net = net[[
        "date",
        "net_liquidity_usa_millions",
        "net_liquidity_usa_millions_d1",
        "net_liquidity_usa_millions_w1",
    ]].copy()

    gcb = gcb[[
        "date",
        "global_cb_assets_usd_millions",
        "global_cb_assets_usd_d1",
        "global_cb_assets_usd_w1",
        "usd_per_eur",
        "jpy_per_usd",
    ]].copy()

    # Merge
    df = net.merge(gcb, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Features adicionales
    df["netliq_d1_pct"] = df["net_liquidity_usa_millions"].pct_change(1)
    df["netliq_w1_pct"] = df["net_liquidity_usa_millions"].pct_change(7)
    df["gcb_d1_pct"] = df["global_cb_assets_usd_millions"].pct_change(1)
    df["gcb_w1_pct"] = df["global_cb_assets_usd_millions"].pct_change(7)

    # Z-scores (ventanas típicas)
    for w in (90, 252):
        df[f"netliq_z{w}"] = zscore(df["net_liquidity_usa_millions"], w)
        df[f"gcb_z{w}"] = zscore(df["global_cb_assets_usd_millions"], w)

        df[f"netliq_d1_z{w}"] = zscore(df["net_liquidity_usa_millions_d1"], w)
        df[f"gcb_d1_z{w}"] = zscore(df["global_cb_assets_usd_d1"], w)

    # Percentiles rolling (más robusto para regímenes)
    for w in (90, 252):
        df[f"netliq_pct{w}"] = pct_rank(df["net_liquidity_usa_millions"], w)
        df[f"gcb_pct{w}"] = pct_rank(df["global_cb_assets_usd_millions"], w)

    out_parquet = FEATURES / "gli_master.parquet"
    out_csv = FEATURES / "gli_master.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print("[OK] GLI master construido.")
    print(f"     Parquet: {out_parquet}")
    print(f"     CSV:     {out_csv}")
    print("\nColumnas:", ", ".join(df.columns))
    print("\nÚltimas filas:")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()



## python 05_build_gli_master.py
