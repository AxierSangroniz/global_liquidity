from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data" / "bronze" / "fred"
SILVER = ROOT / "data" / "silver"


def _read(path: Path, colname: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[ERROR] No existe: {path}")
    df = pd.read_parquet(path)
    if "date" not in df.columns or "value" not in df.columns:
        raise SystemExit(f"[ERROR] Formato inesperado en {path}. Columnas: {list(df.columns)}")
    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    return df.rename(columns={"value": colname})


def main() -> None:
    SILVER.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Assets (native units)
    # -------------------------
    fed = _read(BRONZE / "WALCL.parquet", "fed_usd_millions")  # already USD millions

    ecb = _read(BRONZE / "ECBASSETSW.parquet", "ecb_eur_millions")  # typically EUR millions (from ECB via FRED)
    boj = _read(BRONZE / "JPNASSETS.parquet", "boj_100m_yen")       # 100 million yen units :contentReference[oaicite:4]{index=4}

    # -------------------------
    # FX
    # -------------------------
    # DEXJPUS: JPY per 1 USD :contentReference[oaicite:5]{index=5}
    jpy_per_usd = _read(BRONZE / "DEXJPUS.parquet", "jpy_per_usd")

    # DEXUSEU: USD per 1 EUR (FRED series)
    usd_per_eur = _read(BRONZE / "DEXUSEU.parquet", "usd_per_eur")

    # -------------------------
    # Merge all (outer) and align
    # -------------------------
    df = fed.merge(ecb, on="date", how="outer")
    df = df.merge(boj, on="date", how="outer")
    df = df.merge(jpy_per_usd, on="date", how="outer")
    df = df.merge(usd_per_eur, on="date", how="outer")
    df = df.sort_values("date")

    # forward-fill (porque assets son weekly/monthly y FX daily)
    cols_ffill = ["fed_usd_millions", "ecb_eur_millions", "boj_100m_yen", "jpy_per_usd", "usd_per_eur"]
    df[cols_ffill] = df[cols_ffill].ffill()

    # drop leading NaNs
    df = df.dropna(subset=["fed_usd_millions", "ecb_eur_millions", "boj_100m_yen", "jpy_per_usd", "usd_per_eur"])

    # -------------------------
    # Convert to USD (millions)
    # -------------------------
    # ECB: EUR millions -> USD millions
    df["ecb_usd_millions"] = df["ecb_eur_millions"] * df["usd_per_eur"]

    # BOJ: units = 100 million yen
    # 100 million yen = 100,000,000 JPY
    # Convert to USD: JPY / (JPY per USD) = USD
    # Then convert to "millions USD": divide by 1,000,000
    # => (boj_100m_yen * 100,000,000) / jpy_per_usd / 1,000,000 = boj_100m_yen * 100 / jpy_per_usd
    df["boj_usd_millions"] = df["boj_100m_yen"] * 100.0 / df["jpy_per_usd"]

    # FED already USD millions
    df["global_cb_assets_usd_millions"] = df["fed_usd_millions"] + df["ecb_usd_millions"] + df["boj_usd_millions"]

    # Deltas útiles
    df["global_cb_assets_usd_d1"] = df["global_cb_assets_usd_millions"].diff(1)
    df["global_cb_assets_usd_w1"] = df["global_cb_assets_usd_millions"].diff(7)

    # Output
    out_parquet = SILVER / "global_cb_assets_usd.parquet"
    out_csv = SILVER / "global_cb_assets_usd.csv"

    keep = [
        "date",
        "fed_usd_millions",
        "ecb_usd_millions",
        "boj_usd_millions",
        "global_cb_assets_usd_millions",
        "global_cb_assets_usd_d1",
        "global_cb_assets_usd_w1",
        "usd_per_eur",
        "jpy_per_usd",
    ]
    df_out = df[keep].copy()

    df_out.to_parquet(out_parquet, index=False)
    df_out.to_csv(out_csv, index=False)

    print("[OK] Global CB Assets (USD) construido.")
    print(f"     Parquet: {out_parquet}")
    print(f"     CSV:     {out_csv}")
    print("\nÚltimas filas:")
    print(df_out.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()




## python 04_build_global_cb_assets_usd.py
