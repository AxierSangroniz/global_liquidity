from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data" / "bronze" / "fred"
SILVER = ROOT / "data" / "silver"


def _read_series(path: Path, series_id: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[ERROR] No existe: {path}")

    df = pd.read_parquet(path)

    # Esperamos columnas: date, value, series_id
    if "date" not in df.columns or "value" not in df.columns:
        raise SystemExit(f"[ERROR] Formato inesperado en {path}. Columnas: {list(df.columns)}")

    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")

    df = df.rename(columns={"value": series_id})
    return df


def main() -> None:
    SILVER.mkdir(parents=True, exist_ok=True)

    # Inputs
    walcl = _read_series(BRONZE / "WALCL.parquet", "WALCL")         # Fed total assets
    rrp = _read_series(BRONZE / "RRPONTSYD.parquet", "RRPONTSYD")   # Reverse Repo ON
    tga = _read_series(BRONZE / "WTREGEN.parquet", "WTREGEN")       # TGA weekly avg

    # Unimos por fecha (outer) y hacemos forward-fill para alinear frecuencias
    df = walcl.merge(rrp, on="date", how="outer").merge(tga, on="date", how="outer")
    df = df.sort_values("date")

    # Forward fill (porque WALCL/WTREGEN son semanales y RRP es diario)
    df[["WALCL", "RRPONTSYD", "WTREGEN"]] = df[["WALCL", "RRPONTSYD", "WTREGEN"]].ffill()

    # Quita filas iniciales donde aún no hay datos suficientes
    df = df.dropna(subset=["WALCL", "RRPONTSYD", "WTREGEN"])

    # Calcula Net Liquidity (USA)
    df["net_liquidity_usa"] = df["WALCL"] - df["RRPONTSYD"] - df["WTREGEN"]

    # Deltas (útiles para régimen)
    df["net_liquidity_usa_d1"] = df["net_liquidity_usa"].diff(1)
    df["net_liquidity_usa_w1"] = df["net_liquidity_usa"].diff(7)

    out_parquet = SILVER / "net_liquidity_usa.parquet"
    out_csv = SILVER / "net_liquidity_usa.csv"

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print("[OK] Net Liquidity (USA) construido.")
    print(f"     Parquet: {out_parquet}")
    print(f"     CSV:     {out_csv}")
    print("\nÚltimas filas:")
    print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()


## python 02_build_net_liquidity.py
