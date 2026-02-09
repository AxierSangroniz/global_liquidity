from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features"
OUT = ROOT / "data" / "models"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    p = FEATURES / "gli_master.parquet"
    if not p.exists():
        raise SystemExit(f"[ERROR] Falta: {p}. Ejecuta primero 05_build_gli_master.py")

    df = pd.read_parquet(p).sort_values("date").reset_index(drop=True)

    # Features para régimen: usa percentiles + deltas (robusto a escalas)
    feat_cols = [
        "netliq_d1_z252",
        "gcb_d1_z252",
        "netliq_pct252",
        "gcb_pct252",
        "netliq_z252",
        "gcb_z252",
    ]
    for c in feat_cols:
        if c not in df.columns:
            raise SystemExit(f"[ERROR] Falta feature '{c}' en gli_master. Revisa 05_build_gli_master.py")

    X = df[feat_cols].copy()

    # Limpia NaNs iniciales por rolling
    mask = X.notna().all(axis=1)
    df2 = df.loc[mask].copy()
    X2 = X.loc[mask].to_numpy()

    # Escalado
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X2)

    # Modelo 3 regímenes
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    regimes = gmm.fit_predict(Xs)
    probs = gmm.predict_proba(Xs)

    df2["regime_raw"] = regimes
    df2["regime_p0"] = probs[:, 0]
    df2["regime_p1"] = probs[:, 1]
    df2["regime_p2"] = probs[:, 2]

    # Ordena regímenes por “expansivo → contractivo”
    # Heurística: mayor netliq_z252 y gcb_z252 = más expansivo
    stats = df2.groupby("regime_raw")[["netliq_z252", "gcb_z252"]].mean()
    score = stats["netliq_z252"] + stats["gcb_z252"]
    order = score.sort_values(ascending=False).index.tolist()  # [más expansivo ... más contractivo]
    mapping = {old: new for new, old in enumerate(order)}  # 0=expansivo,1=neutral,2=contractivo
    df2["regime"] = df2["regime_raw"].map(mapping)

    # Guardados
    out_parquet = OUT / "liquidity_regimes.parquet"
    out_csv = OUT / "liquidity_regimes.csv"
    df2[["date", "regime", "regime_p0", "regime_p1", "regime_p2"]].to_parquet(out_parquet, index=False)
    df2[["date", "regime", "regime_p0", "regime_p1", "regime_p2"]].to_csv(out_csv, index=False)

    print("[OK] Modelo de régimen entrenado (GMM 3 estados).")
    print(f"     Parquet: {out_parquet}")
    print(f"     CSV:     {out_csv}")

    # Resumen
    counts = df2["regime"].value_counts().sort_index()
    print("\nRecuento por régimen (0=expansivo,1=neutral,2=contractivo):")
    print(counts.to_string())

    print("\nÚltimas filas:")
    print(df2[["date", "regime", "netliq_z252", "gcb_z252"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()



## python 06_train_liquidity_regime_gmm.py
