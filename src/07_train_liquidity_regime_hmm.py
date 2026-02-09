from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM # type: ignore


ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "data" / "models"


def _order_states_by_expansiveness(df: pd.DataFrame, hidden_states: np.ndarray) -> dict[int, int]:
    tmp = df.copy()
    tmp["state"] = hidden_states
    means = tmp.groupby("state")[["netliq_z252", "gcb_z252"]].mean()
    score = means["netliq_z252"] + means["gcb_z252"]
    order = score.sort_values(ascending=False).index.tolist()
    return {int(old_state): int(new_regime) for new_regime, old_state in enumerate(order)}


def _smooth_transmat(model: GaussianHMM, eps: float = 1e-3) -> None:
    """
    Suaviza la matriz de transición para evitar ceros casi absolutos.
    eps pequeño añade pseudocount y renormaliza filas.
    """
    T = model.transmat_.copy()
    T = T + eps
    T = T / T.sum(axis=1, keepdims=True)
    model.transmat_ = T


def _fit_best_hmm(
    Xs: np.ndarray,
    n_components: int = 3,
    n_tries: int = 10,
    min_covar: float = 1e-2,
    trans_smooth: float = 1e-3,
) -> GaussianHMM:
    best_model = None
    best_score = -np.inf

    for seed in range(42, 42 + n_tries):
        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=500,
            tol=1e-4,
            random_state=seed,
            verbose=False,
            min_covar=min_covar,
        )
        model.fit(Xs)
        _smooth_transmat(model, eps=trans_smooth)
        score = model.score(Xs)
        if score > best_score:
            best_score = score
            best_model = model

    assert best_model is not None
    print(f"[INFO] Best HMM score: {best_score:.3f} | min_covar={min_covar} | trans_smooth={trans_smooth}")
    return best_model


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    p = FEATURES_DIR / "gli_master.parquet"
    if not p.exists():
        raise SystemExit(f"[ERROR] Falta {p}. Ejecuta primero 05_build_gli_master.py")

    df = pd.read_parquet(p).sort_values("date").reset_index(drop=True)

    # SOLO nivel + cambio (sin percentiles)
    feat_cols = [
        "netliq_z252",
        "gcb_z252",
        "netliq_d1_z252",
        "gcb_d1_z252",
    ]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] Faltan columnas en gli_master: {missing}")

    X = df[feat_cols].copy()
    mask = X.notna().all(axis=1)
    df2 = df.loc[mask].copy().reset_index(drop=True)
    X2 = X.loc[mask].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X2)

    hmm = _fit_best_hmm(
        Xs,
        n_components=3,
        n_tries=10,
        min_covar=1e-2,      # <-- más suave
        trans_smooth=1e-3,   # <-- evita transiciones ~0
    )

    hidden_states = hmm.predict(Xs)
    post = hmm.predict_proba(Xs)

    mapping = _order_states_by_expansiveness(df2, hidden_states)
    df2["state_raw"] = hidden_states
    df2["regime"] = df2["state_raw"].map(mapping).astype(int)

    # Probabilidades reordenadas a regime 0..2
    p_regime = np.zeros_like(post)
    for old_state in range(post.shape[1]):
        new_regime = mapping[int(old_state)]
        p_regime[:, new_regime] = post[:, old_state]

    df2["p_regime_0"] = p_regime[:, 0]
    df2["p_regime_1"] = p_regime[:, 1]
    df2["p_regime_2"] = p_regime[:, 2]

    # Matriz de transición en términos de regime
    trans_raw = hmm.transmat_.copy()
    inv = {v: k for k, v in mapping.items()}
    order_raw = [inv[0], inv[1], inv[2]]
    trans_regime = trans_raw[np.ix_(order_raw, order_raw)]

    out_parquet = MODELS_DIR / "liquidity_regimes_hmm.parquet"
    out_csv = MODELS_DIR / "liquidity_regimes_hmm.csv"
    out_trans = MODELS_DIR / "liquidity_regimes_hmm_transition.csv"

    cols_out = [
        "date",
        "regime",
        "p_regime_0",
        "p_regime_1",
        "p_regime_2",
        "netliq_z252",
        "gcb_z252",
        "netliq_d1_z252",
        "gcb_d1_z252",
    ]
    df2[cols_out].to_parquet(out_parquet, index=False)
    df2[cols_out].to_csv(out_csv, index=False)

    trans_df = pd.DataFrame(
        trans_regime,
        index=["from_expansivo", "from_neutral", "from_contractivo"],
        columns=["to_expansivo", "to_neutral", "to_contractivo"],
    )
    trans_df.to_csv(out_trans, index=True)

    print("[OK] HMM entrenado (3 regímenes) - V3 (más suave).")
    print(f"     Parquet: {out_parquet}")
    print(f"     CSV:     {out_csv}")
    print(f"     Transition: {out_trans}")

    counts = df2["regime"].value_counts().sort_index()
    print("\nRecuento por régimen (0=expansivo,1=neutral,2=contractivo):")
    print(counts.to_string())

    print("\nMatriz de transición (regime):")
    print(trans_df.to_string())

    print("\nÚltimas filas:")
    print(df2[["date", "regime", "p_regime_0", "p_regime_1", "p_regime_2"]].tail(12).to_string(index=False))


if __name__ == "__main__":
    main()





## python 07_train_liquidity_regime_hmm.py
