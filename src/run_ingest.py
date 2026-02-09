from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # .../global_liquidity
CONFIGS = ROOT / "configs"
DATA_BRONZE = ROOT / "data" / "bronze"
MANIFESTS = ROOT / "manifests"

FRED_CFG_PATH = CONFIGS / "series_fred.yaml"
SDMX_CFG_PATH = CONFIGS / "sdmx.yaml"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        _die(f"[ERROR] No existe el archivo de config: {path}")

    text = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)

    if cfg is None:
        _die(
            f"[ERROR] YAML vacío o inválido: {path}\n"
            f"Contenido (primeros 400 chars):\n{text[:400]}"
        )
    if not isinstance(cfg, dict):
        _die(f"[ERROR] YAML no es un dict (estructura inesperada): {path}")

    return cfg


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_parquet(df: pd.DataFrame, outpath: Path) -> None:
    _ensure_dir(outpath.parent)
    df.to_parquet(outpath, index=False)


def _append_ingest_log(rows: list[dict]) -> None:
    _ensure_dir(MANIFESTS)
    log_path = MANIFESTS / "ingest_log.parquet"
    new = pd.DataFrame(rows)

    if log_path.exists():
        old = pd.read_parquet(log_path)
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new

    out.to_parquet(log_path, index=False)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    load_dotenv(ROOT / ".env")

    _ensure_dir(DATA_BRONZE)
    _ensure_dir(MANIFESTS)

    logs: list[dict] = []

    # -------------------------
    # FRED (obligatorio)
    # -------------------------
    fred_cfg = _read_yaml(FRED_CFG_PATH)

    # Validación fuerte de estructura
    if "fred" not in fred_cfg or not isinstance(fred_cfg["fred"], dict):
        _die(f"[ERROR] Estructura inválida en {FRED_CFG_PATH}: falta clave 'fred'.")

    if "series" not in fred_cfg["fred"] or not isinstance(fred_cfg["fred"]["series"], dict):
        _die(
            f"[ERROR] Estructura inválida en {FRED_CFG_PATH}: falta 'fred: series:' como dict.\n"
            f"Ejemplo mínimo:\n"
            f"fred:\n  series:\n    WALCL: {{name: '...', freq: 'weekly'}}"
        )

    # Import aquí para que el error sea localizado
    from ingest.fred import fetch_fred_series

    series_items = list(fred_cfg["fred"]["series"].items())
    if not series_items:
        _die(f"[ERROR] No hay series definidas en {FRED_CFG_PATH} (fred.series está vacío).")

    for sid, meta in tqdm(series_items, desc="FRED"):
        df = fetch_fred_series(sid, start=meta.get("start"), end=meta.get("end"))
        out = DATA_BRONZE / "fred" / f"{sid}.parquet"
        _save_parquet(df, out)

        logs.append(
            {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "source": "FRED",
                "dataset": sid,
                "rows": int(len(df)),
                "path": str(out),
            }
        )

    # -------------------------
    # SDMX (opcional)
    # -------------------------
    if SDMX_CFG_PATH.exists():
        try:
            sdmx_cfg = _read_yaml(SDMX_CFG_PATH)
        except SystemExit:
            # Si existe pero está mal, lo reportamos y seguimos (no bloqueamos FRED)
            print(f"[WARN] SDMX config inválida: {SDMX_CFG_PATH}. Se omite SDMX.")
            sdmx_cfg = None

        if isinstance(sdmx_cfg, dict):
            series_list = sdmx_cfg.get("series", [])
            if not isinstance(series_list, list):
                print(f"[WARN] SDMX config: 'series' no es una lista. Se omite SDMX.")
                series_list = []

            if series_list:
                # Import lazy: si pandasdmx/SDMX rompe, que no bloquee FRED
                try:
                    from ingest.sdmx import fetch_sdmx_series
                except Exception as e:
                    print(
                        "[WARN] No se pudo importar ingest.sdmx (se omite SDMX).\n"
                        f"       Motivo: {type(e).__name__}: {e}"
                    )
                    fetch_sdmx_series = None

                if fetch_sdmx_series:
                    for item in tqdm(series_list, desc="SDMX"):
                        # Validación mínima de item
                        for k in ("source", "flow", "key"):
                            if k not in item:
                                print(f"[WARN] SDMX item inválido, falta '{k}': {item}")
                                continue

                        df = fetch_sdmx_series(
                            source=item["source"],
                            flow=item["flow"],
                            key=item["key"],
                            start=item.get("start"),
                            end=item.get("end"),
                        )

                        safe_name = item.get("name") or f'{item["source"]}_{item["flow"]}'
                        out = DATA_BRONZE / "sdmx" / f"{safe_name}.parquet"
                        _save_parquet(df, out)

                        logs.append(
                            {
                                "ts_utc": datetime.now(timezone.utc).isoformat(),
                                "source": str(item["source"]),
                                "dataset": str(safe_name),
                                "rows": int(len(df)),
                                "path": str(out),
                            }
                        )
            else:
                print(f"[INFO] SDMX config existe pero no tiene series. Se omite SDMX.")
    else:
        print("[INFO] No existe configs/sdmx.yaml → SDMX omitido (solo FRED).")

    # -------------------------
    # Log final
    # -------------------------
    _append_ingest_log(logs)

    print("\n[OK] Ingestión terminada.")
    print(f"     Bronze:  {DATA_BRONZE}")
    print(f"     Log:     {MANIFESTS / 'ingest_log.parquet'}")


if __name__ == "__main__":
    main()



## python run_ingest.py
