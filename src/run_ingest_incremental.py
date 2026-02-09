from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]  # ajusta si tu estructura difiere
CONFIGS = ROOT / "configs"
DATA_BRONZE = ROOT / "data" / "bronze"
MANIFESTS = ROOT / "manifests"

FRED_CFG_PATH = CONFIGS / "series_fred.yaml"
SDMX_CFG_PATH = CONFIGS / "sdmx.yaml"

def _die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)

def _read_yaml(path: Path) -> dict:
    if not path.exists():
        _die(f"[ERROR] No existe config: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        _die(f"[ERROR] YAML inválido (no dict): {path}")
    return cfg

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _append_log(rows: list[dict]) -> None:
    _ensure_dir(MANIFESTS)
    log_path = MANIFESTS / "ingest_log.parquet"
    new = pd.DataFrame(rows)
    if log_path.exists():
        old = pd.read_parquet(log_path)
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_parquet(log_path, index=False)

def _max_date_in_parquet(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["date"])
    if df.empty:
        return None
    d = pd.to_datetime(df["date"], utc=True, errors="coerce").dropna()
    if d.empty:
        return None
    return d.max()

def _merge_append(old_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    if old_path.exists():
        old = pd.read_parquet(old_path)
        df = pd.concat([old, new_df], ignore_index=True)
    else:
        df = new_df.copy()

    # Normalización y dedupe
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    if "series_id" in df.columns:
        df = df.drop_duplicates(subset=["date", "series_id"], keep="last")
    else:
        df = df.drop_duplicates(subset=["date"], keep="last")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def main() -> None:
    load_dotenv(ROOT / ".env")
    _ensure_dir(DATA_BRONZE)
    _ensure_dir(MANIFESTS)

    logs: list[dict] = []

    # -------------------------
    # FRED incremental
    # -------------------------
    fred_cfg = _read_yaml(FRED_CFG_PATH)
    if "fred" not in fred_cfg or "series" not in fred_cfg["fred"]:
        _die(f"[ERROR] Estructura inválida en {FRED_CFG_PATH}")

    from ingest.fred import fetch_fred_series  # tu función actual :contentReference[oaicite:1]{index=1}

    items = list(fred_cfg["fred"]["series"].items())
    if not items:
        _die(f"[ERROR] No hay series en {FRED_CFG_PATH}")

    for sid, meta in tqdm(items, desc="FRED incremental"):
        out = DATA_BRONZE / "fred" / f"{sid}.parquet"
        _ensure_dir(out.parent)

        last = _max_date_in_parquet(out)
        # Si ya hay datos, pedimos desde el día siguiente
        start = meta.get("start")
        if last is not None:
            start = (last + pd.Timedelta(days=1)).date().isoformat()

        df_new = fetch_fred_series(sid, start=start, end=meta.get("end"))
        if df_new is None or df_new.empty:
            logs.append({"ts_utc": datetime.now(timezone.utc).isoformat(),
                         "source": "FRED", "dataset": sid, "rows_new": 0, "path": str(out)})
            continue

        df_all = _merge_append(out, df_new)
        df_all.to_parquet(out, index=False)

        logs.append({"ts_utc": datetime.now(timezone.utc).isoformat(),
                     "source": "FRED", "dataset": sid, "rows_new": int(len(df_new)),
                     "rows_total": int(len(df_all)), "path": str(out)})

    # -------------------------
    # SDMX incremental (si existe)
    # -------------------------
    if SDMX_CFG_PATH.exists():
        try:
            sdmx_cfg = _read_yaml(SDMX_CFG_PATH)
        except SystemExit:
            print(f"[WARN] SDMX config inválida: {SDMX_CFG_PATH}. Se omite SDMX.")
            sdmx_cfg = None

        if isinstance(sdmx_cfg, dict):
            series_list = sdmx_cfg.get("series", [])
            if isinstance(series_list, list) and series_list:
                try:
                    from ingest.sdmx import fetch_sdmx_series  # si lo tienes (o adaptas) :contentReference[oaicite:2]{index=2}
                except Exception as e:
                    print(f"[WARN] No se pudo importar ingest.sdmx: {type(e).__name__}: {e}")
                    fetch_sdmx_series = None

                if fetch_sdmx_series:
                    for item in tqdm(series_list, desc="SDMX incremental"):
                        for k in ("source", "flow", "key"):
                            if k not in item:
                                print(f"[WARN] SDMX item inválido: {item}")
                                continue

                        safe_name = item.get("name") or f'{item["source"]}_{item["flow"]}'
                        out = DATA_BRONZE / "sdmx" / f"{safe_name}.parquet"
                        _ensure_dir(out.parent)

                        last = _max_date_in_parquet(out)
                        start = item.get("start")
                        if last is not None:
                            start = (last + pd.Timedelta(days=1)).date().isoformat()

                        df_new = fetch_sdmx_series(
                            source=item["source"], flow=item["flow"], key=item["key"],
                            start=start, end=item.get("end")
                        )

                        if df_new is None or df_new.empty:
                            logs.append({"ts_utc": datetime.now(timezone.utc).isoformat(),
                                         "source": str(item["source"]), "dataset": safe_name,
                                         "rows_new": 0, "path": str(out)})
                            continue

                        df_all = _merge_append(out, df_new)
                        df_all.to_parquet(out, index=False)

                        logs.append({"ts_utc": datetime.now(timezone.utc).isoformat(),
                                     "source": str(item["source"]), "dataset": safe_name,
                                     "rows_new": int(len(df_new)), "rows_total": int(len(df_all)),
                                     "path": str(out)})
            else:
                print("[INFO] SDMX config sin series → omitido.")

    _append_log(logs)
    print("\n[OK] Ingestión incremental terminada.")
    print(f"     Bronze: {DATA_BRONZE}")
    print(f"     Log:    {MANIFESTS / 'ingest_log.parquet'}")

if __name__ == "__main__":
    main()



## 