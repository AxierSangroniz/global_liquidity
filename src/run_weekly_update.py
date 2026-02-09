from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ROOT = carpeta global_liquidity
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

PIPELINE = [
    # 1) Ingest incremental
    [sys.executable, str(SRC / "run_ingest_incremental.py")],

    # 2) Builds
    [sys.executable, str(SRC / "02_build_net_liquidity.py")],
    [sys.executable, str(SRC / "03_build_net_liquidity_fixed_units.py")],
    [sys.executable, str(SRC / "04_build_global_cb_assets_usd.py")],
    [sys.executable, str(SRC / "05_build_gli_master.py")],

    # 3) Modelo HMM
    [sys.executable, str(SRC / "07_train_liquidity_regime_hmm.py")],
]

def run_step(cmd: list[str]) -> None:
    print("\n" + "=" * 90)
    print("[RUN]", " ".join(cmd))
    start = datetime.now()

    p = subprocess.run(cmd)

    elapsed = (datetime.now() - start).total_seconds()
    if p.returncode != 0:
        print(f"[FAIL] code={p.returncode} elapsed={elapsed:.1f}s")
        raise SystemExit(p.returncode)

    print(f"[OK] elapsed={elapsed:.1f}s")

def main() -> None:
    for cmd in PIPELINE:
        run_step(cmd)

    print("\n✅ Weekly update completado correctamente.")

if __name__ == "__main__":
    main()


## python src\run_weekly_update.py 
