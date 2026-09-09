"""Pilot D2 driver: zero-shot rolling 1-step forecasts, 3 assets x 4 TSFMs.

    nohup python -m harness.run_pilot_d2 > results/pilot_d2/run.log 2>&1 &

ctx=512 (proposal main setting), OOS 2018-01-01 -> latest, CPU (deterministic;
GPU is a Stage-4 concern). Levels = each model's feasible native set — E-rules
(D3) consume these stored native outputs offline. One parquet + meta json per
(model, asset), skipped if present, so the run is resumable and every pair is a
checkpoint. All slicing via harness.rolling (single reviewed rolling module).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data.load import load_series
from harness.rolling import run_rolling

OUT = Path(__file__).resolve().parent.parent / "results" / "pilot_d2"
CTX_LEN = 512
OOS_START = "2018-01-01"
ASSETS = ["spx", "btc", "nvda"]

DEEP = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
DECILES = [round(0.1 * k, 1) for k in range(1, 10)]

MODELS = {
    "chronos_bolt": ("harness.chronos_adapters", "ChronosBoltAdapter", DEEP),
    "chronos_2": ("harness.chronos_adapters", "Chronos2Adapter", DEEP),
    "timesfm_2_5": ("harness.timesfm_adapter", "TimesFM25Adapter", DECILES),
    "moirai_2_0": ("harness.moirai_adapter", "Moirai2Adapter", DEEP),
}


def colname(lv: float) -> str:
    return "q" + f"{lv:g}".replace("0.", "")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model_key, (mod_name, cls_name, levels) in MODELS.items():
        adapter = None
        names = [colname(lv) for lv in levels]
        for asset in ASSETS:
            pq = OUT / f"{model_key}_{asset}.parquet"
            if pq.exists():
                print(f"[skip] {model_key}/{asset} exists", flush=True)
                continue
            if adapter is None:
                mod = __import__(mod_name, fromlist=[cls_name])
                adapter = getattr(mod, cls_name)()
            df = load_series(asset)

            def fn(ctx: np.ndarray) -> dict:
                q = adapter.predict_quantiles(ctx, levels)
                return dict(zip(names, map(float, q)))

            t0 = time.time()
            frame, meta = run_rolling(df, fn, value_col="logret",
                                      ctx_len=CTX_LEN, oos_start=OOS_START)
            dt = time.time() - t0
            frame.to_parquet(pq, index=False)
            meta.update(model=model_key, asset=asset, levels=levels,
                        runtime_s=round(dt, 1), device="cpu",
                        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            (OUT / f"{model_key}_{asset}.meta.json").write_text(json.dumps(meta, indent=1))
            print(f"[done] {model_key}/{asset}: {meta['n_windows']} windows "
                  f"in {dt/60:.1f} min", flush=True)
    print("D2 rolling runs complete", flush=True)


if __name__ == "__main__":
    main()
