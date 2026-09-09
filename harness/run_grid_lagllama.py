"""Full-grid Lag-Llama analytic forecasts via the persistent server (condition 1).

    nohup python -m harness.run_grid_lagllama > results/stage4/forecast/lag.log 2>&1 &

Analytic Student-t quantiles (the native parametric head) at the DEEP level set,
one ckpt load for all windows. Resumable per asset. Skips quarantined series.
The sampling path (for D3-style S calibration) is not needed at grid scale — the
analytic path IS the head.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from data.load import QuarantinedSeriesError, load_series
from harness.lagllama_adapter import LagLlamaServerAdapter
from harness.rolling import Window, align_forecasts, audit_no_lookahead, rolling_windows

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "stage4" / "run_manifest.yaml"
OUT = ROOT / "results" / "stage4" / "forecast"
DEEP = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
NAMES = ["q" + f"{lv:g}".replace("0.", "") for lv in DEEP]


def main() -> None:
    mf = yaml.safe_load(MANIFEST.read_text())
    proto = mf["protocol"]
    assets = mf["assets"]
    OUT.mkdir(parents=True, exist_ok=True)

    with LagLlamaServerAdapter(seed=proto["seed"]) as srv:
        for asset in assets:
            pq = OUT / f"lag_llama_{asset}.parquet"
            if pq.exists():
                print(f"[skip] lag_llama/{asset}", flush=True)
                continue
            try:
                df = load_series(asset)
            except QuarantinedSeriesError as e:
                print(f"[quarantine] {asset}: {e}", flush=True)
                continue
            rs = rolling_windows(df, value_col=df.attrs.get("column", "logret"),
                                 ctx_len=proto["ctx_len"], oos_start=proto["oos_start"],
                                 allow_late_start=True)
            t0 = time.time()
            forecasts = []
            for w in rs.windows:
                q = srv.predict_quantiles(w.ctx, DEEP)
                forecasts.append(dict(zip(NAMES, map(float, q))))
            frame = align_forecasts(rs, forecasts)
            audit_no_lookahead(df, df.attrs.get("column", "logret"), rs)
            dt = time.time() - t0
            frame.to_parquet(pq, index=False)
            meta = {**rs.meta, "model": "lag_llama", "asset": asset, "levels": DEEP,
                    "runtime_s": round(dt, 1),
                    "s_per_window": round(dt / max(rs.meta["n_windows"], 1), 4),
                    "device": "cpu-venv-server",
                    "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            (OUT / f"lag_llama_{asset}.meta.json").write_text(json.dumps(meta, indent=1))
            print(f"[done] lag_llama/{asset}: {rs.meta['n_windows']} win "
                  f"{dt/60:.1f}min ({meta['s_per_window']}s/win)", flush=True)
    print("lag_llama grid complete", flush=True)


if __name__ == "__main__":
    main()
