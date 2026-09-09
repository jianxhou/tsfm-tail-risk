"""Mechanism sub-panel forecast-to-disk (proposal §3.1; signer-authorized
2026-07-16). Two within-family contrast heads x the 12-asset subset, rolling
1-step native forecasts at ctx=512, OOS from the frozen stage-4 protocol.

    python -m harness.run_mechanism_forecast [--assets a,b] [--models m]

Both heads are sampling-native; predict_quantiles returns S=1000 empirical
DEEP-grid quantiles (same schema + columns as the main grid, so run_grid_extract
logic applies unchanged). One parquet + meta json per (model, asset), resumable
(skips existing). Run several --assets subsets in parallel to compress wall-clock.

Protocol is inherited verbatim from results/stage4/run_manifest.yaml (ctx_len,
oos_start); this run adds NO new protocol freedom — it extends the registered
panel to the mechanism heads only.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")
for _n in ("chronos", "transformers", "gluonts", "uni2ts"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from data.load import QuarantinedSeriesError, load_series
from harness.rolling import run_rolling

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "stage4" / "run_manifest.yaml"
OUT = ROOT / "results" / "mechanism" / "forecast"

DEEP = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
ASSETS12 = ["brent", "gold", "btc", "eth", "hsi", "spx",
            "aapl", "nvda", "audusd", "gbpusd", "dgs10", "dgs2"]
ADAPTERS = {
    "chronos_base": ("harness.chronos_base_adapter", "ChronosBaseAdapter"),
    "moirai_1_1": ("harness.moirai11_adapter", "Moirai11Adapter"),
}


def colname(lv: float) -> str:
    return "q" + f"{lv:g}".replace("0.", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="", help="comma-separated subset (default: all 12)")
    ap.add_argument("--models", default="", help="comma-separated subset (default: both)")
    cli = ap.parse_args()

    mf = yaml.safe_load(MANIFEST.read_text())
    proto = mf["protocol"]
    assets = cli.assets.split(",") if cli.assets else ASSETS12
    models = cli.models.split(",") if cli.models else list(ADAPTERS)
    bad = [a for a in assets if a not in ASSETS12]
    if bad:
        raise SystemExit(f"unknown mechanism assets: {bad}")
    OUT.mkdir(parents=True, exist_ok=True)
    names = [colname(lv) for lv in DEEP]

    for model in models:
        mod_name, cls_name = ADAPTERS[model]
        adapter = None
        for asset in assets:
            pq = OUT / f"{model}_{asset}.parquet"
            if pq.exists():
                print(f"[skip] {model}/{asset}", flush=True)
                continue
            try:
                df = load_series(asset)
            except QuarantinedSeriesError as e:
                print(f"[quarantine] {asset}: {e}", flush=True)
                continue
            if adapter is None:
                mod = __import__(mod_name, fromlist=[cls_name])
                adapter = getattr(mod, cls_name)()

            def fn(ctx: np.ndarray) -> dict:
                q = adapter.predict_quantiles(ctx, DEEP)
                return dict(zip(names, map(float, q)))

            t0 = time.time()
            frame, meta = run_rolling(df, fn, value_col=df.attrs.get("column", "logret"),
                                      ctx_len=proto["ctx_len"], oos_start=proto["oos_start"],
                                      allow_late_start=True)
            dt = time.time() - t0
            frame.to_parquet(pq, index=False)
            meta.update(model=model, asset=asset, levels=DEEP, runtime_s=round(dt, 1),
                        s_per_window=round(dt / max(meta["n_windows"], 1), 4), device="cpu",
                        panel="mechanism", n_samples=1000,
                        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            (OUT / f"{model}_{asset}.meta.json").write_text(json.dumps(meta, indent=1))
            print(f"[done] {model}/{asset}: {meta['n_windows']} win {dt/60:.1f}min "
                  f"({dt/max(meta['n_windows'],1):.3f}s/win)", flush=True)
    print("mechanism forecast complete", flush=True)


if __name__ == "__main__":
    main()
