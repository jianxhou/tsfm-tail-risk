"""Stage-4 main-grid forecast-to-disk (checklist item 5; amendment A: may run
before the MC-calibration judgment lock — this writes RAW forecasts only, no
backtest reads).

    nohup python -m harness.run_grid_forecast > results/stage4/forecast/run.log 2>&1 &

Reads the frozen run_manifest (refuses to run if config drifted). For each of the
4 quantile heads x 32 grid-eligible assets: rolling 1-step native quantiles at
ctx=512, OOS from the manifest, all slicing via harness.rolling. One parquet +
meta json per (model, asset), resumable. Quarantined series are skipped by
data.load (condition 7). lag_llama's full-grid run waits on the persistent worker
(condition 1) and is NOT launched here.

Amendment D: first-hour wall-clock is checked against the pilot extrapolation;
if a cell's measured s/window exceeds 2x the pilot value, the run self-halts with
a report line (the operator decides).
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

# The Chronos-Bolt clamp warning fires once per window (the clamp IS our H1 finding,
# already documented in the registry); silence the per-window flood in the grid log.
warnings.filterwarnings("ignore")
logging.getLogger("chronos").setLevel(logging.ERROR)
for _n in ("chronos.chronos_bolt", "transformers", "gluonts"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from data.load import QuarantinedSeriesError, load_series
from harness.rolling import run_rolling

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "stage4" / "run_manifest.yaml"
OUT = ROOT / "results" / "stage4" / "forecast"

DEEP = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
DECILES = [round(0.1 * k, 1) for k in range(1, 10)]
ADAPTERS = {
    "chronos_bolt": ("harness.chronos_adapters", "ChronosBoltAdapter", DEEP),
    "chronos_2": ("harness.chronos_adapters", "Chronos2Adapter", DEEP),
    "timesfm_2_5": ("harness.timesfm_adapter", "TimesFM25Adapter", DECILES),
    "moirai_2_0": ("harness.moirai_adapter", "Moirai2Adapter", DEEP),
}
# pilot s/window (results/pilot_go3_compute.md) for the 2x guard
PILOT_SPW = {"chronos_bolt": 0.041, "chronos_2": 0.039,
             "timesfm_2_5": 0.095, "moirai_2_0": 0.008}


def colname(lv: float) -> str:
    return "q" + f"{lv:g}".replace("0.", "")


def main() -> None:
    import subprocess
    chk = subprocess.run(["python", "-m", "harness.freeze_run_manifest", "--check"],
                         capture_output=True, text=True, cwd=ROOT)
    if chk.returncode != 0:
        raise SystemExit(f"manifest check failed: {chk.stdout}{chk.stderr}")

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="", help="comma-separated subset (default: all)")
    cli = ap.parse_args()

    mf = yaml.safe_load(MANIFEST.read_text())
    proto = mf["protocol"]
    assets = mf["assets"]
    if cli.assets:
        subset = cli.assets.split(",")
        unknown = [a for a in subset if a not in assets]
        if unknown:
            raise SystemExit(f"unknown assets: {unknown}")
        assets = subset
        print(f"SUBSET run: {len(assets)} assets", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grid_status.json").write_text(json.dumps(
        {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "config_hash": mf["config_hash"], "assets": assets}, indent=1))

    consec_slow = 0   # amendment D: halt only on SUSTAINED slowdown (2 consecutive
                      # over-threshold cells). A single spike is a transient (e.g. a
                      # system pause inflating wall-clock) — warn, don't die.
    for model, (mod_name, cls_name, levels) in ADAPTERS.items():
        adapter = None
        names = [colname(lv) for lv in levels]
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
                q = adapter.predict_quantiles(ctx, levels)
                return dict(zip(names, map(float, q)))

            t0 = time.time()
            # allow_late_start (A3 panel mode): series starting < ctx_len before the
            # 2016 OOS (btc 2014-09, csi300 2013-11, eth/xrp/ltc 2017-11) begin at
            # their first full-ctx target; per-series actual_oos_start is in meta.
            frame, meta = run_rolling(df, fn, value_col=df.attrs.get("column", "logret"),
                                      ctx_len=proto["ctx_len"],
                                      oos_start=proto["oos_start"],
                                      allow_late_start=True)
            dt = time.time() - t0
            spw = dt / max(meta["n_windows"], 1)
            frame.to_parquet(pq, index=False)
            meta.update(model=model, asset=asset, levels=levels, runtime_s=round(dt, 1),
                        s_per_window=round(spw, 4), device="cpu",
                        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            (OUT / f"{model}_{asset}.meta.json").write_text(json.dumps(meta, indent=1))
            if spw > 2 * PILOT_SPW[model]:
                consec_slow += 1
                print(f"[done] {model}/{asset}: {meta['n_windows']} win {dt/60:.1f}min "
                      f"({spw:.3f}s/win)  !! {spw/PILOT_SPW[model]:.1f}x pilot "
                      f"[consec_slow={consec_slow}] — output validated, likely a "
                      f"transient (system pause) unless it persists", flush=True)
                if consec_slow >= 2:
                    raise SystemExit(f"amendment D: {consec_slow} consecutive cells "
                                     f"> 2x pilot s/window — SUSTAINED slowdown, halting")
            else:
                consec_slow = 0
                print(f"[done] {model}/{asset}: {meta['n_windows']} win {dt/60:.1f}min "
                      f"({spw:.3f}s/win)", flush=True)
    print("grid forecast complete (4 quantile heads x grid-eligible assets)", flush=True)


if __name__ == "__main__":
    main()
