"""GO#3 compute extrapolation, per model (script-generated -> docs/pilot_go3_compute.md).

    python -m harness.run_pilot_go3

Per-window cost = measured D2 pilot runtimes (results/pilot_d2/*.meta.json, CPU,
this machine). Full-grid window count = EXACT count from the parquet panel:
for each of the 34 series, targets with date >= OOS_START and >= CTX_LEN prior
rows (same rule as harness.rolling). lag_llama is measured live here (one
subprocess params-mode call — the analytic path the grid would actually use)
and listed separately: its per-call cost is dominated by checkpoint reload,
which the Stage-4 persistent-worker TODO amortizes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
D2 = ROOT / "results" / "pilot_d2"
CTX_LEN = 512
OOS_START = "2016-01-01"   # proposal §3.2 main OOS


def grid_windows() -> tuple[int, dict]:
    manifest = yaml.safe_load((ROOT / "data" / "data_manifest.yaml").read_text())
    per = {}
    for sid, entry in manifest.items():
        df = pd.read_parquet(ROOT / entry["parquet"])
        dates = pd.to_datetime(df["date"])
        first = int(np.searchsorted(dates.values, np.datetime64(OOS_START)))
        per[sid] = max(0, len(df) - max(first, CTX_LEN))
    return sum(per.values()), per


def d2_per_window() -> dict:
    agg: dict[str, list] = {}
    for f in D2.glob("*.meta.json"):
        m = json.loads(f.read_text())
        agg.setdefault(m["model"], []).append((m["runtime_s"], m["n_windows"]))
    return {k: sum(r for r, _ in v) / sum(n for _, n in v) for k, v in agg.items()}


def measure_lag_llama() -> float:
    from harness.lagllama_adapter import LagLlamaAdapter
    rng = np.random.default_rng(20260706)
    ctx = (rng.standard_t(5, size=CTX_LEN) * 1.2).astype(np.float32)
    ad = LagLlamaAdapter()
    t0 = time.time()
    ad.predict_quantiles(ctx, [0.01, 0.025, 0.05])
    return time.time() - t0


def main() -> None:
    W, per = grid_windows()
    spw = d2_per_window()
    t_lag = measure_lag_llama()

    lines = ["# GO#3 compute extrapolation (script-generated; do not hand-edit)", "",
             f"Assumptions: full grid = 34 series (the committed panel), h=1, ctx={CTX_LEN},",
             f"main OOS {OOS_START} -> panel end, per-series trading calendars.",
             f"Exact window count from the parquet store: **W = {W:,} windows per model**",
             f"(min/median/max per series: {min(per.values())}/"
             f"{int(np.median(list(per.values())))}/{max(per.values())}).",
             "Per-window costs are measured D2 pilot values (CPU, this machine) —",
             "an upper bound for Stage-4 rented-GPU runs.", "",
             "| model | s/window (measured) | full-grid hours (CPU) |",
             "|---|---|---|"]
    for m in ("chronos_bolt", "chronos_2", "timesfm_2_5", "moirai_2_0"):
        lines.append(f"| {m} | {spw[m]:.3f} | {W * spw[m] / 3600:.1f} |")
    lines += [
        f"| lag_llama (analytic path, subprocess-per-call as measured NOW) "
        f"| {t_lag:.1f} | {W * t_lag / 3600:.0f} |", "",
        "lag_llama's per-call cost is dominated by checkpoint reload in the "
        "quarantined venv; the Stage-4 persistent-worker TODO amortizes the load, "
        "leaving the forward pass (sub-second) — engineering precondition for its "
        "full-grid run, tracked in registry/PROGRESS. ctx ablations {256, 2048} on "
        "the 12-asset mechanism subset roughly add 2 x (12/34) of each model's "
        "full-grid cost. Econometric baselines ran at minutes-per-asset in D4 "
        "(CPU) and are not a budget factor. Mechanism-panel sampling heads "
        "(chronos_base, S=1000) are unmeasured until their Stage-2 adapters land "
        "- flagged, not extrapolated.", ""]
    (ROOT / "docs" / "pilot_go3_compute.md").write_text("\n".join(lines))
    print(f"W={W:,}; lag_llama call {t_lag:.1f}s; table written", flush=True)


if __name__ == "__main__":
    main()
