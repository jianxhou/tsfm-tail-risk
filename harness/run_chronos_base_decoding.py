"""v2.0 P0-5 (external review #9): Chronos-base decoding-configuration
sensitivity on the 12-asset mechanism sub-panel.

    PYTHONUNBUFFERED=1 python -u -m harness.run_chronos_base_decoding \
        [--device mps] [--assets a,b] [--probes-only]

The pinned amazon/chronos-t5-base ships `top_k=50` (harness/registry.yaml:114);
every pre-v2.0 chronos_base number was computed on the distribution truncated to
the 50 highest-probability tokens of 4096. This driver runs ONE encoder+decoder
forward per window (adapter._step_logits) and decodes the SAME logits under
three configurations post hoc:

    topk50      top_k=50  (pinned vendor default = the delivered numbers' setting)
    topk500     top_k=500 (larger-k sensitivity)
    untruncated top_k=None (full categorical)

Deep quantiles are read from the EXACT categorical CDF (Q(a) = inf{v: F(v)>=a};
no sampling, no MC error). Windows/targets = the mechanism-panel forecast rows
(results/mechanism/forecast/chronos_base_{asset}.parquet), so day sets match the
delivered mechanism backtests exactly.

Outputs (results/mechanism/, git-tracked as of v2.0 — R-7):
  decoding/{asset}.parquet     per-day exact-CDF VaR per config (resumable cache)
  chronos_base_topk.csv        3-config E0 audit on the frozen MC-calibrated
                               battery (kupiec/cc/dq), one row per
                               (asset, config, alpha)
  chronos_base_probes.csv      --probes(-only): the three Table-4 real-window
                               probes (spx/btc/nvda, bolt-grid midpoint windows),
                               script-sourced (R-8) — exact-CDF quantiles per
                               config + seeded native S=1000 sample under vendor
                               defaults + realized-ctx quantiles + stored bolt row.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.load import load_series
from harness.run_grid_extract import acol, battery

ROOT = Path(__file__).resolve().parent.parent
MECH = ROOT / "results" / "mechanism"
FC = MECH / "forecast"
OUT_DIR = MECH / "decoding"
CTX = 512
ALPHAS = (0.01, 0.025, 0.05)
CONFIGS = (("topk50", 50), ("topk500", 500), ("untruncated", None))
PROBE_ASSETS = ("spx", "btc", "nvda")            # Table-4 real windows (R-8)
PROBE_LEVELS = (0.01, 0.05, 0.10, 0.50, 0.90)


def cdf_quantile(vals: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    """Exact categorical quantile Q(alpha) = inf{v : F(v) >= alpha}."""
    o = np.argsort(vals, kind="stable")
    cdf = np.cumsum(probs[o])
    i = int(np.searchsorted(cdf, alpha, side="left"))
    return float(vals[o][min(i, len(o) - 1)])


def asset_frame(adapter, asset: str) -> pd.DataFrame:
    """Per-day exact-CDF VaR for the three configs on the mechanism windows."""
    fc = pd.read_parquet(FC / f"chronos_base_{asset}.parquet")
    df = load_series(asset)
    x = df[df.attrs.get("column", "logret")].to_numpy()
    rows = []
    t0 = time.time()
    for j, (_, r) in enumerate(fc.iterrows()):
        t = int(r["t"])
        vals, logits = adapter._step_logits(x[t - CTX:t].astype(np.float32))
        rec = {"t": t, "date": r["date"], "y": r["y"]}
        for cname, k in CONFIGS:
            probs = adapter.decode_probs(logits, top_k=k)
            for a in ALPHAS:
                rec[f"{cname}_v{acol(a)}"] = cdf_quantile(vals, probs, a)
        rows.append(rec)
        if (j + 1) % 250 == 0:
            el = time.time() - t0
            print(f"[decode] {asset} {j+1}/{len(fc)} "
                  f"({el/(j+1):.2f}s/win, ETA {(len(fc)-j-1)*el/(j+1)/60:.0f}min)",
                  flush=True)
    return pd.DataFrame(rows)


def probes(adapter, seed: int = 20260706) -> pd.DataFrame:
    """Script-sourced Table-4 real-window probes (R-8): same windows as
    docs/chronos_base_forensic.py (bolt-grid midpoint), exact-CDF quantiles per
    decoding config + seeded native-convention S=1000 sample (vendor defaults)
    + realized-ctx quantiles + the stored chronos_bolt row."""
    rng = np.random.default_rng(seed)
    rows = []
    for asset in PROBE_ASSETS:
        x = pd.read_parquet(ROOT / "data" / "parquet" / f"{asset}.parquet"
                            )["logret"].to_numpy()
        b = pd.read_parquet(ROOT / "results" / "stage4" / "forecast"
                            / f"chronos_bolt_{asset}.parquet")
        row = b.iloc[len(b) // 2]                # same real window as the memo
        t = int(row["t"])
        ctx = x[t - CTX:t].astype(np.float32)
        vals, logits = adapter._step_logits(ctx)
        for cname, k in CONFIGS:
            probs = adapter.decode_probs(logits, top_k=k)
            rec = {"asset": asset, "t": t, "date": row["date"],
                   "method": f"exact_cdf_{cname}"}
            for lv in PROBE_LEVELS:
                rec[f"q{acol(lv)}"] = cdf_quantile(vals, probs, lv)
            rows.append(rec)
        p50 = adapter.decode_probs(logits)       # vendor defaults, S=1000 draw
        p50 = p50 / p50.sum()
        s = vals[rng.choice(len(p50), size=1000, replace=True, p=p50)]
        rec = {"asset": asset, "t": t, "date": row["date"],
               "method": "native_sample_1000_topk50",
               "n_uniq_samples": int(len(np.unique(s)))}
        for lv in PROBE_LEVELS:
            rec[f"q{acol(lv)}"] = float(np.quantile(s, lv))
        rows.append(rec)
        rec = {"asset": asset, "t": t, "date": row["date"], "method": "realized_ctx"}
        for lv in PROBE_LEVELS:
            rec[f"q{acol(lv)}"] = float(np.quantile(ctx, lv))
        rows.append(rec)
        rows.append({"asset": asset, "t": t, "date": row["date"],
                     "method": "chronos_bolt_stored",
                     "q01": float(row["q01"]), "q05": float(row["q05"]),
                     "q1": float(row["q1"])})
    return pd.DataFrame(rows)


# ---- v2.1 P2-4 (review #10 F4): the CONTROLLED sine/trend/t5 probes ----
# tab:probes in sec9 prints the controlled-probe set, which is DISTINCT from
# the real-window probe set of chronos_base_probes.csv above; until v2.1 the
# controlled probes had no in-repo generator (the memo recorded a session-
# transient run whose seeds were not retained). This mode regenerates them
# under pinned seeds on the official ChronosPipeline with vendor defaults
# (S=1000, CPU float32 — the forensic caliber of docs/chronos_base_forensic.py)
# and writes results/mechanism/chronos_base_controlled_probes.csv.

CP_SEED = 20260802
CP_S = 1000
# inline tab:probes values — v2.1 Phase-4 sanctioned re-shot (flag 2, adjudication
# §7): trend/t5 re-anchored to this seeded generator's output at printed precision
# (the original forensic session's sampling/noise seeds were not retained; sine
# reproduces the memo value); --assert-inline now hard-enforces all three rows
CP_INLINE = {"sine":  {"truth": -2.939, "median": -2.931, "spread": 0.023},
             "trend": {"truth": 25.6,   "median": 25.48,  "spread": 0.656},
             "t5":    {"realized_q10": -1.61, "median": -0.050, "spread": 0.545}}


def controlled_probes(seed: int = CP_SEED) -> pd.DataFrame:
    import torch
    import yaml
    from chronos import ChronosPipeline
    from scipy import stats as sps
    rev = yaml.safe_load((ROOT / "harness" / "registry.yaml").read_text())[
        "chronos_base"]["hf_revision"]
    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-base",
                                           revision=rev, device_map="cpu",
                                           torch_dtype=torch.float32)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    i = np.arange(CTX)
    series = {
        "sine":  (5.0 * np.sin(2 * np.pi * i / 20),
                  5.0 * np.sin(2 * np.pi * CTX / 20),
                  "period 20, amplitude 5"),
        "trend": (0.05 * i + 0.3 * rng.standard_normal(CTX),
                  0.05 * CTX,
                  "slope 0.05, noise 0.3 (deterministic part as truth)"),
        "t5":    (sps.t.rvs(5, size=CTX, random_state=rng),
                  np.nan,
                  "iid standard t(5)"),
    }
    rows = []
    for name, (x, truth, spec) in series.items():
        ctx = x.astype(np.float32)
        torch.manual_seed(seed)
        parts = []
        with torch.no_grad():
            for _ in range(CP_S // 100):             # chunked ONLY for memory
                s = pipe.predict(inputs=torch.tensor(ctx).unsqueeze(0),
                                 prediction_length=1, num_samples=100)
                parts.append(np.asarray(s).reshape(-1))
        s = np.concatenate(parts)
        q = lambda arr, lv: float(np.quantile(arr, lv))
        rows.append({
            "probe": name, "spec": spec, "seed": seed, "S": CP_S,
            "revision": rev, "device": "cpu", "dtype": "float32",
            "truth": float(truth) if np.isfinite(truth) else np.nan,
            "realized_q10": q(ctx, 0.10), "realized_q90": q(ctx, 0.90),
            "pred_median": q(s, 0.50), "pred_q10": q(s, 0.10),
            "pred_q90": q(s, 0.90), "spread": q(s, 0.90) - q(s, 0.10),
            "n_unique": int(len(np.unique(s))),
        })
        print(f"[controlled] {name}: median={rows[-1]['pred_median']:+.3f} "
              f"spread={rows[-1]['spread']:.3f}", flush=True)
    return pd.DataFrame(rows)


def controlled_report(pr: pd.DataFrame, assert_inline: bool) -> None:
    """Structural claims of tab:probes (hard asserts) + inline-value
    comparison at printed precision (report; gate only with --assert-inline)."""
    r = pr.set_index("probe")
    # structural: sine/trend tracked accurately with tight uncertainty
    assert abs(r.loc["sine", "pred_median"] - r.loc["sine", "truth"]) < 0.05
    assert r.loc["sine", "spread"] < 0.10
    assert abs(r.loc["trend", "pred_median"] - r.loc["trend", "truth"]) < 0.50
    assert r.loc["trend", "spread"] < 2.0
    # structural: on iid t(5) noise the predictive spread collapses far below
    # the realized spread (the overconfidence the memo/table demonstrate)
    realized_spread = r.loc["t5", "realized_q90"] - r.loc["t5", "realized_q10"]
    assert r.loc["t5", "spread"] < 0.5 * realized_spread
    print("[controlled] structural claims of tab:probes: ALL HOLD")
    dev = []
    for name, exp in CP_INLINE.items():
        for k, v in exp.items():
            col = {"median": "pred_median"}.get(k, k)
            got = float(r.loc[name, col])
            tol = {"truth": 5e-4 if name == "sine" else 0.05,
                   "median": 5e-4, "spread": 5e-4,
                   "realized_q10": 5e-3}[k]
            if abs(got - v) > tol:
                dev.append(f"  {name}.{k}: regenerated {got:+.4f} vs inline "
                           f"{v:+.4f} (printed-precision tol {tol})")
    if dev:
        print("[controlled] inline-value deviations (expected: original seeds "
              "not retained; Phase-4 re-shot decision):")
        print("\n".join(dev))
        if assert_inline:
            raise SystemExit("--assert-inline: inline tab:probes values do "
                             "not reproduce at printed precision")
    else:
        print("[controlled] inline tab:probes values reproduce at printed "
              "precision")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--assets", default="")
    ap.add_argument("--probes-only", action="store_true")
    ap.add_argument("--controlled-probes", action="store_true")
    ap.add_argument("--assert-inline", action="store_true")
    cli = ap.parse_args()

    if cli.controlled_probes:
        pr = controlled_probes()
        pr.to_csv(MECH / "chronos_base_controlled_probes.csv", index=False)
        controlled_report(pr, cli.assert_inline)
        print(f"controlled probes complete: {len(pr)} rows", flush=True)
        return

    from harness.chronos_base_adapter import ChronosBaseAdapter
    adapter = ChronosBaseAdapter(device=cli.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if cli.probes_only:
        pr = probes(adapter)
        pr.to_csv(MECH / "chronos_base_probes.csv", index=False)
        print(f"probes complete: {len(pr)} rows", flush=True)
        return

    assets = sorted(p.stem.replace("chronos_base_", "")
                    for p in FC.glob("chronos_base_*.parquet"))
    if cli.assets:
        assets = [a for a in cli.assets.split(",") if a in assets]

    rows = []
    for asset in assets:
        pq = OUT_DIR / f"{asset}.parquet"
        if pq.exists():
            d = pd.read_parquet(pq)
            print(f"[decode] {asset} cached ({len(d)} windows)", flush=True)
        else:
            t0 = time.time()
            d = asset_frame(adapter, asset)
            d.to_parquet(pq, index=False)
            print(f"[decode] {asset} done ({len(d)} windows, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
        y = d["y"].to_numpy()
        for cname, _ in CONFIGS:
            for a in ALPHAS:
                b = battery(y, d[f"{cname}_v{acol(a)}"].to_numpy(), None, a)
                rows.append({"asset": asset, "config": cname, "alpha": a,
                             "erule": "e0", **b})
    pd.DataFrame(rows).to_csv(MECH / "chronos_base_topk.csv", index=False)
    print(f"chronos_base decoding audit complete: {len(rows)} cells", flush=True)


if __name__ == "__main__":
    main()
