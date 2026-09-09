"""v3.1 approved appendix arm (review #13 item 6, author-approved
2026-08-25): GARCH-t / GJR-t / FHS re-run with fit_len=512 to match the
TSFM 512-observation context, isolating the history-length component of
the baseline comparison. Everything else (refit cadence, OOS start,
assets, seeds) comes from the frozen stage4 run manifest. Output lives
strictly OUTSIDE the frozen tree: results/v31_w512/. Resumable (skips
existing parquets). Primary registered arm remains fit_len=1000.
"""
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import harness.run_grid_baselines as rgb
from harness.run_grid_baseline_battery import ALPHAS, acol, battery
from data.load import QuarantinedSeriesError, load_series

FIT_LEN = 512
BASELINES = ["garch_t", "gjr_t", "fhs"]
OUT = ROOT / "results" / "v31_w512" / "baselines"
CSV = ROOT / "results" / "v31_w512" / "baseline_backtests_w512.csv"


def main() -> None:
    mf = yaml.safe_load(rgb.MANIFEST.read_text())
    proto, assets = mf["protocol"], mf["assets"]
    refit, oos = proto["refit_every"], proto["oos_start"]
    assert proto["fit_len"] == 1000, "manifest changed; arm assumes 1000 primary"
    OUT.mkdir(parents=True, exist_ok=True)
    for bl in BASELINES:
        for asset in assets:
            pq = OUT / f"{bl}_{asset}.parquet"
            if pq.exists():
                print(f"[skip] {bl}/{asset}", flush=True)
                continue
            try:
                df = load_series(asset)
            except QuarantinedSeriesError:
                print(f"[quarantine] {asset}", flush=True)
                continue
            ctx_col = df.attrs.get("column", "logret")
            t0 = time.time()
            if bl == "garch_t":
                rs, fcs = rgb.garch_family(df, ctx_col, oos, FIT_LEN, refit, gjr=False)
            elif bl == "gjr_t":
                rs, fcs = rgb.garch_family(df, ctx_col, oos, FIT_LEN, refit, gjr=True)
            else:
                rs, fcs, _ = rgb.nonparam(df, ctx_col, oos, FIT_LEN, bl)
            frame = rgb.align_forecasts(rs, fcs)
            rgb.audit_no_lookahead(df, ctx_col, rs)
            frame.to_parquet(pq, index=False)
            meta = {"baseline": bl, "asset": asset, "fit_len": FIT_LEN,
                    "refit_every": refit, "n_windows": rs.meta["n_windows"],
                    "runtime_s": round(time.time() - t0, 1),
                    "actual_oos_start": rs.meta["actual_oos_start"],
                    "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            (OUT / f"{bl}_{asset}.meta.json").write_text(json.dumps(meta, indent=1))
            print(f"[done] {bl}/{asset}: {rs.meta['n_windows']} win "
                  f"{meta['runtime_s']}s", flush=True)

    dm = yaml.safe_load((ROOT / "data" / "data_manifest.yaml").read_text())
    rows = []
    for pq in sorted(OUT.glob("*.parquet")):
        name = pq.stem
        bl = next((b for b in BASELINES if name.startswith(b + "_")), None)
        if bl is None:
            continue
        asset = name[len(bl) + 1:]
        d = pd.read_parquet(pq)
        for a in ALPHAS:
            c = acol(a)
            vcol, ecol = f"v{c}", f"e{c}"
            if vcol not in d:
                continue
            e = d[ecol].to_numpy() if ecol in d and d[ecol].notna().any() else None
            b = battery(d["y"].to_numpy(), d[vcol].to_numpy(), e, a)
            rows.append({"forecaster": bl, "asset": asset, "alpha": a,
                         "erule": "param", "group": dm[asset]["group"],
                         "fit_len": FIT_LEN, **b})
        print(f"[w512-battery] {bl}/{asset}", flush=True)
    pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f"w512 arm complete: {len(rows)} cells -> {CSV}", flush=True)


if __name__ == "__main__":
    main()
