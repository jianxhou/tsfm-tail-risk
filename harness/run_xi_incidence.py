"""xi>=1 incidence (ES-existence boundary) — appendix material, NOT dirty data.

    PYTHONUNBUFFERED=1 python -u -m harness.run_xi_incidence

GPD-fit sites where ES (mean-excess) is undefined for xi>=1:
  e3          : GPD on raw context returns (per asset; model-independent — ctxfits).
  residual_f1 : GPD on TSFM-filtered residuals (per model x asset; F1 filter —
                q50 location, IQR/1.349 scale, POT 10%, W=500).
  residual_h  : H filter (q50 location, GARCH sigma scale) — same crashing
                function as F1 (rulings A12; previously unmeasured).
  residual_ge : GARCH-EVT filter (GARCH mu location, sigma scale; model-
                independent) — likewise previously unmeasured.
Residual windows are aligned to the repair-grid frames (t in ctxfits AND the
GARCH grid), fixing the earlier crypto over-count (rulings audit XI-6).
Clustering text below the tables is COMPUTED from the data (the earlier
hardcoded sentence was contradicted by its own tables — rulings A4).
Output: results/stage4/xi_incidence.csv, docs/stage5_xi_incidence.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.load import load_series
from erules.rules import fit_gpd, mad
from fixes.arms import TAU_U, W
from harness import run_grid_repairs as base
from harness.rolling import calibration_view

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"
FC = S4 / "forecast"
TSFM = ["chronos_bolt", "chronos_2", "timesfm_2_5", "moirai_2_0", "lag_llama"]


def e3_xi_incidence(asset) -> tuple[int, int]:
    """xi>=1 fraction on the E3 GPD (from cached ctxfits)."""
    fp = S4 / "ctxfits" / f"{asset}.parquet"
    if not fp.exists():
        return 0, 0
    xi = pd.read_parquet(fp)["xi"].to_numpy()
    fin = np.isfinite(xi)
    return int((xi[fin] >= 1.0).sum()), int(fin.sum())


def _count_ge1(al: pd.DataFrame, loc: np.ndarray, sc: np.ndarray):
    """Per-window GPD xi>=1 count on residuals z=(y-loc)/sc over trailing W."""
    frame = al[["t", "y"]].copy()
    frame["loc"] = loc
    frame["sc"] = sc
    ge1, tot = 0, 0
    for i in range(len(frame)):
        h = calibration_view(frame, frame["t"].iloc[i], window=W)
        if len(h) < 100:
            continue
        z = (h["y"].to_numpy() - h["loc"].to_numpy()) / h["sc"].to_numpy()
        u = float(np.quantile(z, TAU_U))
        exc = u - z[z < u]
        try:
            xi, _, _ = fit_gpd(exc, scale_ref=mad(z))   # v2.0 P0-1 tie-robust fit
        except ValueError:
            continue
        tot += 1
        ge1 += int(xi >= 1.0)
    return ge1, tot


def resid_sites(model, asset, fits, x, oos):
    """(f1, h, ge) xi>=1 counts on repair-grid-aligned windows."""
    e3, _, mu, sig = base.build_aligned(model, asset, fits, x, oos)
    q50 = e3["q50"].to_numpy()
    s_iqr = ((e3["q75"] - e3["q25"]) / 1.349).to_numpy()
    f1 = _count_ge1(e3, q50, s_iqr)
    h = _count_ge1(e3, q50, sig)
    ge = _count_ge1(e3, mu, sig)
    return f1, h, ge


def _input_digest(assets, dm) -> str:
    """v2.1.1 (review #11 _xi_partial empty-run trap): full input basis of one
    run, hashed. Missing files hard-error UP FRONT — under this check the
    expected cell grid per asset is fully determined (12 rows: 1 e3 +
    2 x 5 TSFM residual sites + 1 residual_ge), so the old silent
    forecast-file skip can no longer freeze an incomplete asset."""
    files = [S4 / "run_manifest.yaml", ROOT / "data" / "data_manifest.yaml"]
    for a in assets:
        files += [S4 / "ctxfits" / f"{a}.parquet", ROOT / dm[a]["parquet"]]
        files += [FC / f"{m}_{a}.parquet" for m in TSFM]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"[xi] HARD ERROR — inputs missing, the cell grid "
                         f"would be silently incomplete: {missing[:8]}"
                         f"{' ...' if len(missing) > 8 else ''}")
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


EXPECTED_CELLS = ({("(E3, all)", "e3"), ("(GARCH-EVT)", "residual_ge")}
                  | {(m, s) for m in TSFM
                     for s in ("residual_f1", "residual_h")})


def main():
    mf = yaml.safe_load((S4 / "run_manifest.yaml").read_text())
    oos = mf["protocol"]["oos_start"]
    dm = yaml.safe_load((ROOT / "data" / "data_manifest.yaml").read_text())
    assets = mf["assets"]
    part = S4 / "_xi_partial.csv"           # kill-resumable per-asset persistence
    meta = S4 / "_xi_partial.inputs.json"   # v2.1.1: checkpoint input basis
    digest = _input_digest(assets, dm)
    if part.exists():
        # v2.1.1 (review #11): a checkpoint may only be resumed against the
        # EXACT input state it was built from — otherwise the old code
        # re-emitted stale numbers under a fresh timestamp (empty-run trap)
        if not meta.exists() or json.loads(meta.read_text())["sha256"] != digest:
            raise SystemExit(
                "[xi] HARD ERROR — results/stage4/_xi_partial.csv was built "
                "from different inputs (fingerprint mismatch or missing "
                "_xi_partial.inputs.json). Delete the checkpoint and re-run.")
        prev = pd.read_csv(part)
        # cell-complete done-test: an asset counts as done only with the full
        # 12-row grid; torn/partial assets are dropped and recomputed
        done = {a for a, sub in prev.groupby("asset")
                if set(zip(sub.model, sub.site)) == EXPECTED_CELLS}
        prev[prev.asset.isin(done)].to_csv(part, index=False)
    else:
        meta.write_text(json.dumps({"sha256": digest}))
        done = set()
    for asset in assets:
        if asset in done:
            continue
        rows = []
        grp = dm[asset]["group"]
        e_ge, e_tot = e3_xi_incidence(asset)
        rows.append({"asset": asset, "group": grp, "model": "(E3, all)",
                     "site": "e3", "xi_ge1": e_ge, "n": e_tot,
                     "frac": round(e_ge / e_tot, 4) if e_tot else np.nan})
        fits = pd.read_parquet(S4 / "ctxfits" / f"{asset}.parquet").set_index("t")
        df = load_series(asset)
        x = df[df.attrs.get("column", "logret")].to_numpy()
        ge_site = None
        for m in TSFM:
            if not (FC / f"{m}_{asset}.parquet").exists():
                continue
            (f_ge, f_n), (h_ge, h_n), ge_site = resid_sites(m, asset, fits, x, oos)
            rows.append({"asset": asset, "group": grp, "model": m,
                         "site": "residual_f1", "xi_ge1": f_ge, "n": f_n,
                         "frac": round(f_ge / f_n, 4) if f_n else np.nan})
            rows.append({"asset": asset, "group": grp, "model": m,
                         "site": "residual_h", "xi_ge1": h_ge, "n": h_n,
                         "frac": round(h_ge / h_n, 4) if h_n else np.nan})
        if ge_site is not None:                      # model-independent
            g_ge, g_n = ge_site
            rows.append({"asset": asset, "group": grp, "model": "(GARCH-EVT)",
                         "site": "residual_ge", "xi_ge1": g_ge, "n": g_n,
                         "frac": round(g_ge / g_n, 4) if g_n else np.nan})
        pd.DataFrame(rows).to_csv(part, mode="a", header=not part.exists(),
                                  index=False)
        print(f"[xi] {asset} done", flush=True)
    df = pd.read_csv(part)
    df.to_csv(S4 / "xi_incidence.csv", index=False)
    _summary(df)
    # v2.1.1: a completed run leaves NO checkpoint — re-running recomputes
    # from live inputs instead of replaying the frozen partial (trap fix)
    part.unlink()
    if meta.exists():
        meta.unlink()
    print(f"xi incidence complete: {len(df)} rows", flush=True)


def _summary(df):
    L = ["# xi>=1 incidence — ES existence boundary (script-generated; appendix)", "",
         "xi>=1 means the fitted GPD has infinite mean, so ES (mean-excess) is",
         "undefined and correctly reported NaN. VaR remains finite throughout:",
         "the exception paths are independent (v2.0 P0-1), so an undefined ES",
         "never voids the day's VaR. The counts below are from the tie-robust",
         "estimator (machine-precision tie removal + PWM fallback on boundary",
         "MLE fits): of the 586 xi>=1 events recorded before the v2.0",
         "correction, 490 (83.6%) were parameter-boundary artifacts of ML GPD",
         "estimation on tie-laden exceedances from 1-bp-quantized rate levels,",
         "and the pre-correction reading of the incidence as an empirical",
         "feature of post-2016 rate dynamics is WITHDRAWN (impact report §1;",
         "paper §7.3). The remaining windows are genuine heavy-tail fits — a",
         "property of the tail, not dirty data. Residual windows aligned to the",
         "repair-grid frames; residual sites cover the F1, H and GARCH-EVT",
         "filters (rulings A12/XI-6).", "",
         "## E3 side (GPD on raw context returns; model-independent), by asset class", "",
         "| group | mean xi>=1 frac | max frac (asset) |", "|---|---|---|"]
    e3 = df[df.site == "e3"]
    for g, sub in e3.groupby("group"):
        worst = sub.loc[sub.frac.idxmax()]
        L.append(f"| {g} | {sub.frac.mean():.4f} | {worst.frac:.4f} ({worst.asset}) |")
    L += ["", "## Residual sides (GPD on filtered residuals), by model x site", "",
          "| model | site | mean frac | max frac (asset) |", "|---|---|---|---|"]
    for site in ("residual_f1", "residual_h", "residual_ge"):
        rs = df[df.site == site]
        for m, sub in rs.groupby("model"):
            sub = sub.dropna(subset=["frac"])
            if len(sub):
                worst = sub.loc[sub.frac.idxmax()]
                L.append(f"| {m} | {site.replace('residual_', '')} | "
                         f"{sub.frac.mean():.4f} | {worst.frac:.4f} ({worst.asset}) |")
    # COMPUTED clustering statement (replaces the voided hardcoded sentence)
    nz_e3 = e3[e3.xi_ge1 > 0].sort_values("frac", ascending=False)
    nz_r = df[(df.site != "e3") & (df.xi_ge1 > 0)].sort_values("frac", ascending=False)
    L += ["", "## Clustering (computed)"]
    if len(nz_e3):
        cells = "; ".join(f"{r.asset} {100*r.frac:.1f}% ({r.xi_ge1}/{r.n})"
                          for r in nz_e3.itertuples())
        grps = ", ".join(sorted(nz_e3.group.unique()))
        L.append(f"E3 side: ALL nonzero incidence sits in {grps} — {cells}. "
                 f"Every other asset class is exactly zero.")
    else:
        L.append("E3 side: no xi>=1 fits anywhere.")
    if len(nz_r):
        cells = "; ".join(
            f"{r.asset} x {r.model} [{r.site.replace('residual_', '')}] "
            f"{r.xi_ge1}/{r.n} ({100*r.frac:.2f}%)" for r in nz_r.itertuples())
        L.append(f"Residual sides: nonzero cells grid-wide: {cells}. "
                 f"All remaining residual cells are zero.")
    else:
        L.append("Residual sides: zero xi>=1 windows across all filters.")
    L += ["", "Per-asset detail in results/stage4/xi_incidence.csv.", ""]
    (ROOT / "docs" / "stage5_xi_incidence.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("xi_incidence")
