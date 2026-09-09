"""Stage-5 regime event-time analysis — REBUILT per closeout rulings §1c/§1d
(docs/stage5_closeout_rulings.md, pre-registered at commit 95d9b0a before any
rebuilt-band computation; supersedes the voided self-band version).

    PYTHONUNBUFFERED=1 python -u -m harness.run_regime_analysis

Statistic: null band = distribution of the trailing-window pooled violation
rate UNDER CORRECT CALIBRATION (center alpha), dependence calibrated by
day-cluster bootstrap of CALM-stratum violation indicators (frozen
regimes/rules.py::calm_set, 70c0729); tau_rec = first re-entry from above;
never-elevated is its own category, never reported as recovery. Power
disclosure mde_mult = band/alpha per cell; MOVE/crypto pools (k=4)
direction-only a priori.

Layers (design §B, compliance closed): unrepaired E3 per TSFM (headline),
GJR-t + FHS baselines, F1-repaired overlay. Hit series come from the persisted
per-day series (results/stage4/repairs/, results/stage4/baselines/) — single
source, no recomputation. alpha in {2.5%, 5%} inferential, 1% descriptive-only.
The old self-resampling band is retained SOLELY as the design-§B curve-display
uncertainty band (disp_lo/disp_hi in the curves file).

Outputs: results/stage4/regime_taurec_v2.csv, regime_curves.csv,
docs/stage5_regime.md (regenerated).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from regimes.rules import CLASS_INDEX, calm_set, episode_set

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"
REP = S4 / "repairs"
BL = S4 / "baselines"
SPAN = 60
ALPHAS = (0.025, 0.05, 0.01)            # 0.01 descriptive-only (design §B)
DESC_ONLY = {0.01}
WINDOWS = (21, 10)
B = 2000
SEED = 20260706
TSFM = ["chronos_bolt", "chronos_2", "timesfm_2_5", "moirai_2_0", "lag_llama"]
SMALL_POOLS = {"move", "crypto_rv"}      # k=4 -> direction-only a priori (§1d)
CUTOFF = {"chronos_bolt": "2025-11-21", "chronos_2": "2025-10-31",
          "timesfm_2_5": "2023-11-30", "moirai_2_0": "2025-11-30",
          "lag_llama": "2023-10-31"}


def acol(a):
    return f"{a:g}".replace("0.", "")


def hit_series(layer: str, asset: str, alpha: float) -> pd.Series | None:
    """{date -> 1{y<=v}} for one layer; NaN-v days dropped."""
    c = acol(alpha)
    if layer in ("gjr_t", "fhs"):
        p = BL / f"{layer}_{asset}.parquet"
        if not p.exists():
            return None
        d = pd.read_parquet(p)
        v = d[f"v{c}"].to_numpy()
    else:
        model, kind = layer.split(":")
        p = REP / f"{model}_{asset}.parquet"
        if not p.exists():
            return None
        d = pd.read_parquet(p)
        v = d[("u_e3_" if kind == "e3" else "f1_") + f"v{c}"].to_numpy()
    ok = np.isfinite(v)
    return pd.Series((d["y"].to_numpy()[ok] <= v[ok]).astype(float),
                     index=pd.DatetimeIndex(pd.to_datetime(d["date"]))[ok])


def event_matrix(hit_by_asset: dict, onset: pd.Timestamp):
    """(SPAN, k) hit matrix, day 0 = onset; assets need full-span coverage.
    Guard: the segment must actually START at the onset (within 7 calendar
    days) — a series whose coverage begins after the onset (e.g. the F1
    overlay's warm-up head) is dropped, not silently shifted."""
    cols, names = [], []
    for a, s in hit_by_asset.items():
        pos = s.index.searchsorted(onset)
        if pos >= len(s) or (s.index[pos] - onset).days > 7:
            continue
        seg = s.iloc[pos:pos + SPAN].to_numpy()
        if len(seg) == SPAN and np.isfinite(seg).all():
            cols.append(seg)
            names.append(a)
    return (np.column_stack(cols) if cols else np.empty((SPAN, 0))), names


def calm_matrix(index_id: str, hit_by_asset: dict, names: list[str]):
    """Calm-day violation matrix over the episode's own cross-section."""
    runs = calm_set()
    runs = runs[runs["index"] == index_id]
    dates = None
    for a in names:
        idx = hit_by_asset[a].index
        keep = np.zeros(len(idx), dtype=bool)
        for _, r in runs.iterrows():
            keep |= (idx >= pd.Timestamp(r["start"])) & (idx <= pd.Timestamp(r["end"]))
        da = set(idx[keep])
        dates = da if dates is None else (dates & da)
    dates = sorted(dates) if dates else []
    if not dates:
        return np.empty((0, len(names)))
    return np.column_stack([hit_by_asset[a].loc[dates].to_numpy() for a in names])


def null_band(C: np.ndarray, alpha: float, win: int, rng: np.random.Generator):
    """Rulings §1c: calm day-cluster bootstrap recentered to nominal alpha.
    Returns (band, dep_uncalibrated)."""
    n_calm, k = C.shape
    if n_calm == 0 or C.sum() < 5:      # degenerate guard -> independence MC
        draws = rng.binomial(win * k, alpha, size=B) / (win * k)
        return float(np.quantile(draws, 0.95)), True
    p_hat = C.mean()
    r_b = C[rng.integers(0, n_calm, (B, win))].mean(axis=(1, 2))
    scale = np.sqrt(alpha * (1 - alpha) / (p_hat * (1 - p_hat)))
    r_star = np.clip(alpha + (r_b - p_hat) * scale, 0.0, None)
    return float(np.quantile(r_star, 0.95)), False


def trailing_rate(M: np.ndarray, win: int) -> np.ndarray:
    T = M.shape[0]
    rate = np.full(T, np.nan)
    for t in range(win - 1, T):
        rate[t] = M[t - win + 1:t + 1].mean()
    return rate


def disp_band(M: np.ndarray, win: int, rng: np.random.Generator):
    """Design-§B curve DISPLAY band (90% two-sided self-bootstrap) — display
    only, explicitly not the re-entry reference (rulings §1a)."""
    T = M.shape[0]
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    for t in range(win - 1, T):
        block = M[t - win + 1:t + 1]
        draws = block[rng.integers(0, win, (B, win))].mean(axis=(1, 2))
        lo[t], hi[t] = np.quantile(draws, [0.05, 0.95])
    return lo, hi


def tau_rec(rate: np.ndarray, band: float, win: int):
    """(tau, ever_above, censored): first re-entry from above (1-indexed)."""
    first_above = None
    for t in range(win - 1, SPAN):
        if not np.isfinite(rate[t]):
            continue
        if first_above is None:
            if rate[t] > band:
                first_above = t
        elif rate[t] <= band:
            return t + 1, True, False
    if first_above is None:
        return np.nan, False, False
    return 61, True, True


def main():
    mf = yaml.safe_load((S4 / "run_manifest.yaml").read_text())
    oos = mf["protocol"]["oos_start"]
    dmani = yaml.safe_load((ROOT / "data" / "data_manifest.yaml").read_text())
    asset_group = {a: dmani[a]["group"] for a in dmani
                   if not dmani[a].get("quarantined")}
    eps = episode_set()
    eps = eps[eps.onset >= oos].reset_index(drop=True)
    layers = ([f"{m}:e3" for m in TSFM] + ["gjr_t", "fhs"]
              + [f"{m}:f1" for m in TSFM])

    tau_rows, curve_rows = [], []
    for _, ep in eps.iterrows():
        idx = ep["index"]
        onset = pd.Timestamp(ep["onset"])
        assets = [a for a, g in asset_group.items() if CLASS_INDEX.get(g) == idx]
        for layer in layers:
            for a in ALPHAS:
                hb = {}
                for asset in assets:
                    s = hit_series(layer, asset, a)
                    if s is not None and len(s) > 500:
                        hb[asset] = s
                M, names = event_matrix(hb, onset)
                if M.shape[1] < 2:
                    continue
                C = calm_matrix(idx, hb, names)
                rng = np.random.default_rng(SEED)
                daily = M.mean(axis=1)
                trails = {w: trailing_rate(M, w) for w in WINDOWS}
                bands, flags, taus = {}, {}, {}
                for w in WINDOWS:
                    bands[w], flags[w] = null_band(C, a, w, rng)
                    taus[w] = tau_rec(trails[w], bands[w], w)
                lo21, hi21 = disp_band(M, 21, rng)
                model = layer.split(":")[0]
                post = (onset > pd.Timestamp(CUTOFF[model])
                        if model in CUTOFF else False)
                for w in WINDOWS:
                    tr, ever, cen = taus[w]
                    tau_rows.append({
                        "episode": ep["episode"], "index": idx,
                        "onset": ep["onset"], "layer": layer, "alpha": a,
                        "window": w, "n_assets": M.shape[1],
                        "n_calm_days": C.shape[0], "band": round(bands[w], 5),
                        "mde_mult": round(bands[w] / a, 2),
                        "dep_uncalibrated": flags[w],
                        "direction_only": (idx in SMALL_POOLS
                                           or (w == 10 and a == 0.025)),
                        "descriptive_only": a in DESC_ONLY,
                        "tau_rec": tr, "ever_above": ever, "censored": cen,
                        "onset_rate": round(float(M[0].mean()), 3),
                        "post_cutoff": post})
                for t in range(SPAN):
                    curve_rows.append({
                        "episode": ep["episode"], "layer": layer, "alpha": a,
                        "t": t + 1, "daily_rate": round(float(daily[t]), 4),
                        "trail21": round(float(trails[21][t]), 4),
                        "trail10": round(float(trails[10][t]), 4),
                        "band21": round(bands[21], 5),
                        "band10": round(bands[10], 5),
                        "disp_lo21": round(float(lo21[t]), 4),
                        "disp_hi21": round(float(hi21[t]), 4)})
        print(f"[regime] {ep['episode']} ({idx}) done", flush=True)

    tau = pd.DataFrame(tau_rows)
    curves = pd.DataFrame(curve_rows)
    tau.to_csv(S4 / "regime_taurec_v2.csv", index=False)
    curves.to_csv(S4 / "regime_curves.csv", index=False)
    _summary(tau, curves)
    print(f"regime analysis complete: {len(tau)} tau rows, "
          f"{len(curves)} curve rows", flush=True)


def _fmt_tau(s: pd.DataFrame) -> str:
    rec = s[s.ever_above & ~s.censored]
    med = f"{rec.tau_rec.median():.0f}" if len(rec) else "—"
    return (f"{med} | {len(rec)}/{int(s.censored.sum())}/"
            f"{int((~s.ever_above).sum())}")


def _summary(tau: pd.DataFrame, curves: pd.DataFrame):
    L = ["# Stage-5 regime event-time analysis v2 (script-generated; rebuilt per",
         "# closeout rulings §1c/§1d @95d9b0a; episode set frozen @70c0729)", "",
         "Null band: calm-stratum day-cluster bootstrap recentered to nominal alpha",
         "(one-sided 95%, B=2000, seed 20260706). tau_rec = first re-entry from",
         "above; cell format = median tau_rec^w | recovered/censored/never-elevated",
         "counts over episodes. Direction-only: MOVE + crypto pools (k=4, a priori)",
         "and tau_rec^10 at alpha=2.5% (pooled n_alpha=8<10). alpha=1% descriptive",
         "only. Curves (incl. the two review episodes) in regime_curves.csv;",
         "display bands there are self-bootstrap (display ONLY, not the test).", ""]
    layers_main = [f"{m}:e3" for m in TSFM] + ["gjr_t", "fhs"]
    for a in (0.025, 0.05):
        L += [f"## alpha={a:g}: tau_rec by layer (VIX pool episodes, inferential)", "",
              "| layer | tau^21 med | rec/cens/never (21) | tau^10 med | rec/cens/never (10) |",
              "|---|---|---|---|---|"]
        for lay in layers_main + [f"{m}:f1" for m in TSFM]:
            s21 = tau[(tau.layer == lay) & (tau.alpha == a) & (tau.window == 21)
                      & (tau["index"] == "vix")]
            s10 = tau[(tau.layer == lay) & (tau.alpha == a) & (tau.window == 10)
                      & (tau["index"] == "vix")]
            if len(s21):
                c21 = _fmt_tau(s21).split(" | ")
                c10 = _fmt_tau(s10).split(" | ")
                L.append(f"| {lay} | {c21[0]} | {c21[1]} | {c10[0]} | {c10[1]} |")
        L.append("")
    L += ["## Small pools (MOVE, crypto — direction-only, k=4)", "",
          "| pool | alpha | layer | tau^21 med | rec/cens/never | mde_mult (med) |",
          "|---|---|---|---|---|---|"]
    for idx in ("move", "crypto_rv"):
        for a in (0.025, 0.05):
            for lay in layers_main:
                s = tau[(tau["index"] == idx) & (tau.alpha == a)
                        & (tau.window == 21) & (tau.layer == lay)]
                if len(s):
                    c = _fmt_tau(s).split(" | ")
                    L.append(f"| {idx} | {a:g} | {lay} | {c[0]} | {c[1]} | "
                             f"{s.mde_mult.median():.2f} |")
    L += ["", "## Power disclosure (rulings §1d): median mde_mult = band/alpha", "",
          "| pool | alpha | win | mde_mult med | dep_uncalibrated cells |",
          "|---|---|---|---|---|"]
    for idx in ("vix", "move", "crypto_rv"):
        for a in (0.025, 0.05):
            for w in WINDOWS:
                s = tau[(tau["index"] == idx) & (tau.alpha == a) & (tau.window == w)]
                if len(s):
                    L.append(f"| {idx} | {a:g} | {w} | {s.mde_mult.median():.2f} | "
                             f"{int(s.dep_uncalibrated.sum())}/{len(s)} |")
    L += ["", "## Review episodes, raw trailing-21 rates (alpha=5%, E3 layers)",
          "", "| episode | layer | t=21 | t=25 | t=30 | t=40 | t=60 |",
          "|---|---|---|---|---|---|---|"]
    for ep in ("vix_ep17", "vix_ep23"):
        for m in TSFM:
            s = curves[(curves.episode == ep) & (curves.layer == f"{m}:e3")
                       & (curves.alpha == 0.05)].set_index("t")
            if len(s):
                vals = " | ".join(f"{s.loc[t, 'trail21']:.3f}"
                                  for t in (21, 25, 30, 40, 60))
                L.append(f"| {ep} | {m} | {vals} |")
    L += ["", "Window disagreements (21 vs 10) go to the discussion as",
          "sensitivity. Contamination: post_cutoff per row (2025-04 clean for",
          "timesfm/lag_llama; COVID/2022 in-corpus for all). Full per-episode",
          "detail: regime_taurec_v2.csv; per-day curves: regime_curves.csv.", ""]
    (ROOT / "docs" / "stage5_regime.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("regime_analysis")
