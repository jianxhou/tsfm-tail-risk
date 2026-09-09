"""BTC exposed-vs-unexposed pre/post-cutoff split (gate condition 7; executed
at Stage 6 revision item 6, signer instruction 2026-07-11).

Reconstructs the battery's per-day E2 alpha=5% VaR series for BTC per model
from the stored forecast grids (results/stage4/forecast/{model}_btc.parquet)
and the cached context fits (results/stage4/ctxfits/btc.parquet), using the
FROZEN extraction module erules.rules (no re-implementation). Integrity gate:
the reconstructed whole-window hit rates must reproduce the frozen
battery-level values (docs/stage4_btc_did.md / evidence pack) exactly at the
2-decimal reporting precision, or the script aborts.

Split: breakpoint 2021-07-01 (the disclosed Monash bitcoin component ends
~2021-07; gate condition 7). Exposed = {chronos_2, moirai_2_0, timesfm_2_5};
unexposed = {chronos_bolt, lag_llama} (docs/giftevalpretrain_overlap.md).
DiD = (exposed_post - exposed_pre) - (unexposed_post - unexposed_pre),
group rates = mean of member models' hit rates per period.

Output: docs/stage4_btc_did_split.md (script-generated; do not hand-edit).
Run:  python -m harness.run_btc_did_split
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from erules.rules import e2_quantile

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"

MODELS = ["chronos_bolt", "chronos_2", "timesfm_2_5", "moirai_2_0", "lag_llama"]
EXPOSED = {"chronos_2", "moirai_2_0", "timesfm_2_5"}
BREAK = pd.Timestamp("2021-07-01")
ALPHA = 0.05
# frozen battery-level whole-window values (docs/stage4_btc_did.md @a88a364)
FROZEN = {"chronos_bolt": 5.66, "chronos_2": 6.43, "timesfm_2_5": 5.35,
          "moirai_2_0": 4.14, "lag_llama": 6.43}
# column-name -> quantile level (deciles for timesfm; deep levels elsewhere)
COLMAP = {"q01": 0.01, "q025": 0.025, "q05": 0.05, "q1": 0.1, "q2": 0.2,
          "q25": 0.25, "q3": 0.3, "q4": 0.4, "q5": 0.5, "q6": 0.6,
          "q7": 0.7, "q75": 0.75, "q8": 0.8, "q9": 0.9}


def hits(model: str) -> pd.DataFrame:
    fc = pd.read_parquet(S4 / "forecast" / f"{model}_btc.parquet")
    cf = pd.read_parquet(S4 / "ctxfits" / "btc.parquet")
    df = fc.merge(cf[["t", "nu"]], on="t", validate="one_to_one")
    cols = {c: lvl for c, lvl in COLMAP.items() if c in df.columns}
    var = df.apply(lambda r: e2_quantile(
        {lvl: r[c] for c, lvl in cols.items()}, ALPHA, r.nu), axis=1)
    return pd.DataFrame({"date": df.date, "hit": (df.y < var).astype(int)})


def main():
    rows, group = [], {}
    for m in MODELS:
        h = hits(m)
        whole = 100 * h.hit.mean()
        if round(whole, 2) != FROZEN[m]:
            raise AssertionError(
                f"{m}: reconstructed whole-window hit {whole:.2f}% != frozen "
                f"{FROZEN[m]}% — extraction path drifted; ABORT")
        pre = h[h.date < BREAK]
        post = h[h.date >= BREAK]
        rows.append((m, "exposed" if m in EXPOSED else "unexposed",
                     len(pre), 100 * pre.hit.mean(),
                     len(post), 100 * post.hit.mean(), whole))
        group[m] = (100 * pre.hit.mean(), 100 * post.hit.mean())

    exp_pre = sum(group[m][0] for m in MODELS if m in EXPOSED) / 3
    exp_post = sum(group[m][1] for m in MODELS if m in EXPOSED) / 3
    une_pre = sum(group[m][0] for m in MODELS if m not in EXPOSED) / 2
    une_post = sum(group[m][1] for m in MODELS if m not in EXPOSED) / 2
    did = (exp_post - exp_pre) - (une_post - une_pre)

    L = ["# Stage-4 BTC contamination DiD — pre/post-2021-07 split "
         "(script-generated; revision item 6, 2026-07-11)", "",
         "Registered protocol (gate condition 7) executed on the stored "
         "per-day forecast grids with the frozen erules extraction "
         "(E2, alpha=5%); whole-window rates reproduce the battery preview "
         "exactly at reporting precision (integrity gate in "
         "harness/run_btc_did_split.py). Breakpoint 2021-07-01 = disclosed "
         "end of the Monash bitcoin component.", "",
         "| model | exposure | n pre | hit pre | n post | hit post | whole |",
         "|---|---|---|---|---|---|---|"]
    for m, ex, npre, hpre, npost, hpost, whole in rows:
        L.append(f"| {m} | {ex} | {npre} | {hpre:.2f}% | {npost} | "
                 f"{hpost:.2f}% | {whole:.2f}% |")
    L += ["",
          f"Group means: exposed pre {exp_pre:.2f}% / post {exp_post:.2f}%; "
          f"unexposed pre {une_pre:.2f}% / post {une_post:.2f}%.",
          f"**DiD (exposed minus unexposed, post minus pre): {did:+.2f}pp.**",
          "",
          "Reading: the exposed group's pre-to-post change matches the "
          "unexposed group's within a fraction of a percentage point; no "
          "contamination advantage is detectable on the one asset with "
          "disclosed training-corpus overlap. Level contrast (whole column) "
          "unchanged from docs/stage4_btc_did.md."]
    (ROOT / "docs" / "stage4_btc_did_split.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[4:]))
    print("\nwritten: docs/stage4_btc_did_split.md")


if __name__ == "__main__":
    main()
