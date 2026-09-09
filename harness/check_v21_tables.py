"""v2.1 Phase-2 independent table spot-check (review #10 §2.7 verification).

    python -m harness.check_v21_tables

Independent (read-only, no make_tables imports) recomputation of:
  * Table 2  (tab_var_audit)    — every E0/E1/E2/E3/param percentage from
                                  backtests.csv (pooled 96-cell UC share);
  * Table 8  (tab_success)      — (a)/(b)/joint per (head, arm) from
                                  repair_backtests_matched.csv under the
                                  registered criterion;
  * Table D.2 (tab_repairmatrix)— per-(head, arm) matched mean FZ0 + n
                                  markers from the same CSV;
  * tab_decoding invariance     — the decoding audit is E3-independent: the
                                  generated file must be byte-identical to
                                  HEAD (git diff empty).
Every parsed generated value must equal the independent recomputation at
its printed rounding. Exit 1 on any mismatch.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper/draft/generated"
S4 = ROOT / "results/stage4"
RT = dict(float_precision="round_trip")
HEADS = {"Chronos-Bolt": "chronos_bolt", "Chronos-2": "chronos_2",
         "TimesFM-2.5": "timesfm_2_5", "Moirai-2.0": "moirai_2_0",
         "Lag-Llama": "lag_llama"}
BASE = {"GARCH(1,1)-$t$": "garch_t", "GJR-GARCH-$t$": "gjr_t",
        "EWMA(0.94)": "ewma94", "HS-250": "hs250", "HS-500": "hs500",
        "FHS": "fhs", "CAViaR-SAV": "caviar_sav"}
ARMS = ["unrepaired", "F1", "F2", "F3", "F4", "H"]


def fail(msgs, m):
    msgs.append(m)
    print("MISMATCH:", m)


def check_var_audit(msgs):
    bt = pd.read_csv(S4 / "backtests.csv", **RT)
    txt = (GEN / "tab_var_audit.tex").read_text()
    share = lambda d: int(round(100 * (d.kupiec_p_mc > 0.05).mean()))
    parsed = 0
    for disp, key in HEADS.items():
        mrow = re.search(re.escape(disp) + r" & (.+?) \\\\", txt)
        if not mrow:
            fail(msgs, f"varaudit {disp}: row not parsed from tex")
            continue
        row = mrow.group(1)
        parsed += 1
        cells = [c.strip() for c in row.split("&")]
        for er, cell in zip(("e0", "e1", "e2", "e3"), cells[:4]):
            d = bt[(bt.forecaster == key) & (bt.erule == er)]
            if cell.startswith("---"):
                if len(d) != 0:
                    fail(msgs, f"varaudit {disp}/{er}: dash but {len(d)} rows")
                continue
            want = int(re.match(r"(\d+)\\%", cell).group(1))
            got = share(d)
            if want != got or len(d) != 96:
                fail(msgs, f"varaudit {disp}/{er}: tex {want}% vs recomputed "
                           f"{got}% (n={len(d)})")
    bl = pd.read_csv(S4 / "baseline_backtests.csv", **RT)
    for disp, key in BASE.items():
        mrow = re.search(re.escape(disp) + r" & (.+?) \\\\", txt)
        if not mrow:
            fail(msgs, f"varaudit {disp}: baseline row not parsed from tex")
            continue
        parsed += 1
        cell = [c.strip() for c in mrow.group(1).split("&")][4]
        want = int(re.match(r"(\d+)\\%", cell).group(1))
        d = bl[(bl.forecaster == key)]
        got = share(d)
        if want != got or len(d) != 96:
            fail(msgs, f"varaudit {disp}/param: tex {want}% vs {got}% (n={len(d)})")
    # v2.1 Phase 3 (adjudication §8 ISSUE ii): a parse leg that matches zero
    # rows is a silent-pass hazard — hard-fail instead.
    if parsed == 0:
        fail(msgs, "varaudit: ZERO rows parsed from tex — dead parse leg")
    print(f"Table 2 (tab_var_audit): all parsed cells checked ({parsed} rows)")


def _matched():
    m = pd.read_csv(S4 / "repair_backtests_matched.csv", **RT)
    return m[m.alpha == 0.01]


def _criterion(m, head):
    """(a) pass rate vs FHS-10pp; (b) FZ0 <= 1.05*min(gjr_t, fhs) per asset,
    NaN fails; denominator 32. References taken on the head's OWN matched
    day set (context_model column)."""
    fhs = m[(m.forecaster == "fhs") &
            (m.context_model == head)].set_index("asset")
    gjr = m[(m.forecaster == "gjr_t") &
            (m.context_model == head)].set_index("asset")
    ref_pass = ((fhs.kupiec_p_mc > 0.05) & (fhs.cc_p_mc > 0.05)).mean()
    out = {}
    for arm in ARMS:
        d = m[(m.forecaster == f"{head}+{arm}")
              & (m.erule == "e3")].set_index("asset")   # criterion = E3 primary
        if d.empty:
            continue
        pr = float(((d.kupiec_p_mc > 0.05) & (d.cc_p_mc > 0.05)).mean())
        okb = 0
        for a in fhs.index:
            ref = 1.05 * min(gjr.loc[a, "fz0_mean"], fhs.loc[a, "fz0_mean"])
            v = d.loc[a, "fz0_mean"] if a in d.index else np.nan
            okb += int(np.isfinite(v) and v <= ref)
        out[arm] = (pr, (pr >= ref_pass - 0.10), okb / 32,
                    (pr >= ref_pass - 0.10) and (okb / 32 >= 2 / 3))
    return out, float(ref_pass)


def check_success(msgs):
    m = _matched()
    txt = (GEN / "tab_success.tex").read_text()
    heads_seen = parsed = 0
    for disp, key in HEADS.items():
        crit, _ = _criterion(m, key)
        if "H" not in crit:
            continue
        heads_seen += 1
        # each body line, generated format (v2.1 Phase 3, adjudication §8
        # ISSUE ii — the old (yes|no) pattern matched zero rows):
        #   "Chronos-2 & F1 & 78\% & pass & 0.62 & fails \\"
        #   "Chronos-2 & H & 69\% & pass & 0.84 & \textbf{succeeds} \\"
        for arm, (pr, a_ok, bfrac, joint) in crit.items():
            pat = re.compile(re.escape(disp) + r" & " + arm +
                             r" & (\d+)\\% & (pass|fail) & (\d\.\d{2}) & "
                             r"(?:\\textbf\{)?(succeeds|fails)")
            mm = pat.search(txt)
            if not mm:
                continue                      # table prints a subset of arms
            parsed += 1
            t_pr, t_a, t_b, t_j = mm.groups()
            if (int(t_pr) != int(round(100 * pr))
                    or (t_a == "pass") != a_ok
                    or abs(float(t_b) - bfrac) > 0.006
                    or (t_j == "succeeds") != joint):
                fail(msgs, f"success {disp}+{arm}: tex ({t_pr}%,{t_a},{t_b},"
                           f"{t_j}) vs ({round(100*pr)}%,{a_ok},{bfrac:.2f},{joint})")
    if parsed == 0:
        fail(msgs, "success: ZERO rows parsed from tex — dead parse leg")
    hj = [disp for disp, key in HEADS.items()
          if _criterion(m, key)[0].get("H", (0, 0, 0, False))[3]]
    print(f"Table 8 (tab_success): {heads_seen} heads, {parsed} rows parsed; "
          f"H joint-succeeds on {len(hj)}/5 heads (independent recompute): {hj}")
    if len(hj) != 5:
        fail(msgs, f"H joint success {len(hj)}/5 != 5/5")


def check_repairmatrix(msgs):
    m = _matched()
    txt = (GEN / "tab_repairmatrix.tex").read_text()
    n_checked = n_delta = 0
    for disp, key in HEADS.items():
        base = m[(m.forecaster == f"{key}+unrepaired")
                 & (m.erule == "e3")].set_index("asset").fz0_mean
        for arm in ARMS:
            d = m[(m.forecaster == f"{key}+{arm}") & (m.erule == "e3")]
            v = d.fz0_mean.dropna()
            if not len(v):
                continue
            mean, k = v.mean(), len(v)
            # v2.1 Phase-3 cell form: every FZ0 mean prints n — "1.786 (30)"
            for mm in re.finditer(r"(\d+\.\d{3}) \((\d+)\)", txt):
                if abs(float(mm.group(1)) - mean) < 5e-4 \
                        and int(mm.group(2)) == k:
                    n_checked += 1
                    break
            else:
                fail(msgs, f"repairmatrix {disp}+{arm}: mean {mean:.3f} "
                           f"(n={k}) not found in tex")
            if arm != "unrepaired":
                # paired-difference column: mean(arm − unrepaired), common set
                c = pd.concat([base, d.set_index("asset").fz0_mean],
                              axis=1, keys=["u", "a"]).dropna()
                dv, dk = float((c["a"] - c["u"]).mean()), len(c)
                for mm in re.finditer(r"\$([+-]\d+\.\d{3})\$ \((\d+)\)", txt):
                    if abs(float(mm.group(1)) - dv) < 5e-4 \
                            and int(mm.group(2)) == dk:
                        n_delta += 1
                        break
                else:
                    fail(msgs, f"repairmatrix {disp}+{arm}: delta {dv:+.3f} "
                               f"({dk}) not found in tex")
    print(f"Table D.2 (tab_repairmatrix): {n_checked} (head,arm) means + "
          f"{n_delta} paired differences located")
    if n_checked == 0 or n_delta == 0:
        fail(msgs, "repairmatrix: ZERO cells parsed from tex — dead parse leg")
    elif n_checked < 20:
        fail(msgs, f"repairmatrix: only {n_checked} cells matched")


def check_decoding_invariance(msgs):
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat",
                        "paper/draft/generated/tab_decoding.tex"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        fail(msgs, f"tab_decoding regenerated with CHANGES (must be "
                   f"E3-invariant):\n{r.stdout}")
    else:
        print("Table 6 (tab_decoding): byte-identical after regeneration "
              "(E3-independent, as required)")


def main() -> int:
    msgs: list[str] = []
    check_var_audit(msgs)
    check_success(msgs)
    check_repairmatrix(msgs)
    check_decoding_invariance(msgs)
    print("TABLE SPOT-CHECK:", "PASS" if not msgs else f"FAIL ({len(msgs)})")
    return 1 if msgs else 0


if __name__ == "__main__":
    sys.exit(main())
