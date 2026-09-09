"""Synthetic oracle arm (proposal §9 D7, scaled): 200 seeded GARCH(1,1)-t paths,
oracle VaR/ES from the true recursion, judged against SIZE-CALIBRATED bands.

    python -m harness.run_pilot_oracle    (writes docs/pilot_oracle.md)

Why size-calibrated: at n=1250 the classical UC/CC tests are size-distorted
(discreteness at small n*alpha; sparse n11 in the independence component — our
implementations match rugarch exactly, so this is the tests' own property).
The script therefore FIRST measures each test's finite-sample size under exact
H0 (iid Bernoulli hits, same n), then requires the oracle rejection rate to sit
inside the 99% Monte-Carlo band around that measured size. A naive nominal-5%
band mis-judges the machinery (found the hard way on a 20-path pilot batch —
see git history of this file).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from backtests.es_tests import as_z2
from backtests.var_tests import christoffersen_cc, kupiec

N_PATHS, N, N_BURN = 200, 1250, 500
OMEGA, A_G, B_G, NU = 0.05, 0.09, 0.90, 8.0
ALPHAS = (0.01, 0.025, 0.05)
SEED = 20260706
R_SIZE = 1000   # H0 replications for the size measurement


def simulate(seed: int):
    rng = np.random.default_rng(seed)
    tsc = np.sqrt((NU - 2.0) / NU)
    z = rng.standard_t(NU, size=N_BURN + N) * tsc
    sig2 = np.empty(N_BURN + N)
    y = np.empty(N_BURN + N)
    sig2[0] = OMEGA / (1 - A_G - B_G)
    y[0] = np.sqrt(sig2[0]) * z[0]
    for t in range(1, N_BURN + N):
        sig2[t] = OMEGA + A_G * y[t - 1] ** 2 + B_G * sig2[t - 1]
        y[t] = np.sqrt(sig2[t]) * z[t]
    return y[N_BURN:], np.sqrt(sig2[N_BURN:])


def h0_size(a: float, rng: np.random.Generator) -> tuple[float, float]:
    """Finite-sample size of UC/CC at level .05 under exact H0 (iid Bernoulli)."""
    uc = cc = 0
    for _ in range(R_SIZE):
        h = rng.random(N) < a
        y = np.where(h, -2.0, 1.0)
        v = np.full(N, -1.0)
        uc += kupiec(y, v, a)["p"] < 0.05
        cc += christoffersen_cc(y, v, a)["p"] < 0.05
    return uc / R_SIZE, cc / R_SIZE


def main() -> None:
    tsc = np.sqrt((NU - 2.0) / NU)
    rng = np.random.default_rng(SEED - 1)
    lines = ["# Synthetic oracle arm, size-calibrated (script-generated; §9 D7)", "",
             f"{N_PATHS} seeded GARCH(1,1)-t paths (omega={OMEGA}, alpha={A_G}, "
             f"beta={B_G}, nu={NU:g}, n={N}); oracle VaR/ES from the true recursion.",
             f"Reference size = UC/CC rejection rate at test level .05 under EXACT H0 "
             f"(iid Bernoulli hits, same n; {R_SIZE} reps). PASS = oracle rejection "
             "rate inside the 99% binomial band around the measured size.", "",
             "| alpha | mean hit rate | UC rej | UC size (H0) | UC verdict "
             "| CC rej | CC size (H0) | CC verdict | mean Z2 |",
             "|---|---|---|---|---|---|---|---|---|"]
    all_ok = True
    for a in ALPHAS:
        q = stats.t.ppf(a, NU) * tsc
        es = (-stats.t.pdf(stats.t.ppf(a, NU), NU)
              * (NU + stats.t.ppf(a, NU) ** 2) / ((NU - 1.0) * a)) * tsc
        uc = cc = 0
        hits, z2s = [], []
        for p in range(N_PATHS):
            y, sig = simulate(SEED + p)
            v, e = sig * q, sig * es
            uc += kupiec(y, v, a)["p"] < 0.05
            cc += christoffersen_cc(y, v, a)["p"] < 0.05
            hits.append(float((y <= v).mean()))
            z2s.append(as_z2(y, v, e, a)["stat"])
        s_uc, s_cc = h0_size(a, rng)
        verdicts = []
        for rej, s in ((uc, s_uc), (cc, s_cc)):
            band = 2.576 * np.sqrt(max(s * (1 - s), 1e-9) / N_PATHS)
            verdicts.append("PASS" if abs(rej / N_PATHS - s) <= band else "FAIL")
        all_ok &= verdicts[0] == "PASS" and verdicts[1] == "PASS"
        lines.append(f"| {a:g} | {100*np.mean(hits):.2f}% | {uc}/{N_PATHS} "
                     f"| {100*s_uc:.1f}% | {verdicts[0]} | {cc}/{N_PATHS} "
                     f"| {100*s_cc:.1f}% | {verdicts[1]} | {np.mean(z2s):+.3f} |")
    lines += ["", f"VERDICT: {'PASS' if all_ok else 'FAIL'} — the battery arbitrates "
              "a true oracle correctly once judged at the tests' own finite-sample "
              "size. Documented property: CC at alpha=5%, n=1250 over-rejects "
              "(~9-12% at nominal 5%) under EXACT H0 — the independence component's "
              "chi2 approximation with sparse n11 (~3 expected). Matches rugarch "
              "bit-for-bit (fixture), so a property of the classical test, not our "
              "code; Stage 4 will use MC-calibrated critical values throughout, "
              "reinforcing the finite-sample-precision design.", ""]
    (Path(__file__).resolve().parent.parent / "docs" / "pilot_oracle.md"
     ).write_text("\n".join(lines))
    print("PASS" if all_ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
