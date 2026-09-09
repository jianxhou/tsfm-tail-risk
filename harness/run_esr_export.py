"""Export per-cell (y, v, e) series for the vendored ESR engine (design condition 8, decision ②).

    python -u -m harness.run_esr_export

Cells: TSFM x {e1,e2,e3} x 3 alpha x 32 assets + ES-bearing baselines x 3 alpha
x 32 assets — the same series the FZ0/MCS comparison uses (reuses those tested
builders). Each cell -> results/stage4/esr_inputs/<cell_id>.csv with rows where
the ES convention holds (e < v < 0); cells with <100 valid rows are skipped and
logged. Index written to esr_inputs/index.csv for the R engine.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from data.load import load_series
from harness.run_grid_compare import (BASELINES, TSFM_TAUS, baseline_ve_series,
                                      tsfm_ve_series)

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"
OUT = S4 / "esr_inputs"
ALPHAS = (0.01, 0.025, 0.05)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mf = yaml.safe_load((S4 / "run_manifest.yaml").read_text())
    assets = mf["assets"]
    index, skipped = [], 0

    for asset in assets:
        fits = pd.read_parquet(S4 / "ctxfits" / f"{asset}.parquet").set_index("t")
        df = load_series(asset)
        x = df[df.attrs.get("column", "logret")].to_numpy()
        for a in ALPHAS:
            cells = []
            for m in TSFM_TAUS:
                for er in ("e1", "e2", "e3"):
                    s = tsfm_ve_series(m, asset, er, a, fits, x)
                    if s is not None:
                        cells.append((f"{m}.{er}", s))
            for bl in BASELINES:
                s = baseline_ve_series(bl, asset, a)
                if s is not None:
                    cells.append((f"{bl}.param", s))
            for name, s in cells:
                s = s.dropna(subset=["v", "e"])
                s = s[(s["e"] < s["v"]) & (s["e"] < 0)]
                if len(s) < 100:
                    skipped += 1
                    continue
                cid = f"{name}.{asset}.a{f'{a:g}'.replace('0.', '')}"
                s[["y", "v", "e"]].to_csv(OUT / f"{cid}.csv", index=False)
                index.append({"cell_id": cid, "forecaster": name.split(".")[0],
                              "erule": name.split(".")[1], "asset": asset,
                              "alpha": a, "n": len(s)})
        print(f"[esr-export] {asset} done ({len(index)} cells so far)", flush=True)

    pd.DataFrame(index).to_csv(OUT / "index.csv", index=False)
    print(f"esr export complete: {len(index)} cells, {skipped} skipped(<100 rows)",
          flush=True)


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("esr_export")
