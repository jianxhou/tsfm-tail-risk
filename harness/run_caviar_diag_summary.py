"""Aggregate the CAViaR-SAV per-asset fit diagnostics into
results/stage4/caviar_diag_summary.csv.

    python -m harness.run_caviar_diag_summary

Reads the per-asset sidecars results/stage4/baselines/caviar_sav_*.meta.json
written by the constrained CAViaR baseline run (v2.0 P0-3) and emits one row
per asset: n_windows, failed_fits / invalid_days summed over the three alpha
levels, and the minimum valid-start count across refits. Added v2.1.1
(review #11 P2-3): the shipped CSV previously had no generator on record —
this script reproduces it byte-for-byte from the existing sidecars (verified
at introduction; no CAViaR rerun involved).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S4 = ROOT / "results" / "stage4"
OUT = S4 / "caviar_diag_summary.csv"


def main() -> None:
    rows = []
    for p in sorted((S4 / "baselines").glob("caviar_sav_*.meta.json")):
        m = json.loads(p.read_text())
        assert m["baseline"] == "caviar_sav", p
        rows.append((m["asset"], int(m["n_windows"]),
                     sum(int(v) for v in m["caviar_failed_fits"].values()),
                     sum(int(v) for v in m["caviar_invalid_days"].values()),
                     int(m["caviar_valid_starts_min"])))
    rows.sort()
    lines = ["asset,n_windows,failed_fits,invalid_days,valid_starts_min"]
    lines += [",".join(str(v) for v in r) for r in rows]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(rows)} assets)")


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("caviar_diag")
