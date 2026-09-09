"""Generator for caviar_divergence_window.csv (v2.0 P0-3 regression fixture).

Provenance (machine-sourced, rule 11): dgs5 rolling fit windows 756 (ctx_fit)
and 757 (ctx_next), ctx_len=1000, oos_start from results/stage4/run_manifest.yaml,
allow_late_start=True — the refit window whose pre-v2.0 unconstrained CAViaR-SAV
fit at alpha=5% produced b1 = -1.01434 (|b1| > 1, explosive recursion) and, rolled
onto the next window, the delivered grid's divergent path (stored v05 min
-5,729,758.7 in results/stage4/baselines/caviar_sav_dgs5.parquet, pre-v2.0).
Vendored so the regression test needs no data/ access (same pattern as the
mcneilfrey_* fixtures). Requires the local data store to regenerate.
"""

import pandas as pd
import yaml

from data.load import load_series
from harness.rolling import rolling_windows

if __name__ == "__main__":
    mf = yaml.safe_load(open("results/stage4/run_manifest.yaml"))
    df = load_series("dgs5")
    rs = rolling_windows(df, value_col=df.attrs.get("column", "logret"),
                         ctx_len=1000, oos_start=mf["protocol"]["oos_start"],
                         allow_late_start=True)
    pd.DataFrame({"ctx_fit": rs.windows[756].ctx,
                  "ctx_next": rs.windows[757].ctx}
                 ).to_csv("tests/fixtures/caviar_divergence_window.csv", index=False)
    print("caviar_divergence_window.csv written")
