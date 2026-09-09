"""Fetch → QC → parquet pipeline.

Usage:
    python -m data.build --group equity_index [--group fx ...]
    python -m data.build --all

Per series:   data/parquet/<id>.parquet   (date, close, logret | dbp)
              data/raw/<id>.json|csv      (verbatim source response; local-only,
                                           gitignored — sha256 lives in the manifest)
Global:       data/data_manifest.yaml     (provenance, ranges, hashes)
              data/qc_report.md           (QC for the groups built in this run)

Transforms (proposal §3.2; unit separation is deliberate — column names differ):
    logret_x100 (default) -> column `logret` = 100·ln(P_t/P_{t-1});  fails loudly on P<=0
    diff_bp (rates)       -> column `dbp`    = 100·(y_t − y_{t-1})   (yields in percent)

Everything here is regenerated from source on every run; nothing is hand-edited.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.fetch import FETCHERS
from data.qc import PROFILES, qc_series

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "parquet"
RAW_DIR = ROOT / "raw"
MANIFEST = ROOT / "data_manifest.yaml"
REPORT = ROOT / "qc_report.md"
QC_STORE = ROOT / "qc_results.yaml"   # per-group QC persisted so partial rebuilds
                                      # still render the full multi-group report

FETCH_PAUSE_S = 0.7  # be polite to the source APIs


MAX_DIFF_GAP_DAYS = 30   # a first difference spanning a longer calendar gap is not a
                         # daily change (e.g. dgs30 across the 2002-2006 issuance gap)


def logret_x100(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    bad = close[close <= 0]
    if len(bad):
        raise ValueError(f"non-positive prices at {list(bad.index[:5])} — log return undefined")
    return (100.0 * np.log(close / close.shift(1))).iloc[1:]


def diff_bp(df: pd.DataFrame) -> pd.Series:
    d = (100.0 * df["close"].diff()).iloc[1:]
    gap = pd.to_datetime(df["date"]).diff().dt.days.iloc[1:]
    return d.mask(gap > MAX_DIFF_GAP_DAYS)


def level(df: pd.DataFrame) -> pd.Series:
    """Identity: store the raw close as a level (vol indices for regime dating).
    Drops the first row to match the returns-transform row alignment."""
    return df["close"].iloc[1:]


TRANSFORMS = {"logret_x100": (logret_x100, "logret"), "diff_bp": (diff_bp, "dbp"),
              "level": (level, "level")}


def build_groups(groups: list[str]) -> None:
    universe = yaml.safe_load((ROOT / "universe.yaml").read_text())
    unknown = [g for g in groups if g not in universe]
    if unknown:
        raise SystemExit(f"groups not in universe.yaml: {unknown}")

    PARQUET_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    manifest = yaml.safe_load(MANIFEST.read_text()) or {} if MANIFEST.exists() else {}
    qc_by_group: dict[str, list[dict]] = (
        yaml.safe_load(QC_STORE.read_text()) or {} if QC_STORE.exists() else {}
    )
    qc_by_group = {g: v for g, v in qc_by_group.items() if g in universe}

    for group in groups:
        specs = universe[group]
        profile = PROFILES[group]
        qc_all: list[dict] = []

        for spec in specs:
            sid, symbol, source = spec["id"], spec["symbol"], spec["source"]
            transform_name = spec.get("transform", "logret_x100")
            transform, col = TRANSFORMS[transform_name]

            print(f"[{sid}] fetching {symbol} from {source} ...", flush=True)
            res = FETCHERS[source](symbol)
            time.sleep(FETCH_PAUSE_S)

            raw_path = RAW_DIR / f"{sid}.{res.raw_format}"
            raw_path.write_bytes(res.raw)

            ret = transform(res.df)
            out = res.df.iloc[1:].copy()
            out[col] = ret.values

            pq_path = PARQUET_DIR / f"{sid}.parquet"
            out.to_parquet(pq_path, index=False)

            qc = qc_series(sid, res.df, ret, res.timezone, res.splits, res.dividends, profile)
            qc_all.append(qc)

            manifest[sid] = {
                "name": spec["name"],
                "group": group,
                "symbol": symbol,
                "source": source,
                "url": res.url,
                "timezone": res.timezone,
                "currency": res.currency,
                "price_field": res.price_field,
                "transform": transform_name,
                "column": col,
                "unit": profile["unit"],
                "calendar": spec.get("calendar", "exchange"),
                "fetched_at": res.fetched_at,
                "rows": len(out),
                "start": qc["start"],
                "end": qc["end"],
                "parquet": str(pq_path.relative_to(ROOT.parent)),
                "sha256": hashlib.sha256(pq_path.read_bytes()).hexdigest(),
                "raw_snapshot": str(raw_path.relative_to(ROOT.parent)),
                "raw_sha256": hashlib.sha256(res.raw).hexdigest(),
                **({"note": spec["note"]} if spec.get("note") else {}),
                **({"quarantined": True,
                    "quarantine_reason": spec.get("quarantine_reason", "unspecified")}
                   if spec.get("quarantined") else {}),
            }
            print(f"[{sid}] {qc['n_obs']} bars {qc['start']} → {qc['end']}"
                  f"  flags: {qc['flags'] or 'none'}", flush=True)

        # drop manifest entries (and their files) for series removed from this group
        current = {s["id"] for s in specs}
        stale = [k for k, v in manifest.items() if v.get("group") == group and k not in current]
        for k in stale:
            (PARQUET_DIR / f"{k}.parquet").unlink(missing_ok=True)
            for ext in ("json", "csv"):
                (RAW_DIR / f"{k}.{ext}").unlink(missing_ok=True)
            del manifest[k]
            print(f"[{k}] removed (no longer in universe group '{group}')", flush=True)

        qc_by_group[group] = qc_all

    MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=True, allow_unicode=True))
    QC_STORE.write_text(yaml.safe_dump(_plain(qc_by_group), sort_keys=False,
                                       allow_unicode=True))
    _write_report(qc_by_group)
    n = sum(len(v) for v in qc_by_group.values())
    print(f"\nwrote {MANIFEST.name}, {QC_STORE.name}, {REPORT.name} ({n} series in report)")


def _plain(obj):
    """yaml-safe: numpy scalars/tuples → native python."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _write_report(qc_by_group: dict[str, list[dict]]) -> None:
    lines = [
        "# QC report",
        "",
        f"Groups: {', '.join(qc_by_group)}. Generated by `python -m data.build`. Do not hand-edit.",
        "Raw source snapshots live in `data/raw/` (local-only); their sha256 is pinned in the manifest.",
        "",
    ]
    for group, qc_all in qc_by_group.items():
        unit = qc_all[0]["unit"] if qc_all else ""
        lines += [
            f"## {group}  (unit: {unit})",
            "",
            "| id | n_obs | start | end | bars/yr | σ | skew | ex.kurt | zero | masked | stale | gaps | tz | flags |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for f in qc_all:
            lines.append(
                f"| {f['id']} | {f['n_obs']} | {f['start']} | {f['end']} | {f['bars_per_year']} "
                f"| {f['ret_std']:.2f} | {f['ret_skew']:.2f} | {f['ret_kurt']:.1f} "
                f"| {f['zero_ret']} | {f.get('masked', 0)} | {f['max_stale_run']} | {len(f['gaps'])} "
                f"| {f['timezone']} | {'; '.join(f['flags']) or '—'} |"
            )
        lines += ["", f"**Largest |moves| ({unit}) — verify vs known events:**", ""]
        for f in qc_all:
            moves = ", ".join(f"{d}: {r:+.2f}" for d, r in f["top_moves"])
            lines.append(f"- **{f['id']}**: {moves}")
        gap_lines = []
        for f in qc_all:
            if f["gaps"]:
                gap_lines.append("- **" + f["id"] + "**: "
                                 + "; ".join(f"{a} → {b} ({g}d)" for a, b, g in f["gaps"]))
        if gap_lines:
            lines += ["", "**Calendar gaps:**", ""] + gap_lines
        lines.append("")

    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        universe = yaml.safe_load((ROOT / "universe.yaml").read_text())
        build_groups(list(universe.keys()))
    elif args.group:
        build_groups(args.group)
    else:
        raise SystemExit("specify --group <name> (repeatable) or --all")
