"""Freeze the Stage-4 full-grid run manifest (amendment A: before the main grid
opens). The manifest is the reproducibility + governance anchor: frozen config,
model revisions (from registry), and per-statistic enablement gates.

    python -m harness.freeze_run_manifest        # writes results/stage4/run_manifest.yaml
    python -m harness.freeze_run_manifest --check # verify current config hashes to the frozen one
    python -m harness.freeze_run_manifest --refreeze-variant
        # package-builder mode: recompute the hash fields of an EXISTING
        # manifest from the current tree (used on the distribution copy
        # after commit-token redaction; never needed in the repo itself)

v2.1 dual-layer hashing (review #10 P0-U2): the v2.0 single config_hash
covered the display-only provenance token gate.memo_frozen_commit, so the
distribution copy (which redacts that token in this file's source) could
never pass its own self-check. Now:
  * scientific_config_hash — covers ONLY results-affecting configuration
    (display/provenance tokens excluded); IDENTICAL across the repo, the
    named package and the distribution copy.
  * package_variant_hash — covers the full materialized config including
    display tokens; each variant freezes its own value (the package builder re-freezes it reproducibly via --refreeze-variant).
  * config_hash — legacy alias of package_variant_hash (same construction as
    the v2.0 hash, byte-compatible: d8d9c-era values remain valid); kept so
    MANIFEST.csv consumers and old manifests keep working.
--check verifies the scientific hash always, plus whichever variant fields
the frozen manifest carries.

Statistics gated per gate_decision.md conditions: esr/gas start `pending`; enabling
them requires a CHANGELOG entry (condition 8) AND flipping the gate here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "stage4" / "run_manifest.yaml"

# display/provenance-only tokens excluded from the scientific hash: they do
# not affect any computed result, and the package builder may transform them on
# distribution copies (review #10 §0 defect class vi)
DISPLAY_ONLY = (("gate", "memo_frozen_commit"),)

CORE_MODELS = ["chronos_bolt", "chronos_2", "timesfm_2_5", "moirai_2_0", "lag_llama"]
QUARANTINED = ["eurusd", "usdjpy"]   # condition 7; excluded until FRED H.10 cross-check


def build_config() -> dict:
    reg = yaml.safe_load((ROOT / "harness" / "registry.yaml").read_text())
    dm = yaml.safe_load((ROOT / "data" / "data_manifest.yaml").read_text())
    # v2.0 R-1 (external review #9): vol_index series (vix/move) are regime-dating
    # inputs, NOT grid forecast assets — the checker's asset roster must exclude
    # them, else adding them to the data manifest reads as scientific config
    # drift (current cac842d1afb4431b vs frozen d8d9e7c8f58dc5d2).
    assets = sorted(k for k, v in dm.items()
                    if not v.get("quarantined") and v.get("group") != "vol_index")

    return {
        "stage": 4,
        "protocol": {
            "ctx_len": 512,
            "ctx_ablation": [256, 2048],   # mechanism 12-asset subset only
            "oos_start": "2016-01-01",
            "fit_len": 1000,               # A1 econometric main setting
            "refit_every": 21,
            "alphas": [0.01, 0.025, 0.05],
            "erules": ["e0", "e1", "e2", "e3"],
            "seed": 20260706,
            "horizon": 1,
        },
        "models": {m: {"hf_revision": reg[m]["hf_revision"],
                       "head_type": reg[m]["head_type"],
                       "native_tail_behavior": reg[m].get("native_tail_behavior")}
                   for m in CORE_MODELS},
        "assets": assets,
        "quarantined_excluded": QUARANTINED,
        "econometric_baselines": ["garch_t", "gjr_t", "ewma94", "hs250", "hs500",
                                  "fhs", "caviar_sav"],
        # governance: statistic -> enablement gate (gate_decision.md)
        "statistics": {
            "kupiec": "enabled", "christoffersen_cc": "enabled", "dq": "enabled",
            "quantile_score": "enabled", "fz0": "enabled", "as_z2": "enabled",
            "dm": "enabled", "mcs": "enabled",
            "n_alpha": "enabled", "precision_floor": "enabled", "fragile_flag": "enabled",
            "mc_calibrated_critical_values": "enabled",  # condition 2 — landed 2026-07-06
            "esr": "vendored-r-engine",                  # design condition 8, decision ② 2026-07-07
            "gas_crosscheck": "dropped-per-3.1",         # design condition 8, decision ③ 2026-07-07
        },
        "ranking_rules": {
            "conditional_on_erule": True,                # condition 3
            "headline_heads": ["chronos_2", "moirai_2_0", "lag_llama"],  # condition 4
            "e1_headline": False,                        # condition 9: exhibit only
            "bolt_deep_tail_via": ["e1", "e2", "e3"],    # condition 5
            "kendall_tau_e2_e3": "pre_registered",       # condition 11
        },
        "gate": {"decision_record": "docs/gate_decision.md",
                 "memo_frozen_commit": "8ef8f1b926e8",
                 "verdict": "GO", "n_conditions": 11},
    }


def _hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def scientific_config(cfg: dict) -> dict:
    """The results-affecting configuration: cfg minus display-only tokens."""
    sc = copy.deepcopy(cfg)
    for path in DISPLAY_ONLY:
        node = sc
        for key in path[:-1]:
            node = node.get(key, {})
        node.pop(path[-1], None)
    return sc


def scientific_hash(cfg: dict) -> str:
    return _hash(scientific_config(cfg))


def variant_hash(cfg: dict) -> str:
    """Full-config hash incl. display tokens — byte-compatible with the v2.0
    config_hash construction, so legacy frozen values remain checkable."""
    return _hash(cfg)


def refreeze_fields(frozen: dict, cfg: dict) -> dict:
    """Recompute the hash fields of an existing manifest against cfg (pure;
    used by the package builder on the distribution copy after redaction). The
    scientific hash must NOT move — a change there is config drift, not a
    display-variant fork."""
    old_sci = frozen.get("scientific_config_hash")
    new_sci = scientific_hash(cfg)
    if old_sci is not None and old_sci != new_sci:
        raise SystemExit(f"REFREEZE REFUSED: scientific config drifted "
                         f"({new_sci} != frozen {old_sci}) — this is not a "
                         f"display-variant fork")
    out = dict(frozen)
    out["scientific_config_hash"] = new_sci
    out["package_variant_hash"] = variant_hash(cfg)
    out["config_hash"] = out["package_variant_hash"]     # legacy alias
    # date-only precision (v2.1.1 B-MINOR-7): a second-resolution stamp made
    # every anon rebuild's manifest — and hence its zip — differ; the field
    # is provenance ("refrozen on this date"), not covered by either hash
    out["variant_refrozen_at"] = datetime.now(timezone.utc).date().isoformat()
    return out


def check(frozen: dict, cfg: dict) -> None:
    sh, vh = scientific_hash(cfg), variant_hash(cfg)
    frozen_sci = frozen.get("scientific_config_hash")
    if frozen_sci is not None:
        if frozen_sci != sh:
            raise SystemExit(f"CONFIG DRIFT (scientific): current {sh} != "
                             f"frozen {frozen_sci}")
        fv = frozen.get("package_variant_hash", frozen.get("config_hash"))
        if fv is not None and fv != vh:
            raise SystemExit(f"VARIANT DRIFT: current {vh} != frozen {fv} — "
                             f"the materialized display tokens changed after "
                             f"this variant's freeze")
        print(f"OK: scientific config {sh} and variant {vh} match the "
              f"frozen manifest")
        return
    # legacy v2.0 manifest: single config_hash == the variant construction
    if frozen["config_hash"] != vh:
        raise SystemExit(f"CONFIG DRIFT: current {vh} != frozen "
                         f"{frozen['config_hash']}")
    print(f"OK: config matches frozen manifest ({vh}) [legacy single-hash "
          f"manifest; dual-layer fields land at the next authorized freeze]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--refreeze-variant", action="store_true",
                    help="package-builder mode: rewrite the hash fields of the "
                         "existing manifest from the current tree")
    args = ap.parse_args()
    cfg = build_config()

    if args.check:
        if not OUT.exists():
            raise SystemExit("no frozen manifest")
        check(yaml.safe_load(OUT.read_text()), cfg)
        return

    if args.refreeze_variant:
        if not OUT.exists():
            raise SystemExit("no frozen manifest to refreeze")
        frozen = yaml.safe_load(OUT.read_text())
        out = refreeze_fields(frozen, cfg)
        OUT.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
        print(f"variant refrozen: scientific {out['scientific_config_hash']} "
              f"/ variant {out['package_variant_hash']}")
        return

    vh, sh = variant_hash(cfg), scientific_hash(cfg)
    if OUT.exists():
        frozen = yaml.safe_load(OUT.read_text())
        if frozen["config_hash"] != vh:
            raise SystemExit(f"manifest already frozen at {frozen['config_hash']}, "
                             f"current config hashes to {vh} — refusing silent overwrite; "
                             f"a config change needs a CHANGELOG entry + explicit re-freeze")
        print(f"already frozen, unchanged ({vh})")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = {"config_hash": vh,                       # legacy alias, kept first
           "scientific_config_hash": sh,
           "package_variant_hash": vh,
           "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           **cfg}
    OUT.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    print(f"FROZEN run_manifest scientific {sh} / variant {vh} — "
          f"{len(cfg['assets'])} assets, {len(cfg['models'])} models")


if __name__ == "__main__":
    main()
