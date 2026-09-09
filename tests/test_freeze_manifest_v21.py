"""v2.1 P0-U2 dual-layer manifest hashing (review #10 §4).

The scientific hash covers only results-affecting configuration and is
invariant to display/provenance-token transforms (the package builder's
memo_frozen_commit redaction); the variant hash covers the full materialized
config and is byte-compatible with the v2.0 single config_hash, so the
currently frozen repo manifest keeps checking green without being rewritten.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from harness.freeze_run_manifest import (OUT, build_config, refreeze_fields,
                                         scientific_config, scientific_hash,
                                         variant_hash)

ROOT = Path(__file__).resolve().parent.parent


def _redacted(cfg):
    """Flip the display token to a value GUARANTEED to differ from the
    current one — in the distribution copy build_config() already
    returns '[redacted-commit]', so a fixed sentinel made the v2.1 Phase-4
    in-package run a vacuous no-op (variant hashes equal, assert failed).
    The redaction-invariance property under test is token-value-independent."""
    c = yaml.safe_load(yaml.safe_dump(cfg))          # deep copy
    cur = c["gate"]["memo_frozen_commit"]
    c["gate"]["memo_frozen_commit"] = ("[redacted-commit]"
                                       if cur != "[redacted-commit]"
                                       else "[redacted-commit-alt]")
    return c


def test_scientific_hash_invariant_to_display_redaction():
    cfg = build_config()
    red = _redacted(cfg)
    assert scientific_hash(cfg) == scientific_hash(red)
    assert variant_hash(cfg) != variant_hash(red)
    # the display token is genuinely absent from the scientific config
    assert "memo_frozen_commit" not in scientific_config(cfg)["gate"]
    # and nothing else was dropped
    sc, full = scientific_config(cfg), cfg
    assert set(full) == set(sc)
    assert set(full["gate"]) - set(sc["gate"]) == {"memo_frozen_commit"}


def test_variant_hash_byte_compatible_with_v20_frozen_value():
    """The variant hash is the v2.0 construction bit-for-bit: it must equal
    the config_hash frozen in the repo manifest (which Phase 1 does NOT
    rewrite)."""
    frozen = yaml.safe_load(OUT.read_text())
    assert variant_hash(build_config()) == frozen["config_hash"]


def test_check_passes_on_current_repo():
    r = subprocess.run([sys.executable, "-m",
                        "harness.freeze_run_manifest", "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


def test_refreeze_fields_variant_moves_scientific_pinned():
    cfg = build_config()
    frozen = {"config_hash": variant_hash(cfg),
              "scientific_config_hash": scientific_hash(cfg),
              "package_variant_hash": variant_hash(cfg)}
    red = _redacted(cfg)
    out = refreeze_fields(frozen, red)
    assert out["scientific_config_hash"] == scientific_hash(cfg)   # pinned
    assert out["package_variant_hash"] == variant_hash(red)        # moved
    assert out["config_hash"] == out["package_variant_hash"]       # alias
    assert "variant_refrozen_at" in out


def test_refreeze_refuses_scientific_drift():
    cfg = build_config()
    frozen = {"config_hash": variant_hash(cfg),
              "scientific_config_hash": scientific_hash(cfg)}
    drifted = yaml.safe_load(yaml.safe_dump(cfg))
    drifted["protocol"]["ctx_len"] = 1024                          # scientific!
    with pytest.raises(SystemExit, match="REFREEZE REFUSED"):
        refreeze_fields(frozen, drifted)
