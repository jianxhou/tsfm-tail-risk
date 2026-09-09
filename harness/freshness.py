"""G6 freshness layer (v2.1.1, review #11 structural adoption).

Defect class being closed: an upstream artifact is regenerated but a
downstream consumer is not, and existing checks keep passing because their
asserts compare against frozen intermediates (review #10 §0 class (vii)).
G6 makes staleness itself a hard check failure.

Mechanism:
- `paper/tools/freshness_graph.yaml` is the DECLARED dependency graph (the
  mechanized form of the review-#11 F5 census): one entry per generator
  run-group with its data inputs (paths or globs) and outputs.
- At the end of a successful run, each covered generator calls
  `stamp(<group>)`, which expands the declared inputs, hashes inputs AND
  outputs (sha256, streamed), and writes a deterministic sidecar
  (sorted keys, no timestamps).
- `check_gates.py` G6 re-expands and re-hashes at check time and FAILS on:
  missing sidecar, generator mismatch, input-set drift (file added or
  removed), input-hash drift (upstream changed after generation), output
  missing, or output-hash drift (hand edit / unstamped regeneration).

Volatile project logs (CHANGELOG.md, PROGRESS.md) are deliberately NOT
declarable inputs: ledger rebuilds classify their lag as an expected
"known-lag refresh" class, so hashing them would keep G6 red between
authorized rebuilds without adding safety.

CLI:
    python -m harness.freshness --check              # all groups (G6 body)
    python -m harness.freshness --stamp GROUP        # one group
    python -m harness.freshness --bootstrap          # stamp every group
The one-time --bootstrap run that armed this layer is the census
consolidation of record (CHANGELOG 2026-08-05, v2.1.1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "paper" / "tools" / "freshness_graph.yaml"
VOLATILE = {"CHANGELOG.md", "PROGRESS.md"}


def load_registry() -> list[dict]:
    reg = yaml.safe_load(REGISTRY.read_text())
    names = [g["group"] for g in reg]
    assert len(names) == len(set(names)), "duplicate group names in registry"
    for g in reg:
        for p in g["inputs"]:
            assert Path(p).name not in VOLATILE, (
                f"{g['group']}: volatile governance log {p} may not be a "
                f"declared G6 input (see module docstring)")
    return reg


def _expand(patterns: list[str], exclude: tuple | list = ()) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            hits = sorted(p for p in ROOT.glob(pat) if p.is_file())
            if not hits:
                raise FileNotFoundError(f"glob matched nothing: {pat}")
            out.extend(hits)
        else:
            p = ROOT / pat
            if not p.exists():
                raise FileNotFoundError(pat)
            out.append(p)
    excl = {str(ROOT / e) for e in exclude}
    seen: set[str] = set()
    uniq = []
    for p in out:
        s = str(p)
        if s in excl or s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


_SHA_CACHE: dict[tuple[str, float, int], str] = {}


def _sha(p: Path) -> str:
    st = p.stat()
    key = (str(p), st.st_mtime, st.st_size)
    if key in _SHA_CACHE:
        return _SHA_CACHE[key]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def _hash_map(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(ROOT)): _sha(p) for p in paths}


def _sidecar_path(g: dict) -> Path:
    return ROOT / g["sidecar"]


def stamp(group: str) -> Path:
    g = next(x for x in load_registry() if x["group"] == group)
    payload = {
        "group": g["group"],
        "generator": g["generator"],
        "inputs": _hash_map(_expand(g["inputs"], g.get("exclude_inputs", ()))),
        "outputs": _hash_map(_expand(g["outputs"], g.get("exclude_outputs", ()))),
    }
    sc = _sidecar_path(g)
    sc.parent.mkdir(parents=True, exist_ok=True)
    sc.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"[freshness] stamped {g['group']} -> {sc.relative_to(ROOT)} "
          f"({len(payload['inputs'])} inputs, {len(payload['outputs'])} outputs)")
    return sc


def stamp_safe(group: str) -> None:
    """Generator-side entry point. In the repo (ROOT/.git present) this is a
    HARD stamp — a missing declared input crashes the generator, which is
    correct there. In a distributed package copy (no .git; the per-day
    upstream layer is not shipped) it skips LOUDLY: the in-package
    regeneration equality checks, not G6, are the package-side integrity
    mechanism (G6 runs repo-side only)."""
    if not (ROOT / ".git").exists():
        print(f"[freshness] SKIP stamp({group}): package copy (no .git); "
              f"repo-side G6 is the freshness authority")
        return
    stamp(group)


def check(g: dict) -> list[str]:
    bad: list[str] = []
    name = g["group"]
    sc = _sidecar_path(g)
    if not sc.exists():
        return [f"G6 {name}: sidecar missing ({g['sidecar']}) — regenerate "
                f"via {g['generator']} (which stamps on success)"]
    rec = json.loads(sc.read_text())
    if rec.get("generator") != g["generator"]:
        bad.append(f"G6 {name}: generator mismatch "
                   f"(sidecar {rec.get('generator')} != registry {g['generator']})")
    for kind in ("inputs", "outputs"):
        try:
            now = _hash_map(_expand(g[kind], g.get(f"exclude_{kind}", ())))
        except FileNotFoundError as e:
            bad.append(f"G6 {name}: {kind} path missing: {e}")
            continue
        old = rec.get(kind, {})
        for p in sorted(set(old) - set(now)):
            bad.append(f"G6 {name}: {kind[:-1]} removed since stamp: {p}")
        for p in sorted(set(now) - set(old)):
            bad.append(f"G6 {name}: {kind[:-1]} added since stamp: {p}")
        for p in sorted(set(now) & set(old)):
            if now[p] != old[p]:
                what = ("upstream changed after generation" if kind == "inputs"
                        else "output changed without a re-stamp")
                bad.append(f"G6 {name}: {kind[:-1]} hash drift ({what}): {p}")
    return bad


def check_all() -> list[str]:
    bad: list[str] = []
    for g in load_registry():
        bad.extend(check(g))
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--bootstrap", action="store_true")
    cli = ap.parse_args()
    if cli.stamp:
        stamp(cli.stamp)
    elif cli.bootstrap:
        for g in load_registry():
            stamp(g["group"])
    elif cli.check:
        bad = check_all()
        for b in bad:
            print(b)
        print("G6:", "GREEN" if not bad else f"RED ({len(bad)})")
        sys.exit(1 if bad else 0)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
