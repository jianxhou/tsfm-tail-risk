"""Mechanism-panel adapters (proposal §3.1: 12-asset subset).

Within-family head contrasts: Chronos-base (token-quantized sampling) vs
Chronos-Bolt, and Moirai-1.1 (mixture) vs Moirai-2.0. Implemented in the
mechanism sub-panel round (2026-07-16, signer-authorized); this module now
re-exports the real adapters so the historical registry path stays valid.
"""

from __future__ import annotations

from harness.chronos_base_adapter import ChronosBaseAdapter
from harness.moirai11_adapter import Moirai11Adapter

__all__ = ["ChronosBaseAdapter", "Moirai11Adapter"]
