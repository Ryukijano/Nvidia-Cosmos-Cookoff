"""Backwards-compatible re-export of dynamic surface code builders.

The dynamic builders now live in `surface_code_in_stem.dynamic.*` modules to
make each variant self-contained. Importing from this module continues to work
but will forward to the split implementations.
"""
from __future__ import annotations

from surface_code_in_stem.dynamic import (
    DynamicLayout,
    StimStringBuilder,
    hexagonal_surface_code,
    iswap_surface_code,
    walking_surface_code,
    xyz2_hexagonal_code,
)

__all__ = [
    "DynamicLayout",
    "StimStringBuilder",
    "hexagonal_surface_code",
    "iswap_surface_code",
    "walking_surface_code",
    "xyz2_hexagonal_code",
]
