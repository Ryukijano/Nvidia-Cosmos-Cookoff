"""iSWAP-native dynamic surface code circuit builder."""
from __future__ import annotations

from typing import Dict, Optional

from surface_code_in_stem.noise_models import NoiseModel

from .base import DynamicLayout, StimStringBuilder, stabilizer_cycle


def iswap_surface_code(distance: int, rounds: int, p: float, noise_model: Optional[NoiseModel] = None) -> str:
    """Return a Stim circuit string for the iSWAP-native dynamic surface code.

    Forward cycles use one orientation ordering while odd cycles reverse it to
    mimic the time-reversal pairing described in the paper. The entangling gate
    is `ISWAP`, emphasizing compatibility with exchange-based couplers.
    """

    layout = DynamicLayout.build(distance)
    index_to_coord = {idx: coord for coord, idx in layout.coord_to_index.items()}
    builder = StimStringBuilder(coord_lookup=index_to_coord)
    builder.qubit_coords()

    orientations_fwd = (0, 3, 1, 2)
    orientations_rev = tuple(reversed(orientations_fwd))

    prev_meas: Dict[tuple[float, float], int] = {}
    for cycle in range(rounds):
        orient = orientations_fwd if cycle % 2 == 0 else orientations_rev
        prev_meas = stabilizer_cycle(
            builder,
            layout,
            p=p,
            orientations=orient,
            gate="ISWAP",
            noise_model=noise_model,
            reset_data=False,
            measure_data=False,
            prev_meas=prev_meas,
        )

    logical_targets = [layout.coord_to_index[c] for c in layout.datas[:distance]]
    logical_meas = builder.measure(logical_targets)
    builder.observable(logical_meas)
    return builder.build()
