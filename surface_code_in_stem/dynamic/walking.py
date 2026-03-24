"""Walking dynamic surface code circuit builder."""
from __future__ import annotations

from typing import Dict, Optional

from surface_code_in_stem.noise_models import NoiseModel

from .base import DynamicLayout, StimStringBuilder, stabilizer_cycle


def walking_surface_code(distance: int, rounds: int, p: float, noise_model: Optional[NoiseModel] = None) -> str:
    """Return a Stim circuit string for the walking surface code.

    Each cycle swaps which sublattice receives a reset so that every physical
    qubit is refreshed every other round. Detectors stitch consecutive
    measurements of the same plaquette to expose leakage-induced correlations.
    """

    layout = DynamicLayout.build(distance)
    index_to_coord = {idx: coord for coord, idx in layout.coord_to_index.items()}
    builder = StimStringBuilder(coord_lookup=index_to_coord)
    builder.qubit_coords()

    prev_meas: Dict[tuple[float, float], int] = {}
    orientations = (0, 1, 2, 3)
    for cycle in range(rounds):
        reset_data = cycle % 2 == 1
        measure_data = cycle % 2 == 1
        prev_meas = stabilizer_cycle(
            builder,
            layout,
            p=p,
            orientations=orientations,
            gate="CX",
            noise_model=noise_model,
            reset_data=reset_data,
            measure_data=measure_data,
            prev_meas=prev_meas,
        )

    data_indices = [layout.coord_to_index[c] for c in layout.datas[:distance]]
    logical_meas = builder.measure(data_indices)
    builder.observable(logical_meas)
    return builder.build()
