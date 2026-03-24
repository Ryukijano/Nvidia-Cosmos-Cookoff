"""Shared utilities for dynamic surface code circuit builders.

These helpers centralize the Stim string assembly, coordinate bookkeeping, and
noisy layer construction used by the dynamic variants inspired by Morvan et al.
(2025). Each builder returns a Stim circuit string with qubit coordinates,
noisy entangling layers, detectors that stitch time-adjacent measurements, and a
terminal logical observable constructed from a boundary measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from surface_code_in_stem.noise_models import NoiseModel, resolve_noise_model
from surface_code_in_stem.surface_code import adjacent_coords, prepare_coords

Coord = Tuple[float, float]


@dataclass
class DynamicLayout:
    """Container for the square-lattice surface code coordinates used here."""

    datas: List[Coord]
    x_measures: List[Coord]
    z_measures: List[Coord]
    coord_to_index: Dict[Coord, int]

    @classmethod
    def build(cls, distance: int) -> "DynamicLayout":
        datas, x_measures, z_measures, coord_to_index = prepare_coords(distance)
        return cls(datas=datas, x_measures=x_measures, z_measures=z_measures, coord_to_index=coord_to_index)


class StimStringBuilder:
    """Utility for assembling Stim circuit strings with detector bookkeeping."""

    def __init__(self, coord_lookup: Dict[int, Coord]):
        self.lines: List[str] = []
        self.measurement_count = 0
        self.coord_lookup = coord_lookup

    def add(self, line: str) -> None:
        if line.strip():
            self.lines.append(line.rstrip())

    def measure(self, qubits: Sequence[int]) -> List[int]:
        if not qubits:
            return []
        targets = " ".join(map(str, qubits))
        self.add(f"M {targets}")
        indices = list(range(self.measurement_count, self.measurement_count + len(qubits)))
        self.measurement_count += len(qubits)
        return indices

    def detector(self, coord: Coord, rec_indices: Sequence[int]) -> None:
        offsets = [self.measurement_count - idx for idx in rec_indices]
        rec_terms = " ".join(f"rec[-{offset}]" for offset in offsets)
        self.add(f"DETECTOR({coord[0]}, {coord[1]}) {rec_terms}")

    def observable(self, qubit_measure_indices: Sequence[int]) -> None:
        offsets = [self.measurement_count - idx for idx in qubit_measure_indices]
        rec_terms = " ".join(f"rec[-{offset}]" for offset in offsets)
        self.add(f"OBSERVABLE_INCLUDE(0) {rec_terms}")

    def qubit_coords(self) -> None:
        for index, coord in self.coord_lookup.items():
            self.add(f"QUBIT_COORDS({coord[0]},{coord[1]}) {index}")

    def build(self) -> str:
        return "\n".join(self.lines) + "\n"


def index_string(coords: Iterable[Coord], c2i: Dict[Coord, int]) -> str:
    return " ".join(str(c2i[c]) for c in coords)


def noisy_layer(
    builder: StimStringBuilder,
    gate: str,
    pairs: List[int],
    idle_qubits: Sequence[int],
    noise_model: NoiseModel,
    layer_id: str,
) -> None:
    if not pairs:
        return
    targets = " ".join(map(str, pairs))
    builder.add(f"{gate} {targets}")
    for line in noise_model.gate_noise(
        gate=gate, pair_targets=pairs, idle_targets=idle_qubits, layer_id=layer_id
    ):
        builder.add(line)
    builder.add("TICK")


def orientation_pairs(
    measures: List[Coord], orient: int, c2i: Dict[Coord, int], reorder: Sequence[int] | None = None
) -> List[int]:
    pairs: List[int] = []
    for measure in measures:
        adj = adjacent_coords(measure)
        if reorder is not None:
            adj = [adj[i] for i in reorder]
        if orient < len(adj) and adj[orient] in c2i:
            pairs.extend([c2i[measure], c2i[adj[orient]]])
    return pairs


def stabilizer_cycle(
    builder: StimStringBuilder,
    layout: DynamicLayout,
    p: float,
    orientations: Sequence[int],
    gate: str,
    noise_model: Optional[NoiseModel] = None,
    reset_data: bool = False,
    measure_data: bool = False,
    prev_meas: Dict[Coord, int] | None = None,
) -> Dict[Coord, int]:
    prev_meas = prev_meas or {}
    datas, x_measures, z_measures, c2i = (
        layout.datas,
        layout.x_measures,
        layout.z_measures,
        layout.coord_to_index,
    )
    noise_model = resolve_noise_model(p, noise_model)
    measure_qubits = x_measures + z_measures
    all_qubits = datas + measure_qubits

    reset_targets = measure_qubits + (datas if reset_data else [])
    if reset_targets:
        builder.add(f"R {index_string(reset_targets, c2i)}")
        for line in noise_model.reset_noise(qubits=[c2i[q] for q in reset_targets], layer_id="dynamic_reset"):
            builder.add(line)
        builder.add("TICK")

    builder.add(f"H {index_string(x_measures, c2i)}")
    for line in noise_model.gate_noise(gate="H", pair_targets=[], idle_targets=[c2i[q] for q in all_qubits], layer_id="dynamic_h_pre"):
        builder.add(line)
    builder.add("TICK")

    reorder = [0, 2, 1, 3]
    for orient in orientations:
        cz_pairs = orientation_pairs(z_measures, orient, c2i)
        cx_pairs = orientation_pairs(x_measures, orient, c2i, reorder=reorder)
        active_pair_targets = set(cz_pairs) | set(cx_pairs)
        idle_targets = [c2i[q] for q in all_qubits if c2i[q] not in active_pair_targets]
        noisy_layer(builder, gate, cz_pairs, idle_targets, noise_model, f"dynamic_z_orient_{orient}")
        noisy_layer(builder, gate, cx_pairs, idle_targets, noise_model, f"dynamic_x_orient_{orient}")

    builder.add(f"H {index_string(x_measures, c2i)}")
    for line in noise_model.gate_noise(gate="H", pair_targets=[], idle_targets=[c2i[q] for q in all_qubits], layer_id="dynamic_h_post"):
        builder.add(line)
    builder.add("TICK")

    measurement_targets = measure_qubits + (datas if measure_data else [])
    for line in noise_model.measurement_noise(qubits=[c2i[q] for q in measurement_targets], layer_id="dynamic_measure"):
        builder.add(line)
    record_indices = builder.measure([c2i[q] for q in measurement_targets])
    coord_order = measure_qubits + (datas if measure_data else [])
    coord_to_rec = {coord: idx for coord, idx in zip(coord_order, record_indices)}

    for coord in measure_qubits:
        current = coord_to_rec[coord]
        if coord in prev_meas:
            builder.detector(coord, [prev_meas[coord], current])
        else:
            builder.detector(coord, [current])

    return coord_to_rec
