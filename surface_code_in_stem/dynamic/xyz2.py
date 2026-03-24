"""XYZ2-inspired dynamic hexagonal surface code builder.

This is a lightweight stand-in for the XYZ2 construction: an inner YZZY-like
hexagonal surface-code cycle is wrapped with an outer phase-flip repetition
layer that repeatedly parity-checks the logical rail. The exact hardware-tuned
schedule is not available in this repository, but the returned circuit is a
valid Stim detector circuit that preserves the intended concatenated structure.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from surface_code_in_stem.noise_models import NoiseModel, resolve_noise_model

from .base import DynamicLayout, StimStringBuilder, noisy_layer, stabilizer_cycle


def _single_qubit_layer(
    builder: StimStringBuilder,
    gate: str,
    qubits: Sequence[int],
    noise_qubits: Sequence[int],
    noise_model: NoiseModel,
    layer_id: str,
) -> None:
    if not qubits:
        return
    builder.add(f"{gate} {' '.join(map(str, qubits))}")
    for line in noise_model.gate_noise(
        gate=gate,
        pair_targets=[],
        idle_targets=list(noise_qubits),
        layer_id=layer_id,
    ):
        builder.add(line)
    builder.add("TICK")


def _phase_flip_blocks(layout: DynamicLayout) -> List[List[int]]:
    source_indices = [layout.coord_to_index[coord] for coord in layout.z_measures]
    return [source_indices[start : start + 2] for start in range(len(source_indices) - 1)]


def _phase_flip_cycle(
    builder: StimStringBuilder,
    *,
    outer_indices: Sequence[int],
    blocks: Sequence[Sequence[int]],
    all_qubits: Sequence[int],
    noise_model: NoiseModel,
) -> None:
    if not outer_indices or not blocks:
        return

    builder.add(f"R {' '.join(map(str, outer_indices))}")
    for line in noise_model.reset_noise(qubits=outer_indices, layer_id="xyz2_outer_reset"):
        builder.add(line)
    builder.add("TICK")

    max_block_width = max(len(block) for block in blocks)
    for position in range(max_block_width):
        pair_targets: List[int] = []
        active = set()
        for outer_index, block in zip(outer_indices, blocks):
            if position >= len(block):
                continue
            data_index = block[position]
            pair_targets.extend([data_index, outer_index])
            active.add(data_index)
            active.add(outer_index)
        idle_targets = [qubit for qubit in all_qubits if qubit not in active]
        noisy_layer(
            builder,
            gate="CX",
            pairs=pair_targets,
            idle_qubits=idle_targets,
            noise_model=noise_model,
            layer_id=f"xyz2_outer_cx_{position}",
        )

    for line in noise_model.measurement_noise(qubits=outer_indices, layer_id="xyz2_outer_measure"):
        builder.add(line)
    builder.measure(outer_indices)


def xyz2_hexagonal_code(
    distance: int,
    rounds: int,
    p: float,
    noise_model: Optional[NoiseModel] = None,
) -> "stim.Circuit":
    """Return an XYZ2-inspired concatenated hexagonal code as `stim.Circuit`."""

    try:
        import stim
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Stim is required to build XYZ2 hexagonal circuits.") from exc

    if distance < 3:
        raise ValueError("distance must be at least 3.")
    if rounds <= 0:
        raise ValueError("rounds must be positive.")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1.")

    noise_model = resolve_noise_model(p, noise_model)
    layout = DynamicLayout.build(distance)

    logical_rail = layout.datas[:distance]
    phase_flip_blocks = _phase_flip_blocks(layout)
    min_outer_y = min(coord[1] for coord in layout.z_measures) - 1.5
    outer_coords = [(0.5 + index, min_outer_y) for index in range(len(phase_flip_blocks))]
    coord_to_index = dict(layout.coord_to_index)
    next_index = max(coord_to_index.values()) + 1
    outer_indices: List[int] = []
    for coord in outer_coords:
        coord_to_index[coord] = next_index
        outer_indices.append(next_index)
        next_index += 1

    index_to_coord = {index: coord for coord, index in coord_to_index.items()}
    builder = StimStringBuilder(coord_lookup=index_to_coord)
    builder.qubit_coords()

    data_indices = [coord_to_index[coord] for coord in layout.datas]
    measure_indices = [coord_to_index[coord] for coord in layout.x_measures + layout.z_measures]
    all_qubits = data_indices + measure_indices + outer_indices

    # A checkerboard S/S† dressing rotates alternating data qubits into a
    # Y-biased stabilizer frame, approximating the inner YZZY layer.
    y_rotated_data = [
        coord_to_index[coord]
        for coord in layout.datas
        if int(coord[0] + coord[1]) % 2 == 0
    ]

    orientations_fwd = (0, 1, 2)
    orientations_rev = tuple(reversed(orientations_fwd))
    prev_inner_meas: Dict[tuple[float, float], int] = {}

    for cycle in range(rounds):
        orientations = orientations_fwd if cycle % 2 == 0 else orientations_rev
        _single_qubit_layer(
            builder,
            gate="S",
            qubits=y_rotated_data,
            noise_qubits=all_qubits,
            noise_model=noise_model,
            layer_id=f"xyz2_y_rot_pre_{cycle}",
        )
        prev_inner_meas = stabilizer_cycle(
            builder,
            layout,
            p=p,
            orientations=orientations,
            gate="CZ",
            noise_model=noise_model,
            reset_data=cycle == 0,
            measure_data=False,
            prev_meas=prev_inner_meas,
        )
        _single_qubit_layer(
            builder,
            gate="S_DAG",
            qubits=y_rotated_data,
            noise_qubits=all_qubits,
            noise_model=noise_model,
            layer_id=f"xyz2_y_rot_post_{cycle}",
        )
        _phase_flip_cycle(
            builder,
            outer_indices=outer_indices,
            blocks=phase_flip_blocks,
            all_qubits=all_qubits,
            noise_model=noise_model,
        )

    logical_targets = [layout.coord_to_index[coord] for coord in logical_rail]
    for line in noise_model.measurement_noise(qubits=logical_targets, layer_id="xyz2_logical_measure"):
        builder.add(line)
    logical_meas = builder.measure(logical_targets)
    builder.observable(logical_meas)
    return stim.Circuit(builder.build())
