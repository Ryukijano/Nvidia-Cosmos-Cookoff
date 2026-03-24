# ============================
# Provided utility functions
from __future__ import annotations

from typing import Iterable, List, Optional

from surface_code_in_stem.noise_models import NoiseModel, resolve_noise_model

def data_coords(distance):
    # Returns coordinate pairs from (1,1) to (distance,distance).
    coords = []
    for row in range(1, distance+1):
        for col in range(1, distance+1):
            coords.append((col, row))
    return coords

def z_measure_coords(distance):
    # Returns coordinate pairs for Z measure qubits, offset from
    #  the data qubits by 0.5.
    coords = []
    for row in range(1, distance): # don't include the last row
        for col in range(1, distance+1, 2): # only take every other qubit
            if row%2: 
                coords.append((col-0.5, row+0.5))
            else:
                coords.append((col+0.5, row+0.5))
    return coords

def x_measure_coords(distance):
    # Returns coordinate pairs for X measure qubits, offset from
    #  the data qubits by 0.5 and opposite the Y measure qubits.
    coords = []
    for row in range(1, distance+2): # include extra for last row measures
        for col in range(2, distance, 2): # start from second column, ignore last
            if row%2:
                coords.append((col+0.5, row-0.5))
            else:
                coords.append((col-0.5, row-0.5))
    return coords

def coords_to_index(coords):
    # Inverts a list of coordinates into a dict that maps the coord 
    #  to its index in the list.
    return {tuple(c):i for i,c in dict(enumerate(coords)).items()}

def adjacent_coords(coord):
    # Returns the four coordinates at diagonal 0.5 offsets from the input coord.
    # Follows the X-stabilizer plaquette corner ordering from the lecture: 
    #  top-left, top-right, bottom-left, bottom-right.
    col, row = coord
    adjacents = [(col-0.5, row-0.5), (col+0.5, row-0.5),
               (col-0.5, row+0.5), (col+0.5, row+0.5),
              ]
    return adjacents

def index_string(coord_list, c2i):
    # Returns the indicies for each coord in a list as space-delimited string.
    return ' '.join(str(c2i[coord]) for coord in coord_list)

def prepare_coords(distance):
    # Returns coordinates for data qubits, x measures and z measures, along with 
    #  a coordinate-to-index mapping for all of the qubits.
    # The indices are ordered: data first, then x measures, then z measures.
    datas = data_coords(distance)
    x_measures = x_measure_coords(distance)
    z_measures = z_measure_coords(distance)
    c2i = coords_to_index(datas+x_measures+z_measures)
    return datas, x_measures, z_measures, c2i

def coord_circuit(distance):
    # Returns a Stim circuit string that adds a QUBIT_COORDS instruction for each
    #  qubit, based on the coordinate-to-index mapping.
    _, _, _, c2i = prepare_coords(distance)
    stim_circuit = ""
    for coord, index in c2i.items():
        stim_circuit += f"QUBIT_COORDS({','.join(map(str, coord))}) {index}\n"
    return stim_circuit

def label_indices(distance):
    # Returns a Stim circuit string that labels each of the qubits with their 
    #  type and index in the coordinate-to-index mapping.
    # Uses ERROR operations to do the labeling: X_ and Z_ERRORs correspond to
    #  qubits that will be used for X and Z type stabilizer measurements, and
    #  Y_ERRORs label the data qubits. 
    # The index of the qubit is encoded in the operation's error probability: 
    #  The value after the decimal is the index. Eg. 0.01 is 1 and 0.1 is 10.
    datas, x_measures, z_measures, c2i = prepare_coords(distance)
    all_qubits = datas + x_measures + z_measures
    i = 0
    stim_string = ""
    for coord in datas:
        stim_string += f"Y_ERROR(0.{i:>02}) {c2i[coord]}\n"
        i += 1
    stim_string += "TICK\n"
    for coord in x_measures:
        stim_string += f"X_ERROR(0.{i:>02}) {c2i[coord]}\n"
        i += 1
    stim_string += "TICK\n"
    for coord in z_measures:
        stim_string += f"Z_ERROR(0.{i:>02}) {c2i[coord]}\n"
        i += 1

    return stim_string

# ======================================================
# hidden answer functions

def _extend_noise(lines: List[str], noise_lines: Iterable[str]) -> None:
    for line in noise_lines:
        if line:
            lines.append(line)


def lattice_with_noise(distance, p, noise_model: Optional[NoiseModel] = None):
    noise_model = resolve_noise_model(p, noise_model)
    datas, x_measures, z_measures, c2i = prepare_coords(distance)
    # create a stim circuit string for just the lattice of CX gates
    #  required by the stabilizers.
    lines: List[str] = []
    for i in range(4):
        cx_qubits = []
        for measure in z_measures:
            z_controls = adjacent_coords(measure)
            control = z_controls[i]
            if control in c2i:
                cx_qubits.extend([control, measure])

        for measure in x_measures:
            x_targets = adjacent_coords(measure)
            index_reorder = [0, 2, 1, 3]
            target = x_targets[index_reorder[i]]
            if target in c2i:
                cx_qubits.extend([measure, target]) # flipped order!

        idle_qubits = [coord for coord in c2i.keys() if coord not in cx_qubits]

        pair_indices = [c2i[q] for q in cx_qubits]
        idle_indices = [c2i[q] for q in idle_qubits]
        lines.append(f"CX {' '.join(map(str, pair_indices))}")
        _extend_noise(
            lines,
            noise_model.gate_noise(
                gate="CX",
                pair_targets=pair_indices,
                idle_targets=idle_indices,
                layer_id=f"lattice_orient_{i}",
            ),
        )
        lines.append("TICK")

    return "\n".join(lines) + "\n"


def stabilizers_with_noise(distance, p, noise_model: Optional[NoiseModel] = None):
    noise_model = resolve_noise_model(p, noise_model)
    datas, x_measures, z_measures, c2i = prepare_coords(distance)
    all_measures = x_measures + z_measures
    all_qubits = datas + all_measures
    # Use `lattice_with_noise` to create a full lattice of stabilizers
    #  including the resets and measurements. No detectors yet.
    lines = [f"R {index_string(all_measures, c2i)}"]
    _extend_noise(lines, noise_model.reset_noise(qubits=[c2i[q] for q in all_measures], layer_id="stabilizer_reset"))
    _extend_noise(
        lines,
        noise_model.gate_noise(
            gate="IDLE",
            pair_targets=[],
            idle_targets=[c2i[q] for q in datas],
            layer_id="stabilizer_post_reset_idle",
        ),
    )
    lines.append("TICK")
    lines.append(f"H {index_string(x_measures, c2i)}")
    _extend_noise(
        lines,
        noise_model.gate_noise(
            gate="H",
            pair_targets=[],
            idle_targets=[c2i[q] for q in all_qubits],
            layer_id="stabilizer_h_pre",
        ),
    )
    lines.append("TICK")

    lines.append(lattice_with_noise(distance, p, noise_model=noise_model).strip())

    lines.append(f"H {index_string(x_measures, c2i)}")
    _extend_noise(
        lines,
        noise_model.gate_noise(
            gate="H",
            pair_targets=[],
            idle_targets=[c2i[q] for q in all_qubits],
            layer_id="stabilizer_h_post",
        ),
    )
    lines.append("TICK")
    _extend_noise(lines, noise_model.measurement_noise(qubits=[c2i[q] for q in all_measures], layer_id="stabilizer_meas"))
    _extend_noise(
        lines,
        noise_model.gate_noise(
            gate="IDLE",
            pair_targets=[],
            idle_targets=[c2i[q] for q in datas],
            layer_id="stabilizer_pre_meas_idle",
        ),
    )
    lines.append(f"M {index_string(all_measures, c2i)}")
    lines.append("TICK")

    return "\n".join(lines) + "\n"

def initialization_step(distance, p, noise_model: Optional[NoiseModel] = None):
    noise_model = resolve_noise_model(p, noise_model)
    datas, x_measures, z_measures, c2i = prepare_coords(distance)
    all_measures = x_measures + z_measures
    all_qubits = datas + all_measures
    # Use `lattice_with_noise` to create the first round of stabilizer
    #  measurements in the surface code. Reference but don't use
    #  `stabilizers_with_noise`. Add first-round detectors.
    lines = [f"R {index_string(all_qubits, c2i)}"]
    _extend_noise(lines, noise_model.reset_noise(qubits=[c2i[q] for q in all_qubits], layer_id="init_reset"))
    lines.append("TICK")
    lines.append(f"H {index_string(x_measures, c2i)}")
    _extend_noise(
        lines,
        noise_model.gate_noise(gate="H", pair_targets=[], idle_targets=[c2i[q] for q in all_qubits], layer_id="init_h_pre"),
    )
    lines.append("TICK")

    lines.append(lattice_with_noise(distance, p, noise_model=noise_model).strip())

    lines.append(f"H {index_string(x_measures, c2i)}")
    _extend_noise(
        lines,
        noise_model.gate_noise(gate="H", pair_targets=[], idle_targets=[c2i[q] for q in all_qubits], layer_id="init_h_post"),
    )
    lines.append("TICK")
    _extend_noise(lines, noise_model.measurement_noise(qubits=[c2i[q] for q in all_measures], layer_id="init_meas"))
    lines.append(f"M {index_string(all_measures, c2i)}")
    _extend_noise(lines, noise_model.gate_noise(gate="IDLE", pair_targets=[], idle_targets=[c2i[q] for q in datas], layer_id="init_meas_data_idle"))
    lines.append("TICK")

    for i in range(1, len(z_measures) + 1):
        lines.append(f"DETECTOR({i}, 0) rec[{-i}]")
    return "\n".join(lines) + "\n"

def rounds_step(distance, rounds, p, noise_model: Optional[NoiseModel] = None):
    noise_model = resolve_noise_model(p, noise_model)
    # Use `stabilizers_with_noise` to implement the `REPEAT` block of
    #  stabilizers. Include the mid-round detectors.
    stim_string = f""
    if rounds <= 2:
        return "\n"
    datas, x_measures, z_measures, c2i = prepare_coords(distance)

    stim_string = f"REPEAT {rounds-2} {{\n"
    stim_string += stabilizers_with_noise(distance, p, noise_model=noise_model)

    num_measures_per_type = len(z_measures) # number of measures per type per round
    for i in range(1, num_measures_per_type + 1): # offset to the previous round
        stim_string += f"DETECTOR({i}, 0) rec[{-i}] rec[{-(i+2*num_measures_per_type)}]\n"
    for i in range(1, num_measures_per_type + 1): # offset to the other type and to the previous round
        stim_string += f"DETECTOR({i}, 0) rec[{-(i+num_measures_per_type)}] rec[{-(i+3*num_measures_per_type)}]\n"

    stim_string += """
    }
    """ # end repeat block

    return stim_string
    
def final_step(distance, p, noise_model: Optional[NoiseModel] = None):
    noise_model = resolve_noise_model(p, noise_model)
    datas, x_measures, z_measures, c2i = prepare_coords(distance)
    all_measures = x_measures + z_measures
    all_qubits = datas + all_measures
    # Use `lattice_with_noise` to implement the final round of stabilizer
    #  measurements and the final data measurements. Add the last round
    #  detectors, the final data measure detectors, and the
    #  `OBSERVABLE_INCLUDE` instruction.
    lines = [f"R {index_string(all_measures, c2i)}"]
    _extend_noise(lines, noise_model.reset_noise(qubits=[c2i[q] for q in all_measures], layer_id="final_reset"))
    _extend_noise(lines, noise_model.gate_noise(gate="IDLE", pair_targets=[], idle_targets=[c2i[q] for q in datas], layer_id="final_reset_data_idle"))
    lines.append("TICK")
    lines.append(f"H {index_string(x_measures, c2i)}")
    _extend_noise(lines, noise_model.gate_noise(gate="H", pair_targets=[], idle_targets=[c2i[q] for q in all_qubits], layer_id="final_h_pre"))
    lines.append("TICK")

    lines.append(lattice_with_noise(distance, p, noise_model=noise_model).strip())

    lines.append(f"H {index_string(x_measures, c2i)}")
    _extend_noise(lines, noise_model.gate_noise(gate="H", pair_targets=[], idle_targets=[c2i[q] for q in all_qubits], layer_id="final_h_post"))
    lines.append("TICK")
    _extend_noise(lines, noise_model.measurement_noise(qubits=[c2i[q] for q in all_qubits], layer_id="final_measurement"))
    lines.append(f"M {index_string(all_qubits, c2i)}")
    # remember measure order is datas, x_measures, z_measures
    # do previous-round detectors first
    num_measures_per_type = len(z_measures) # number of measures per type per round
    num_datas = len(datas)
    for i in range(1, num_measures_per_type + 1): # offset to the previous round
        lines.append(f"DETECTOR({i}, 0) rec[{-i}] rec[{-(i+2*num_measures_per_type+num_datas)}]")
    for i in range(1, num_measures_per_type + 1): # offset to the other type and to the previous round
        lines.append(f"DETECTOR({i}, 0) rec[{-(i+num_measures_per_type)}] rec[{-(i+3*num_measures_per_type+num_datas)}]")

    # now the confusing one: the final data measurements and their adjacent measure measurements
    # create a dict that maps each coord to the record index of the most recent measurement on it
    coord_to_record_index = {coord: i - len(all_qubits) for i, coord in enumerate(all_qubits)}
    for i, measure in enumerate(z_measures):
        record_indices = []
        record_indices.append(coord_to_record_index[measure])
        adjacent_datas = adjacent_coords(measure)

        for data in adjacent_datas:
            if data in all_qubits:
                record_indices.append(coord_to_record_index[data])
        recs = [f"rec[{j}]" for j in record_indices]
        lines.append(f"DETECTOR({i}, 0) {' '.join(recs)}")

    obs_recs = [f"rec[{-(i+2*num_measures_per_type)}]" for i in range(1, distance + 1)]
    lines.append(f"OBSERVABLE_INCLUDE(0) {' '.join(obs_recs)}")

    return "\n".join(lines) + "\n"

def surface_code_circuit_string(distance, rounds, p, noise_model: Optional[NoiseModel] = None):
    noise_model = resolve_noise_model(p, noise_model)
    string = coord_circuit(distance)
    string += initialization_step(distance, p, noise_model=noise_model)
    string += rounds_step(distance, rounds, p, noise_model=noise_model)
    string += final_step(distance, p, noise_model=noise_model)
    return string
