# Dynamic surface code circuits (Morvan et al. 2025)

This note summarises how the reference paper "Demonstration of dynamic surface codes" (Morvan et al., Nature Physics 2025) maps
onto the circuits implemented in the `surface_code_in_stem.dynamic` package.

## Hexagonal surface code
- **Paper concept:** Reduce the degree-4 connectivity of the square-lattice surface code to degree-3 on a hexagonal graph by time-alternating the stabilizer support.
- **Implementation:** `surface_code_in_stem.dynamic.hexagonal.hexagonal_surface_code(distance, rounds, p)` alternates three-edge stabilizer footprints forward and backward in time. Each cycle runs noisy entangling layers for orientations `(0, 1, 2)` then `(2, 1, 0)`, stitching detectors between consecutive measurements of the same stabilizer qubit. The logical observable is taken from a final measurement of the first logical boundary line.

## Walking surface code
- **Paper concept:** Swap data and measurement roles every cycle so that all physical qubits are reset/measured frequently, suppressing leakage and long-time error correlations.
- **Implementation:** `surface_code_in_stem.dynamic.walking.walking_surface_code(distance, rounds, p)` resets and measures the data sublattice on odd cycles while keeping the ancilla plaquettes active every round. Detectors are connected across rounds to highlight temporal correlations from leakage. A final measurement of the data boundary provides the logical observable.

## iSWAP-native surface code
- **Paper concept:** Use native iSWAP interactions with alternating forward/time-reversed cycles so that stabilizers refocus despite the SWAP-like action of the gate.
- **Implementation:** `surface_code_in_stem.dynamic.iswap.iswap_surface_code(distance, rounds, p)` replaces the entangling gate with `ISWAP` and alternates orientation orderings `(0, 3, 1, 2)` / `(2, 1, 3, 0)` to mimic the time-reversal structure. Detectors again link each stabilizer’s consecutive outcomes, and the logical observable is measured at the end.

## Usage
Each helper returns a Stim circuit string that can be simulated or decoded directly. Example:

```python
from surface_code_in_stem.dynamic import (
    hexagonal_surface_code,
    walking_surface_code,
    iswap_surface_code,
)

stim_string = hexagonal_surface_code(distance=5, rounds=6, p=0.001)
print(stim_string)
```

The generated circuits include:
- `QUBIT_COORDS` annotations for easier visualisation with Stim tools.
- Noisy entangling layers (`DEPOLARIZE1`/`DEPOLARIZE2`) after every gate layer.
- Detectors that stitch measurements of the same stabilizer across time.
- A logical observable built from a terminal measurement of a distance-long logical boundary.
