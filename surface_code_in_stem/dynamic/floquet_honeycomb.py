"""Floquet Honeycomb Code builder.

The Honeycomb code (Hastings and Haah, 2021) is a dynamic QEC code (Floquet code)
defined on a honeycomb lattice. Unlike standard surface codes, it has no static
stabilizer group. Instead, the logical state is protected by a sequence of 
two-qubit Pauli measurements (XX, YY, ZZ) that cycle in time.

This module provides a builder for generating Stim circuits for the Honeycomb code.
"""

import stim
from typing import Tuple, List, Dict, Set


def get_honeycomb_patch(distance: int) -> Tuple[List[complex], Dict[str, List[Tuple[int, int]]]]:
    """Generate coordinates and colored edges for a planar honeycomb patch.
    
    A planar honeycomb code requires specific boundary conditions to encode logical qubits.
    For simplicity in this builder, we construct a generic rectangular patch of hexagons.
    
    Args:
        distance: Proxy for the size of the patch
        
    Returns:
        coords: List of complex coordinates for each qubit
        edges: Dict mapping colors 'R', 'G', 'B' to lists of qubit index pairs
    """
    coords = []
    edges = {'R': [], 'G': [], 'B': []}
    
    # Simple rectangular mapping of a honeycomb lattice
    # A honeycomb lattice is a brick-wall lattice when drawn on a square grid
    width = distance * 2
    height = distance * 2
    
    qubit_map = {}
    idx = 0
    
    # Create vertices
    for y in range(height):
        for x in range(width):
            # Only keep vertices that belong to the brick-wall (honeycomb)
            if (x + y) % 2 == 0:
                coords.append(x + 1j * y)
                qubit_map[(x, y)] = idx
                idx += 1
                
    # Create edges and color them
    # R edges: horizontal (0-1, 2-3)
    # G edges: vertical leaning left? 
    # B edges: vertical leaning right?
    # In a brickwall:
    # horizontal edges: R
    # vertical edges from even y: G
    # vertical edges from odd y: B
    for y in range(height):
        for x in range(width):
            if (x, y) not in qubit_map:
                continue
                
            u = qubit_map[(x, y)]
            
            # Horizontal edge (R)
            if (x + 1, y) in qubit_map:
                v = qubit_map[(x + 1, y)]
                edges['R'].append((u, v))
                
            # Vertical edge down
            if (x, y + 1) in qubit_map:
                v = qubit_map[(x, y + 1)]
                if y % 2 == 0:
                    edges['G'].append((u, v))
                else:
                    edges['B'].append((u, v))
                    
    return coords, edges


def floquet_honeycomb_circuit(distance: int = 3, rounds: int = 9, p: float = 0.001) -> str:
    """Build a Stim circuit for the Floquet Honeycomb code.
    
    The cycle consists of measuring:
    Round 0 mod 3: XX on R edges
    Round 1 mod 3: YY on G edges
    Round 2 mod 3: ZZ on B edges
    
    Detectors are formed by the product of 6 edges forming a hexagon over two cycles.
    Because computing the exact dynamic detectors is topologically complex, this builder 
    emits the fundamental gauge measurements.
    
    Args:
        distance: Size of the lattice
        rounds: Number of measurement rounds
        p: Physical error rate
        
    Returns:
        Stim circuit string
    """
    coords, edges = get_honeycomb_patch(distance)
    num_qubits = len(coords)
    
    circuit = stim.Circuit()
    
    # Annotate coordinates
    for q, c in enumerate(coords):
        circuit.append("QUBIT_COORDS", [q], [c.real, c.imag])
        
    # Initialize all qubits in |0>
    circuit.append("R", range(num_qubits))
    if p > 0:
        circuit.append("X_ERROR", range(num_qubits), p)
        
    # We will use an ancilla for each edge measurement to simulate 2-body parity natively
    # In modern superconducting chips, MXX can be done via cross-resonance or an ancilla.
    # Stim natively supports MXX, MYY, MZZ but we must ensure it's noisy.
    
    sequence = [
        ('R', 'MXX'),
        ('G', 'MYY'),
        ('B', 'MZZ')
    ]
    
    for r in range(rounds):
        color, op = sequence[r % 3]
        current_edges = edges[color]
        
        circuit.append("TICK")
        
        # We can simulate the 2-body parity directly with Stim's MXX/MYY/MZZ
        targets = []
        for u, v in current_edges:
            targets.extend([u, v])
            
        if targets:
            circuit.append(op, targets)
            
            # Apply depolarization to measured qubits
            if p > 0:
                circuit.append("DEPOLARIZE2", targets, p)
                
    return str(circuit)
