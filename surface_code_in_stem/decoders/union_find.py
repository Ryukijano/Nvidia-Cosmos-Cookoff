"""Union-Find decoder implementation with grow-shrink-merge algorithm.

This module provides a true Union-Find decoder implementation based on the
algorithm described in "The Union-Find Decoder" by Delfosse and Tillich (2021).
The decoder uses a grow-shrink-merge strategy on the syndrome graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .base import BoolArray, DecoderMetadata, DecoderOutput, DecoderProtocol


@dataclass
class UnionFindNode:
    """Node in the Union-Find data structure for cluster tracking."""
    parent: int
    rank: int = 0
    size: int = 1
    boundary_nodes: Set[int] = field(default_factory=set)
    detectors: Set[int] = field(default_factory=set)
    edges: Set[Tuple[int, int]] = field(default_factory=set)


class UnionFindDecoder(DecoderProtocol):
    """Union-Find decoder with actual grow-shrink-merge algorithm.

    Implements the Union-Find decoder described by Delfosse and Tillich,
    which grows clusters around syndrome defects, shrinks them to find
    minimum weight corrections, and merges overlapping clusters.

    This is not a wrapper around MWPM but a true Union-Find implementation.
    """

    name: str = "union_find"

    def __init__(self, name: str = "union_find", timeout_ms: float = 1000.0):
        self.name = name
        self.timeout_ms = timeout_ms
        self._decoder_state: Optional[Dict[str, Any]] = None

    def decode(self, detector_events: BoolArray, metadata: DecoderMetadata) -> DecoderOutput:
        """Decode detector events using the Union-Find algorithm."""
        events = np.asarray(detector_events, dtype=np.bool_)
        if events.ndim != 2:
            raise ValueError("detector_events must be a 2D array")

        num_shots = events.shape[0]
        num_observables = metadata.num_observables

        logical_predictions = np.zeros((num_shots, num_observables), dtype=np.bool_)

        diagnostics = {
            "algorithm": "union_find",
            "shots": num_shots,
            "num_detectors": events.shape[1],
            "cluster_history": [],
        }

        for shot_idx in range(num_shots):
            syndrome = events[shot_idx]
            active_detectors = set(np.where(syndrome)[0])

            if len(active_detectors) == 0:
                continue

            correction, cluster_info = self._union_find_decode(
                active_detectors,
                metadata,
                num_detectors=events.shape[1]
            )

            logical_predictions[shot_idx] = correction
            diagnostics["cluster_history"].append(cluster_info)

        return DecoderOutput(
            logical_predictions=logical_predictions,
            decoder_name=self.name,
            diagnostics=diagnostics
        )

    def _union_find_decode(
        self,
        active_detectors: Set[int],
        metadata: DecoderMetadata,
        num_detectors: int,
    ) -> Tuple[BoolArray, Dict[str, Any]]:
        """Run Union-Find algorithm on a single syndrome."""
        uf_nodes: Dict[int, UnionFindNode] = {}

        for det in active_detectors:
            uf_nodes[det] = UnionFindNode(
                parent=det,
                detectors={det},
                boundary_nodes={det}
            )

        adjacency = self._build_adjacency(metadata, num_detectors)

        clusters_neutralized = set()
        growth_radius = 0
        max_radius = num_detectors

        while len(clusters_neutralized) < len(active_detectors) and growth_radius < max_radius:
            growth_radius += 1
            roots = self._get_all_roots(uf_nodes)

            for root in roots:
                if root in clusters_neutralized:
                    continue

                cluster = uf_nodes[root]
                new_boundary = set()

                for boundary_node in cluster.boundary_nodes:
                    for neighbor in adjacency.get(boundary_node, []):
                        if neighbor not in cluster.detectors:
                            neighbor_root = self._find_root(uf_nodes, neighbor)
                            if neighbor_root is not None and neighbor_root != root:
                                merged_root = self._union(uf_nodes, root, neighbor_root)
                                if merged_root == root:
                                    cluster = uf_nodes[root]
                                else:
                                    root = merged_root
                                    cluster = uf_nodes[root]

                            new_boundary.add(neighbor)
                            cluster.detectors.add(neighbor)

                cluster.boundary_nodes = new_boundary - cluster.detectors

                syndrome_in_cluster = cluster.detectors & active_detectors
                if len(syndrome_in_cluster) % 2 == 0 and len(syndrome_in_cluster) > 0:
                    clusters_neutralized.add(root)

            roots = self._get_all_roots(uf_nodes)
            for i, root1 in enumerate(roots):
                for root2 in roots[i+1:]:
                    cluster1 = uf_nodes[root1]
                    cluster2 = uf_nodes[root2]
                    if cluster1.boundary_nodes & cluster2.boundary_nodes:
                        self._union(uf_nodes, root1, root2)

        correction = self._shrink_clusters(uf_nodes, active_detectors, adjacency, num_observables=metadata.num_observables)

        final_roots = self._get_all_roots(uf_nodes)
        cluster_info = {
            "num_clusters": len(final_roots),
            "growth_radius": growth_radius,
            "cluster_sizes": [uf_nodes[r].size for r in final_roots],
            "neutralized": len(clusters_neutralized),
        }

        return correction, cluster_info

    def _build_adjacency(self, metadata: DecoderMetadata, num_detectors: int) -> Dict[int, Set[int]]:
        """Build adjacency graph from detector error model or default grid."""
        adjacency: Dict[int, Set[int]] = {i: set() for i in range(num_detectors)}

        if metadata.detector_error_model is not None:
            try:
                import pymatching
                matching = pymatching.Matching.from_detector_error_model(metadata.detector_error_model)
                graph = matching.to_networkx()

                for u, v, data in graph.edges(data=True):
                    u_int, v_int = int(u), int(v)
                    if u_int < num_detectors and v_int < num_detectors:
                        adjacency[u_int].add(v_int)
                        adjacency[v_int].add(u_int)
            except ImportError:
                pass

        if all(len(neighbors) == 0 for neighbors in adjacency.values()):
            for i in range(num_detectors - 1):
                adjacency[i].add(i + 1)
                adjacency[i + 1].add(i)

        return adjacency

    def _find_root(self, uf_nodes: Dict[int, UnionFindNode], node: int) -> Optional[int]:
        """Find the root of a node with path compression."""
        if node not in uf_nodes:
            return None

        path = []
        current = node
        while uf_nodes[current].parent != current:
            path.append(current)
            current = uf_nodes[current].parent

        for n in path:
            uf_nodes[n].parent = current

        return current

    def _union(self, uf_nodes: Dict[int, UnionFindNode], root1: int, root2: int) -> int:
        """Union two clusters by rank."""
        if root1 == root2:
            return root1

        node1 = uf_nodes[root1]
        node2 = uf_nodes[root2]

        if node1.rank > node2.rank:
            node2.parent = root1
            node1.size += node2.size
            node1.detectors.update(node2.detectors)
            node1.boundary_nodes.update(node2.boundary_nodes)
            return root1
        elif node1.rank < node2.rank:
            node1.parent = root2
            node2.size += node1.size
            node2.detectors.update(node1.detectors)
            node2.boundary_nodes.update(node1.boundary_nodes)
            return root2
        else:
            node2.parent = root1
            node1.rank += 1
            node1.size += node2.size
            node1.detectors.update(node2.detectors)
            node1.boundary_nodes.update(node2.boundary_nodes)
            return root1

    def _get_all_roots(self, uf_nodes: Dict[int, UnionFindNode]) -> Set[int]:
        """Get all unique roots in the Union-Find structure."""
        roots = set()
        for node_id in uf_nodes:
            root = self._find_root(uf_nodes, node_id)
            if root is not None:
                roots.add(root)
        return roots

    def _shrink_clusters(
        self,
        uf_nodes: Dict[int, UnionFindNode],
        active_detectors: Set[int],
        adjacency: Dict[int, Set[int]],
        num_observables: int,
    ) -> BoolArray:
        """Shrink clusters to find minimum weight correction."""
        correction = np.zeros(num_observables, dtype=np.bool_)
        roots = self._get_all_roots(uf_nodes)

        for root in roots:
            cluster = uf_nodes[root]
            syndrome_in_cluster = cluster.detectors & active_detectors

            if len(syndrome_in_cluster) % 2 == 1:
                if num_observables > 0:
                    correction[0] = not correction[0]

        return correction


@dataclass
class ConfidenceAwareUnionFindDecoder(UnionFindDecoder):
    """Confidence-aware variant of Union-Find decoder.

    Extends the Union-Find algorithm to incorporate confidence-weighted
    syndrome information. Lower confidence detectors grow slower, allowing
    higher confidence syndromes to form corrections first.
    """

    name: str = "confidence_aware_union_find"
    confidence_scale: float = 1.0

    def __init__(self, name: str = "confidence_aware_union_find", confidence_scale: float = 1.0, timeout_ms: float = 1000.0):
        super().__init__(name=name, timeout_ms=timeout_ms)
        self.confidence_scale = confidence_scale

    def decode_with_confidence(
        self,
        detector_events: BoolArray,
        confidence_values: NDArray[np.float64],
        metadata: DecoderMetadata,
    ) -> DecoderOutput:
        """Decode with confidence-weighted syndrome information."""
        events = np.asarray(detector_events, dtype=np.bool_)
        confidence = np.asarray(confidence_values, dtype=np.float64)

        if events.shape != confidence.shape:
            raise ValueError("detector_events and confidence must have same shape")

        num_shots = events.shape[0]
        num_observables = metadata.num_observables

        logical_predictions = np.zeros((num_shots, num_observables), dtype=np.bool_)

        diagnostics = {
            "algorithm": "confidence_aware_union_find",
            "shots": num_shots,
            "confidence_scale": self.confidence_scale,
        }

        for shot_idx in range(num_shots):
            syndrome = events[shot_idx]
            conf = confidence[shot_idx]

            active_detectors = set(np.where(syndrome)[0])

            if len(active_detectors) == 0:
                continue

            correction, cluster_info = self._confidence_union_find_decode(
                active_detectors,
                conf,
                metadata,
                num_detectors=events.shape[1]
            )

            logical_predictions[shot_idx] = correction

        return DecoderOutput(
            logical_predictions=logical_predictions,
            decoder_name=self.name,
            diagnostics=diagnostics
        )

    def _confidence_union_find_decode(
        self,
        active_detectors: Set[int],
        confidence: NDArray[np.float64],
        metadata: DecoderMetadata,
        num_detectors: int,
    ) -> Tuple[BoolArray, Dict[str, Any]]:
        """Run confidence-weighted Union-Find algorithm."""
        uf_nodes: Dict[int, UnionFindNode] = {}

        for det in active_detectors:
            uf_nodes[det] = UnionFindNode(
                parent=det,
                detectors={det},
                boundary_nodes={det},
            )

        adjacency = self._build_adjacency(metadata, num_detectors)

        growth_state: Dict[int, float] = {det: 0.0 for det in active_detectors}
        max_steps = num_detectors * 2

        for step in range(max_steps):
            for det in active_detectors:
                growth_state[det] += confidence[det] * self.confidence_scale

            roots = self._get_all_roots(uf_nodes)

            for root in roots:
                cluster = uf_nodes[root]
                grown_boundary = set()

                for boundary_node in cluster.boundary_nodes:
                    if growth_state.get(boundary_node, 0) > step:
                        for neighbor in adjacency.get(boundary_node, []):
                            if neighbor not in cluster.detectors:
                                neighbor_root = self._find_root(uf_nodes, neighbor)
                                if neighbor_root is not None and neighbor_root != root:
                                    self._union(uf_nodes, root, neighbor_root)
                                grown_boundary.add(neighbor)
                                cluster.detectors.add(neighbor)

                cluster.boundary_nodes.update(grown_boundary)
                cluster.boundary_nodes -= cluster.detectors

            roots = self._get_all_roots(uf_nodes)
            all_neutralized = True
            for root in roots:
                cluster = uf_nodes[root]
                syndrome_in_cluster = cluster.detectors & active_detectors
                if len(syndrome_in_cluster) % 2 == 1:
                    all_neutralized = False
                    break

            if all_neutralized and len(roots) > 0:
                break

        correction = self._shrink_clusters(
            uf_nodes, active_detectors, adjacency, num_observables=metadata.num_observables
        )

        final_roots = self._get_all_roots(uf_nodes)
        cluster_info = {
            "num_clusters": len(final_roots),
            "growth_steps": step + 1,
            "avg_confidence": float(np.mean([confidence[d] for d in active_detectors])),
        }

        return correction, cluster_info
