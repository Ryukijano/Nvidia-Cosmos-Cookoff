"""Sparse-graph MWPM adapter with graph-pruning hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import BoolArray, DecoderMetadata, DecoderOutput, DecoderProtocol
from .mwpm import MWPMDecoder

GraphPruner = Callable[[Any], Any]


@dataclass
class SparseBlossomDecoder(DecoderProtocol):
    """Sparse Blossom style decoder wrapper around MWPM infrastructure."""

    graph_pruner: GraphPruner | None = None
    name: str = "sparse_blossom"

    def __post_init__(self) -> None:
        self._delegate = MWPMDecoder(name=self.name)

    def _maybe_prune_metadata(self, metadata: DecoderMetadata) -> DecoderMetadata:
        if self.graph_pruner is None or metadata.detector_error_model is None:
            return metadata

        pruned_dem = self.graph_pruner(metadata.detector_error_model)
        return DecoderMetadata(
            num_observables=metadata.num_observables,
            detector_error_model=pruned_dem,
            circuit=metadata.circuit,
            seed=metadata.seed,
            extra=metadata.extra,
        )

    def decode(self, detector_events: BoolArray, metadata: DecoderMetadata) -> DecoderOutput:
        pruned_metadata = self._maybe_prune_metadata(metadata)
        output = self._delegate.decode(detector_events=detector_events, metadata=pruned_metadata)
        diagnostics = dict(output.diagnostics)
        diagnostics["graph_pruned"] = self.graph_pruner is not None
        return DecoderOutput(
            logical_predictions=output.logical_predictions,
            decoder_name=self.name,
            diagnostics=diagnostics,
        )
