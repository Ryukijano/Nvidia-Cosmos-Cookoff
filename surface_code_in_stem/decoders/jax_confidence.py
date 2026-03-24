"""JAX-accelerated confidence-aware decoding with Flash Attention.

This module provides high-performance implementations of confidence-weighted
decoding using JAX for GPU/TPU acceleration and Flash Attention for
efficient syndrome sequence modeling.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# Optional JAX imports with graceful fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

# Optional Flash Attention
try:
    import flash_attn
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_func = None


class JAXConfidenceDecoder:
    """Vectorized confidence-aware decoder using JAX acceleration.

    Implements fast confidence-weighted decoding with:
    - JAX JIT compilation for GPU acceleration
    - Vectorized batch processing
    - Flash Attention for syndrome sequence attention
    """

    def __init__(self, confidence_scale: float = 1.0, use_flash_attn: bool = True):
        """Initialize JAX confidence decoder.

        Args:
            confidence_scale: Scaling factor for confidence weighting
            use_flash_attn: Whether to use Flash Attention if available
        """
        if not HAS_JAX:
            raise ImportError("JAX is required for JAXConfidenceDecoder. Install with: pip install jax jaxlib")

        self.confidence_scale = confidence_scale
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN

        # JIT compile core functions
        self._jit_compute_weights = jit(self._compute_adjusted_weights)
        self._jit_batch_decode = jit(self._batch_decode)

    def _compute_adjusted_weights(
        self,
        base_weights: jnp.ndarray,
        shot_confidence: jnp.ndarray,
        edge_u: jnp.ndarray,
        edge_v: jnp.ndarray,
        u_is_det: jnp.ndarray,
        v_is_det: jnp.ndarray,
    ) -> jnp.ndarray:
        """Vectorized confidence weight adjustment (JAX).

        Args:
            base_weights: Base edge weights (num_edges,)
            shot_confidence: Per-detector confidence values (num_detectors,)
            edge_u: Edge endpoint u indices (num_edges,)
            edge_v: Edge endpoint v indices (num_edges,)
            u_is_det: Boolean mask for u being real detector (num_edges,)
            v_is_det: Boolean mask for v being real detector (num_edges,)

        Returns:
            Adjusted edge weights (num_edges,)
        """
        # Safe index lookup (boundary is replaced with 0)
        conf_u = jnp.where(u_is_det, shot_confidence[edge_u], 1.0)
        conf_v = jnp.where(v_is_det, shot_confidence[edge_v], 1.0)

        # Count real endpoints
        num_real_endpoints = u_is_det.astype(jnp.float32) + v_is_det.astype(jnp.float32)

        # Compute mean confidence (avoid division by zero)
        mean_conf = jnp.where(
            num_real_endpoints > 0,
            (conf_u * u_is_det + conf_v * v_is_det) / jnp.maximum(num_real_endpoints, 1.0),
            1.0
        )

        # Scale weights inversely with confidence
        scale = 1.0 + self.confidence_scale * (1.0 - mean_conf)

        return base_weights * scale

    def _batch_decode(
        self,
        hard_bits: jnp.ndarray,
        confidence: jnp.ndarray,
        base_weights: jnp.ndarray,
        edge_u: jnp.ndarray,
        edge_v: jnp.ndarray,
        u_is_det: jnp.ndarray,
        v_is_det: jnp.ndarray,
    ) -> jnp.ndarray:
        """Decode a batch of syndromes with confidence.

        Args:
            hard_bits: Binary detector outcomes (batch_size, num_detectors)
            confidence: Confidence values (batch_size, num_detectors)
            base_weights: Base edge weights (num_edges,)
            edge_u, edge_v: Edge endpoint indices (num_edges,)
            u_is_det, v_is_det: Boolean masks (num_edges,)

        Returns:
            Adjusted weights for each shot (batch_size, num_edges)
        """
        # Vectorize over batch dimension
        def decode_single(hard, conf):
            return self._compute_adjusted_weights(
                base_weights, conf, edge_u, edge_v, u_is_det, v_is_det
            )

        # Apply to each shot in batch
        return vmap(decode_single, in_axes=(0, 0))(hard_bits, confidence)

    def decode_batch(
        self,
        hard_bits: NDArray[np.float64],
        confidence: NDArray[np.float64],
        base_weights: NDArray[np.float64],
        edge_u: NDArray[np.int64],
        edge_v: NDArray[np.int64],
        u_is_det: NDArray[np.bool_],
        v_is_det: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Decode batch with JAX acceleration.

        Args:
            hard_bits: (batch_size, num_detectors) binary outcomes
            confidence: (batch_size, num_detectors) confidence values
            base_weights: (num_edges,) base edge weights
            edge_u, edge_v: (num_edges,) edge endpoint indices
            u_is_det, v_is_det: (num_edges,) boolean masks

        Returns:
            (batch_size, num_edges) adjusted weights
        """
        # Convert to JAX arrays
        hard_bits_jax = jnp.array(hard_bits, dtype=jnp.float32)
        confidence_jax = jnp.array(confidence, dtype=jnp.float32)
        base_weights_jax = jnp.array(base_weights, dtype=jnp.float32)
        edge_u_jax = jnp.array(edge_u, dtype=jnp.int32)
        edge_v_jax = jnp.array(edge_v, dtype=jnp.int32)
        u_is_det_jax = jnp.array(u_is_det, dtype=jnp.bool_)
        v_is_det_jax = jnp.array(v_is_det, dtype=jnp.bool_)

        # Run JIT-compiled decode
        adjusted_weights = self._jit_batch_decode(
            hard_bits_jax,
            confidence_jax,
            base_weights_jax,
            edge_u_jax,
            edge_v_jax,
            u_is_det_jax,
            v_is_det_jax,
        )

        # Convert back to numpy
        return np.array(adjusted_weights)


class FlashAttentionSyndromeModel:
    """Syndrome sequence model using Flash Attention.

    Uses Flash Attention for O(n) memory and O(n²) compute attention
    on syndrome sequences, enabling efficient modeling of long-range
dependencies in syndrome histories.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize Flash Attention syndrome model.

        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        if not HAS_JAX and not HAS_FLASH_ATTN:
            raise ImportError(
                "FlashAttentionSyndromeModel requires JAX and Flash Attention. "
                "Install with: pip install jax jaxlib flash-attn"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Initialize weights (would typically be loaded from trained model)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        if not HAS_JAX:
            return

        key = jax.random.PRNGKey(0)

        # Q, K, V projection weights
        key, *subkeys = jax.random.split(key, 4)
        self.w_q = jax.random.normal(subkeys[0], (self.embed_dim, self.embed_dim)) * 0.02
        self.w_k = jax.random.normal(subkeys[1], (self.embed_dim, self.embed_dim)) * 0.02
        self.w_v = jax.random.normal(subkeys[2], (self.embed_dim, self.embed_dim)) * 0.02

        # Output projection
        self.w_o = jax.random.normal(subkeys[3], (self.embed_dim, self.embed_dim)) * 0.02

        # Layer norms
        self.ln1_scale = jnp.ones(self.embed_dim)
        self.ln1_bias = jnp.zeros(self.embed_dim)
        self.ln2_scale = jnp.ones(self.embed_dim)
        self.ln2_bias = jnp.zeros(self.embed_dim)

    def _layer_norm(self, x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
        """Layer normalization."""
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + 1e-5) * scale + bias

    def _apply_attention(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Apply Flash Attention.

        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Attention output (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = jnp.dot(x, self.w_q.T)  # (batch, seq, embed)
        k = jnp.dot(x, self.w_k.T)
        v = jnp.dot(x, self.w_v.T)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        if HAS_FLASH_ATTN and self.use_flash_attn:
            # Use Flash Attention for O(n) memory efficiency
            # Flash Attention expects: (batch, seq, nheads, headdim)
            out = flash_attn_func(q, k, v, dropout_p=self.dropout, causal=False)
        else:
            # Fallback to standard attention
            # Transpose for attention computation
            q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, nheads, seq, headdim)
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            # Compute attention scores
            scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)

            if mask is not None:
                scores = jnp.where(mask, scores, -1e9)

            attn_weights = jax.nn.softmax(scores, axis=-1)
            attn_weights = jax.nn.dropout(attn_weights, self.dropout, key=jax.random.PRNGKey(0))

            out = jnp.matmul(attn_weights, v)
            out = jnp.transpose(out, (0, 2, 1, 3))  # (batch, seq, nheads, headdim)

        # Reshape back
        out = out.reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        out = jnp.dot(out, self.w_o.T)

        return out

    def forward(
        self,
        syndrome_sequence: NDArray[np.float64],
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through Flash Attention syndrome model.

        Args:
            syndrome_sequence: (batch_size, seq_len, embed_dim) syndrome embeddings
            return_attention: Whether to return attention weights

        Returns:
            Dict with 'output' and optionally 'attention_weights'
        """
        if not HAS_JAX:
            raise RuntimeError("JAX is required for Flash Attention model")

        # Convert to JAX array
        x = jnp.array(syndrome_sequence, dtype=jnp.float32)

        # Layer norm + attention + residual
        residual = x
        x = self._layer_norm(x, self.ln1_scale, self.ln1_bias)
        x = self._apply_attention(x)
        x = x + residual

        # Layer norm + FFN + residual
        residual = x
        x = self._layer_norm(x, self.ln2_scale, self.ln2_bias)
        x = jax.nn.gelu(x)  # GELU activation
        x = x + residual

        output = np.array(x)

        result = {"output": output}

        return result

    def encode_syndrome_history(
        self,
        syndrome_history: NDArray[np.bool_],
        window_size: int = 10,
    ) -> NDArray[np.float64]:
        """Encode syndrome history using Flash Attention.

        Args:
            syndrome_history: (batch_size, total_rounds, num_detectors) syndrome bits
            window_size: Number of rounds to attend over

        Returns:
            (batch_size, window_size, embed_dim) encoded representations
        """
        if not HAS_JAX:
            raise RuntimeError("JAX is required for syndrome encoding")

        batch_size, total_rounds, num_detectors = syndrome_history.shape

        # Truncate or pad to window_size
        if total_rounds > window_size:
            syndrome_history = syndrome_history[:, -window_size:, :]
        elif total_rounds < window_size:
            pad_width = ((0, 0), (0, window_size - total_rounds), (0, 0))
            syndrome_history = np.pad(syndrome_history, pad_width, mode='constant')

        # Simple embedding: convert binary to float and project
        syndrome_float = syndrome_history.astype(np.float32)

        # Pad detector dimension to match embed_dim if needed
        if num_detectors < self.embed_dim:
            pad_width = ((0, 0), (0, 0), (0, self.embed_dim - num_detectors))
            syndrome_float = np.pad(syndrome_float, pad_width, mode='constant')
        else:
            # Truncate or use learned projection
            syndrome_float = syndrome_float[:, :, :self.embed_dim]

        # Apply Flash Attention model
        result = self.forward(syndrome_float)

        return result["output"]


def create_fast_confidence_decoder(
    confidence_scale: float = 1.0,
    use_jax: bool = True,
) -> Any:
    """Factory function to create optimal confidence decoder.

    Args:
        confidence_scale: Confidence scaling factor
        use_jax: Whether to use JAX acceleration if available

    Returns:
        Decoder instance (JAX or numpy fallback)
    """
    if use_jax and HAS_JAX:
        return JAXConfidenceDecoder(confidence_scale=confidence_scale)
    else:
        # Return numpy-based fallback
        from ..confidence_decoding import WeightedMWPMDecoder
        return WeightedMWPMDecoder
