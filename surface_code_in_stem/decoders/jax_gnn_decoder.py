"""JAX-accelerated Graph Neural Network and Neural Belief Propagation Decoders.

This module provides high-performance JAX implementations of the Neural BP and 
GNN decoders for qLDPC codes. By leveraging `jax.vmap` and `jax.jit`, these
decoders achieve extremely fast parallel execution on GPUs and TPUs.

References:
- "Decoding Quantum LDPC Codes Using Graph Neural Networks" (2024)
- "Machine learning message-passing for the scalable decoding of QLDPC codes" (Nature, 2025)
"""

from typing import Tuple, Optional
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import sparse
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class JAXNeuralBPDecoder:
    """JAX implementation of Neural Belief Propagation for qLDPC codes."""
    
    def __init__(self, num_vars: int, num_checks: int, max_iter: int = 15):
        if not HAS_JAX:
            raise ImportError("JAX is required. Run: pip install jax jaxlib")
            
        self.num_vars = num_vars
        self.num_checks = num_checks
        self.max_iter = max_iter
        
        # Learnable parameters (represented as a dict for JAX functional purity)
        self.init_params = {
            "w_cv": jnp.array([1.0], dtype=jnp.float32),
            "w_vc": jnp.array([1.0], dtype=jnp.float32),
            "damping": jnp.array([0.5], dtype=jnp.float32)
        }
        
        # JIT-compiled decode function
        self._jit_decode = jax.jit(self._decode_step, static_argnames=['num_vars', 'num_checks', 'max_iter'])
        self._jit_batch_decode = jax.jit(
            jax.vmap(self._decode_step, in_axes=(None, 0, None, None, None, None, None)),
            static_argnames=['num_vars', 'num_checks', 'max_iter']
        )

    @staticmethod
    def _decode_step(
        params: dict,
        syndrome: jnp.ndarray,
        H_indices: jnp.ndarray,
        H_values: jnp.ndarray,
        num_vars: int,
        num_checks: int,
        max_iter: int = 15
    ) -> jnp.ndarray:
        """Single-shot Neural BP decoding step for JAX JIT compilation."""
        
        # Construct BCOO sparse matrix
        H = sparse.BCOO((H_values, H_indices), shape=(num_checks, num_vars))
        H_t = sparse.BCOO((H_values, H_indices[:, ::-1]), shape=(num_vars, num_checks))
        
        # Initialize beliefs
        v_beliefs = jnp.zeros(num_vars, dtype=jnp.float32)
        
        # Map syndrome {0, 1} -> {-1, 1}
        s_sign = 1.0 - 2.0 * syndrome
        
        w_cv = params["w_cv"]
        w_vc = params["w_vc"]
        damping = params["damping"]
        
        def bp_iter(i, val):
            v_bel = val
            
            # 1. Variable to Check
            v_to_c = w_vc * (H @ v_bel)
            
            # 2. Check node processing (Min-Sum approx)
            c_vals = jnp.tanh(v_to_c / 2.0)
            c_msg = s_sign * c_vals
            
            # 3. Check to Variable
            c_to_v = w_cv * (H_t @ c_msg)
            
            # 4. Update beliefs with damping
            v_bel_new = damping * v_bel + (1.0 - damping) * c_to_v
            return v_bel_new
            
        final_beliefs = jax.lax.fori_loop(0, max_iter, bp_iter, v_beliefs)
        
        return jax.nn.sigmoid(-final_beliefs)

    def decode_batch(self, params: dict, syndrome: np.ndarray, H_dense: np.ndarray) -> np.ndarray:
        """Run batched decoding.
        
        Args:
            params: Dictionary of Neural BP weights.
            syndrome: (batch_size, num_checks) numpy array.
            H_dense: (num_checks, num_vars) parity matrix.
        """
        # Extract sparse coordinates for JAX
        rows, cols = np.nonzero(H_dense)
        H_indices = jnp.column_stack((rows, cols))
        H_values = jnp.ones(len(rows), dtype=jnp.float32)
        
        syndrome_jax = jnp.array(syndrome, dtype=jnp.float32)
        
        probs = self._jit_batch_decode(
            params, 
            syndrome_jax, 
            H_indices, 
            H_values, 
            self.num_vars, 
            self.num_checks, 
            self.max_iter
        )
        return np.array(probs)
