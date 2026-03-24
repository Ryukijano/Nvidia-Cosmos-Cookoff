"""State-of-the-art Graph Neural Network (GNN) and Neural Belief Propagation Decoders.

This module implements highly optimized, parallelizable decoders for qLDPC codes
using PyTorch sparse operations. Graph Neural Networks (GNNs) and Neural BP 
represent the bleeding edge of qLDPC decoding research (2024/2025).

References:
- "Decoding Quantum LDPC Codes Using Graph Neural Networks" (2024)
- "Machine learning message-passing for the scalable decoding of QLDPC codes" (Nature, 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class NeuralBPDecoder(nn.Module):
    """Neural Belief Propagation (BP) Decoder for qLDPC codes.
    
    Implements a differentiable BP-like message passing algorithm where the 
    messages are scaled by learnable weights, improving upon standard Min-Sum 
    or Sum-Product algorithms.
    
    Optimized for GPU execution using PyTorch sparse matrix multiplications.
    """
    
    def __init__(self, num_vars: int, num_checks: int, max_iter: int = 15):
        super().__init__()
        self.num_vars = num_vars
        self.num_checks = num_checks
        self.max_iter = max_iter
        
        # Learnable scaling factors for check-to-variable and variable-to-check messages
        self.w_cv = nn.Parameter(torch.ones(1))
        self.w_vc = nn.Parameter(torch.ones(1))
        
        # Learnable damping factor for residual connections
        self.damping = nn.Parameter(torch.tensor(0.5))

    def forward(
        self, 
        syndrome: torch.Tensor, 
        parity_matrix: torch.Tensor, 
        channel_llrs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run Neural BP decoding.
        
        Args:
            syndrome: (batch_size, num_checks) Check node values
            parity_matrix: (num_checks, num_vars) Sparse or dense parity check matrix
            channel_llrs: (batch_size, num_vars) Initial log-likelihood ratios. If None, assumes 0.
            
        Returns:
            (batch_size, num_vars) Predicted bit-flip probabilities
        """
        batch_size = syndrome.size(0)
        device = syndrome.device
        
        if channel_llrs is None:
            channel_llrs = torch.zeros((batch_size, self.num_vars), device=device)
            
        # Ensure parity matrix is sparse for fast O(E) multiplication
        if not parity_matrix.is_sparse:
            parity_matrix = parity_matrix.to_sparse()
            
        H_t = parity_matrix.t() # (num_vars, num_checks)
        
        # Variable node beliefs
        v_beliefs = channel_llrs.clone()
        
        # Map syndrome from {0, 1} to {-1, 1} where -1 is a defect
        s_sign = 1.0 - 2.0 * syndrome
        
        for _ in range(self.max_iter):
            # 1. Variable to Check messages (Approximated as dense matrix multiply for parallel batching)
            # v_to_c = w_vc * H * v_beliefs
            v_to_c = self.w_vc * torch.sparse.mm(parity_matrix, v_beliefs.t()).t()
            
            # 2. Check node processing (Min-Sum approximation with syndrome sign)
            # Use tanh to keep values stable
            c_vals = torch.tanh(v_to_c / 2.0)
            
            # Multiply by syndrome sign to flip parity where there's a defect
            c_msg = s_sign * c_vals
            
            # 3. Check to Variable messages
            c_to_v = self.w_cv * torch.sparse.mm(H_t, c_msg.t()).t()
            
            # 4. Update Variable beliefs with damping
            v_beliefs = self.damping * v_beliefs + (1 - self.damping) * (channel_llrs + c_to_v)
            
        # Convert output LLRs to probabilities using sigmoid
        probabilities = torch.sigmoid(-v_beliefs)
        return probabilities


class GNNLayer(nn.Module):
    """Single layer of the Message Passing Neural Network for qLDPC."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Message processing networks
        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update networks
        self.var_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.chk_update = nn.GRUCell(hidden_dim, hidden_dim)
        
    def forward(
        self, 
        v_feats: torch.Tensor, 
        c_feats: torch.Tensor, 
        H: torch.Tensor, 
        H_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            v_feats: Variable node features (batch, num_vars, hidden_dim)
            c_feats: Check node features (batch, num_checks, hidden_dim)
            H: Sparse parity matrix (num_checks, num_vars)
            H_t: Transposed parity matrix (num_vars, num_checks)
        """
        batch_size = v_feats.size(0)
        hidden_dim = v_feats.size(-1)
        
        # Flatten for sparse MM
        # We need to aggregate messages: H @ v_feats 
        v_flat = v_feats.reshape(-1, hidden_dim) # (batch * num_vars, hidden_dim)
        c_flat = c_feats.reshape(-1, hidden_dim)
        
        # 1. Variable to Check message aggregation
        # To batch sparse.mm, we loop over batch or use block diagonal.
        # For simplicity and speed, we loop if batch is small, or just reshape if H is static.
        v_to_c_msgs = []
        for i in range(batch_size):
            msg = torch.sparse.mm(H, v_feats[i]) # (num_checks, hidden_dim)
            v_to_c_msgs.append(msg)
        v_to_c = torch.stack(v_to_c_msgs)
        v_to_c = self.msg_net(v_to_c)
        
        # 2. Check node update
        c_flat_new = self.chk_update(v_to_c.reshape(-1, hidden_dim), c_flat)
        c_feats = c_flat_new.reshape(batch_size, -1, hidden_dim)
        
        # 3. Check to Variable message aggregation
        c_to_v_msgs = []
        for i in range(batch_size):
            msg = torch.sparse.mm(H_t, c_feats[i])
            c_to_v_msgs.append(msg)
        c_to_v = torch.stack(c_to_v_msgs)
        c_to_v = self.msg_net(c_to_v)
        
        # 4. Variable node update
        v_flat_new = self.var_update(c_to_v.reshape(-1, hidden_dim), v_flat)
        v_feats = v_flat_new.reshape(batch_size, -1, hidden_dim)
        
        return v_feats, c_feats


class qLDPCGNNDecoder(nn.Module):
    """Graph Neural Network Decoder for Quantum LDPC Codes.
    
    Transforms the Tanner graph of the code into a neural message passing network.
    SOTA for highly connected qLDPC codes (like hypergraph product codes) where 
    standard MWPM fails.
    """
    
    def __init__(self, num_vars: int, num_checks: int, hidden_dim: int = 64, num_layers: int = 5):
        super().__init__()
        self.num_vars = num_vars
        self.num_checks = num_checks
        self.hidden_dim = hidden_dim
        
        # Initial embeddings
        self.syndrome_embed = nn.Embedding(2, hidden_dim) # 0 or 1
        self.var_embed = nn.Parameter(torch.randn(1, num_vars, hidden_dim))
        
        self.layers = nn.ModuleList([GNNLayer(hidden_dim) for _ in range(num_layers)])
        
        # Output classification head
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, syndrome: torch.Tensor, parity_matrix: torch.Tensor) -> torch.Tensor:
        """Run GNN decoding.
        
        Args:
            syndrome: (batch_size, num_checks) binary syndrome tensor
            parity_matrix: (num_checks, num_vars) parity matrix
            
        Returns:
            (batch_size, num_vars) logits for each qubit being flipped
        """
        batch_size = syndrome.size(0)
        
        if not parity_matrix.is_sparse:
            parity_matrix = parity_matrix.to_sparse()
        H_t = parity_matrix.t()
        
        # Initialize node features
        # Check nodes get embedding based on syndrome value
        c_feats = self.syndrome_embed(syndrome.long()) # (batch, num_checks, hidden)
        
        # Variable nodes get learned positional embeddings, expanded for batch
        v_feats = self.var_embed.expand(batch_size, -1, -1)
        
        # Message Passing
        for layer in self.layers:
            v_feats, c_feats = layer(v_feats, c_feats, parity_matrix, H_t)
            
        # Final prediction for variable nodes
        logits = self.out_net(v_feats).squeeze(-1) # (batch, num_vars)
        
        return logits

