import torch
import torch.nn as nn

class NeuralMemory(nn.Module):
    """
    Neural Memory module based on TITANS (Neural Long-Term Memory).
    Learns to memorize historical context using a recurrent update rule.
    """
    def __init__(self, embed_dim: int, memory_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        
        # Projections for Query, Key, Value
        self.proj_q = nn.Linear(embed_dim, memory_dim)
        self.proj_k = nn.Linear(embed_dim, memory_dim)
        self.proj_v = nn.Linear(embed_dim, memory_dim)
        
        # Learnable update rate (alpha)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # Persistent memory state for step-by-step updates
        self.register_buffer('persistent_memory', None)

    def forward(self, x: torch.Tensor, memory_state: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass processing a sequence or a single step.
        x: (batch_size, seq_len, embed_dim) or (batch_size, embed_dim)
        Returns: (retrieved_memory, updated_memory_state)
        """
        is_3d = x.dim() == 3
        if not is_3d:
            x = x.unsqueeze(1)  # (batch_size, 1, embed_dim)
            
        batch_size, seq_len, _ = x.shape
        
        if memory_state is None:
            memory_state = torch.zeros(batch_size, self.memory_dim, self.memory_dim, device=x.device)
            
        outputs = []
        alpha_val = torch.sigmoid(self.alpha)
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, embed_dim)
            
            q_t = self.proj_q(x_t)  # (batch_size, memory_dim)
            k_t = self.proj_k(x_t)  # (batch_size, memory_dim)
            v_t = self.proj_v(x_t)  # (batch_size, memory_dim)
            
            # Retrieve from memory: q_t * M_{t-1}
            # q_t: (batch_size, 1, memory_dim)
            # memory_state: (batch_size, memory_dim, memory_dim)
            retrieved = torch.bmm(q_t.unsqueeze(1), memory_state).squeeze(1)  # (batch_size, memory_dim)
            outputs.append(retrieved)
            
            # Update memory: M_t = (1 - alpha) M_{t-1} + alpha (k_t \otimes v_t)
            kv_t = torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))  # (batch_size, memory_dim, memory_dim)
            memory_state = (1 - alpha_val) * memory_state + alpha_val * kv_t
            
        out = torch.stack(outputs, dim=1)  # (batch_size, seq_len, memory_dim)
        
        if not is_3d:
            out = out.squeeze(1)
            
        return out, memory_state

    def update_memory(self, new_key: torch.Tensor, new_value: torch.Tensor):
        """
        Compatibility method for NestedLearningAgent.
        In the recurrent formulation, memory is updated during the forward pass.
        """
        pass

    def reset_memory(self):
        """Reset the persistent memory state."""
        self.persistent_memory = None


class TitansMemory(nn.Module):
    """
    Wrapper module for TITANS memory architecture.
    Combines short-term processing with long-term NeuralMemory using a Gating Mechanism.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.neural_memory = NeuralMemory(embed_dim=hidden_dim, memory_dim=hidden_dim)
        
        # Gating Mechanism to combine Short-term and Long-term memory
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        """
        syndrome: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        """
        # Encode current syndrome (Short-term context)
        encoded = torch.relu(self.encoder(syndrome))
        
        # Retrieve from long-term memory
        memory_context, _ = self.neural_memory(encoded)
        
        # Gating Mechanism: Combine short-term and long-term context
        gate_val = torch.sigmoid(self.gate(torch.cat([encoded, memory_context], dim=-1)))
        combined = gate_val * encoded + (1 - gate_val) * memory_context
        
        output = torch.relu(self.decoder(combined))
        
        return output
