"""Experience replay buffer for RL agents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    """Experience replay buffer for off-policy RL algorithms.

    Stores and samples experience tuples (s, a, r, s', done) for training.
    Supports prioritized experience replay with TD-error-based priorities.
    """

    def __init__(self, capacity: int = 10000, prioritized: bool = True, alpha: float = 0.6):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            prioritized: Whether to use prioritized experience replay
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self.priorities: deque[float] = deque(maxlen=capacity)
        self.position = 0

    def push(self, experience: Experience, priority: Optional[float] = None) -> None:
        """Add experience to buffer.

        Args:
            experience: Experience tuple to store
            priority: Priority for prioritized replay (default: max priority)
        """
        self.buffer.append(experience)

        if self.prioritized:
            if priority is None:
                # Use max priority for new experiences
                priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer contains {len(self.buffer)} experiences, need {batch_size}")

        if self.prioritized:
            # Sample based on priorities
            priorities = np.array(self.priorities)
            probs = priorities ** self.alpha
            probs /= probs.sum()

            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

            # Compute importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights /= weights.max()  # Normalize
            weights = torch.FloatTensor(weights)
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            weights = torch.ones(batch_size)

        experiences = [self.buffer[idx] for idx in indices]

        states = torch.stack([e.state for e in experiences])
        
        # Actions might be discrete (Long) or continuous (Float). If they are tensors, stack them.
        # If they are not tensors, convert to tensors.
        if isinstance(experiences[0].action, torch.Tensor):
            actions = torch.stack([e.action for e in experiences])
        else:
            try:
                actions = torch.FloatTensor([e.action for e in experiences])
            except (TypeError, ValueError):
                actions = torch.stack([torch.tensor(e.action) for e in experiences])
                
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences.

        Args:
            indices: Indices of sampled experiences
            priorities: New priority values (typically TD errors)
        """
        if not self.prioritized:
            return

        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return len(self.buffer) >= batch_size
