import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, Optional
import numpy as np

from .titans import TitansMemory
from .replay_buffer import ReplayBuffer, Experience


class BaseAgent:
    """Base class for RL agents."""
    def __init__(self):
        pass

    def act(self, state: Any) -> Any:
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

class NestedLearningAgent(BaseAgent):
    """
    Nested Learning Agent for QEC with experience replay and proper RL.

    Implements nested optimization with:
    - Inner loop: Fast updates to decoder weights using policy gradients
    - Outer loop: Slow updates to memory/hyperparameters
    - Experience replay buffer for sample efficiency
    - Actor-critic architecture for stable learning
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        replay_capacity: int = 10000,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # TITANS memory for long-term syndrome history
        self.memory_module = TitansMemory(input_dim=state_dim, hidden_dim=hidden_dim)

        # Actor (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic (value network) for advantage estimation
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Inner loop optimizers (fast)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # Outer loop optimizer (slow)
        self.outer_optimizer = torch.optim.Adam(self.memory_module.parameters(), lr=1e-4)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity, prioritized=True)

        # Training statistics
        self.training_step = 0
        self.episode_rewards = []

    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """Select action using current policy.

        Returns:
            Tuple of (action_index, log_probability)
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)

            context = self.memory_module(state)
            logits = self.actor(context)
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.multinomial(probs, 1).squeeze(-1)

            # Compute log probability
            log_prob = F.log_softmax(logits, dim=-1)
            action_log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return action.item(), action_log_prob

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns.

        Args:
            rewards: Tensor of rewards (batch_size,)
            values: Tensor of value estimates (batch_size,)
            dones: Tensor of done flags (batch_size,)

        Returns:
            Tuple of (advantages, returns)
        """
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Compute GAE advantages
        next_advantage = 0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * (1 - dones[t])
            next_advantage = advantages[t]

        returns = advantages + values

        return advantages, returns

    def inner_loop(
        self,
        batch_size: int = 32,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> Tuple[float, float, float]:
        """Policy gradient update using experience replay.

        Implements PPO-style clipped surrogate objective with GAE advantages.

        Returns:
            Tuple of (actor_loss, critic_loss, entropy)
        """
        if not self.replay_buffer.is_ready(batch_size):
            return 0.0, 0.0, 0.0

        # Sample from replay buffer
        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(batch_size)

        # Prepare states for TITANS (add sequence dimension if needed)
        if states.dim() == 2:
            states = states.unsqueeze(1)
            next_states = next_states.unsqueeze(1)

        # Get memory context
        context = self.memory_module(states.squeeze(1) if states.dim() == 3 else states)

        # Compute current policy logits and values
        logits = self.actor(context)
        values = self.critic(context).squeeze(-1)

        # Compute log probabilities of actions
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute entropy for exploration
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Compute advantages using GAE
        advantages, returns = self.compute_advantages(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO-style clipped surrogate loss (without the ratio for simplicity, using vanilla PG)
        policy_loss = -(action_log_probs * advantages * weights).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns, reduction='none')
        value_loss = (value_loss * weights).mean()

        # Combined loss
        actor_loss = policy_loss - entropy_coef * entropy
        critic_loss = value_coef * value_loss

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Update replay buffer priorities (using TD errors as priorities)
        with torch.no_grad():
            td_errors = torch.abs(returns - values).cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        self.training_step += 1

        return policy_loss.item(), value_loss.item(), entropy.item()

    def outer_loop(self, states: torch.Tensor, performance_metric: float) -> float:
        """Slow update: Update memory representations and hyperparameters.

        Uses meta-learning signal (performance_metric) to adapt the memory module.
        """
        if states.dim() == 2:
            states = states.unsqueeze(1)

        self.outer_optimizer.zero_grad()

        # Get memory context
        context = self.memory_module(states.squeeze(1) if states.dim() == 3 else states)

        # Meta-learning loss: encourage representations that predict performance
        # Use auxiliary task: predict the performance metric from context
        performance_prediction = self.critic(context).squeeze(-1).mean()

        # Loss encourages accurate performance prediction
        meta_loss = F.mse_loss(performance_prediction, torch.tensor(performance_metric))

        # Add regularization to encourage diverse representations
        context_variance = context.var(dim=0).mean()
        diversity_bonus = -0.01 * context_variance  # Encourage variance

        total_loss = meta_loss + diversity_bonus
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.memory_module.parameters(), 0.5)
        self.outer_optimizer.step()

        return total_loss.item()

    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """Store experience in replay buffer with optional priority."""
        experience = Experience(
            state=state if state.dim() > 0 else state.unsqueeze(0),
            action=action,
            reward=reward,
            next_state=next_state if next_state.dim() > 0 else next_state.unsqueeze(0),
            done=done,
        )
        self.replay_buffer.push(experience, priority)

    def get_stats(self) -> dict:
        """Return training statistics."""
        return {
            "training_step": self.training_step,
            "buffer_size": len(self.replay_buffer),
            "episode_rewards": self.episode_rewards[-10:] if self.episode_rewards else [],
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent state to checkpoint."""
        torch.save({
            "memory_module": self.memory_module.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "outer_optimizer": self.outer_optimizer.state_dict(),
            "training_step": self.training_step,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent state from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self.memory_module.load_state_dict(checkpoint["memory_module"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.outer_optimizer.load_state_dict(checkpoint["outer_optimizer"])
        self.training_step = checkpoint.get("training_step", 0)
