"""State-of-the-art Reinforcement Learning Agents for Quantum Error Correction.

This module implements two advanced RL algorithms tailored for QEC:
1. Transformer-PPO: Proximal Policy Optimization with a Transformer encoder
   for the highly degenerate, multi-discrete action space of QEC decoding.
2. Continuous SAC: Soft Actor-Critic for the continuous parameter optimization
   of the QEC calibration environment.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli, Normal
import numpy as np
import copy
from typing import Tuple, List, Dict, Any, Optional

from .titans import TitansMemory


# ============================================================================
# 1. Transformer-PPO for Discrete Decoding (MultiBinary Action Space)
# ============================================================================

class TransformerPPOActorCritic(nn.Module):
    """PPO Actor-Critic Network using TITANS memory for syndrome sequence modeling."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # We use the TITANS memory module to encode the syndrome (treating it as a sequence of 1)
        self.memory_module = TitansMemory(
            input_dim=state_dim, 
            hidden_dim=hidden_dim
        )
        
        # Actor network: outputs logits for M independent Bernoulli distributions
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic network: outputs state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get action logits and state value."""
        # Ensure state has sequence dimension for TITANS [batch, seq_len=1, state_dim]
        if state.dim() == 2:
            state = state.unsqueeze(1)
            
        context = self.memory_module(state) # [batch, hidden_dim]
        
        logits = self.actor(context)
        value = self.critic(context)
        
        return logits, value
        
    def get_action_and_value(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute probabilities/values for PPO update."""
        logits, value = self.forward(state)
        
        # Independent Bernoulli distributions for each logical observable
        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs=probs)
        
        if action is None:
            action = dist.sample()
            
        # Log probability of the multi-discrete action is the sum of independent log probs
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)


class PPOAgent:
    """Proximal Policy Optimization agent for QEC Decoding."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        
        self.network = TransformerPPOActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select an action using the current policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_t)
            
        # Squeeze the batch dimension and convert to int for the environment
        # The tensor might have shape [1, 1, num_obs] if a sequence dim got added,
        # or [1, num_obs]. We want it to be [num_obs].
        action_np = action.detach().cpu().numpy().astype(np.int8)
        action_np = action_np.flatten()
            
        return action_np, log_prob.item(), value.item()
        
    def update(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        log_probs: torch.Tensor, 
        returns: torch.Tensor, 
        advantages: torch.Tensor
    ) -> dict:
        """Perform PPO update steps."""
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(self.ppo_epochs):
            _, new_log_probs, entropy, values = self.network.get_action_and_value(states, actions)
            
            # Policy ratio: pi_theta / pi_theta_old
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            value_loss = F.mse_loss(values.squeeze(-1), returns)
            
            # Entropy bonus to encourage exploration
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            
        return {
            "policy_loss": total_policy_loss / self.ppo_epochs,
            "value_loss": total_value_loss / self.ppo_epochs,
            "entropy_loss": total_entropy_loss / self.ppo_epochs
        }


# ============================================================================
# 2. Continuous SAC for Calibration Environment
# ============================================================================

def weights_init_(m):
    """Xavier initialization for SAC networks."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class SACQNetwork(nn.Module):
    """Twin Q-Network for Soft Actor-Critic."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([state, action], 1)
        return self.q1(xu), self.q2(xu)


class SACGaussianPolicy(nn.Module):
    """Gaussian Policy Network for Soft Actor-Critic."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, action_space=None):
        super().__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Enforcing Action Bound
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
        
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)




class FlowMatchingPolicy(nn.Module):
    """
    Generative Flow-Matching (Rectified Flow) Policy.
    
    Transforms base noise N(0, I) into target actions via an ODE.
    Provides a highly expressive, multi-modal policy class for complex 
    quantum calibration landscapes, surpassing standard Gaussian policies.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, action_space=None):
        super().__init__()
        self.action_dim = action_dim
        
        # Velocity field network v_theta(x_t, t, s)
        # Inputs: action x_t, time t, state s
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(weights_init_)
        
        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
                
    def forward(self, state: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity vector field."""
        # Ensure t is broadcasted correctly
        if t.dim() == 0 or t.size(0) != state.size(0):
            t = t.expand(state.size(0), 1)
        inputs = torch.cat([state, xt, t], dim=1)
        return self.net(inputs)
        
    def sample(self, state: torch.Tensor, steps: int = 10, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Euler integration of the learned ODE to generate actions.
        """
        batch_size = state.size(0)
        device = state.device
        
        if deterministic:
            xt = torch.zeros((batch_size, self.action_dim), device=device)
        else:
            xt = torch.randn((batch_size, self.action_dim), device=device)
            
        dt = 1.0 / steps
        
        # Keep track of actions for differentiable graph
        for i in range(steps):
            t = torch.ones((batch_size, 1), device=device) * (i * dt)
            vt = self.forward(state, xt, t)
            xt = xt + vt * dt
            
        # We don't have exact log_prob analytically for flow matching without Hutchinson trace estimator.
        # For RL actor updates, we usually just train via Deterministic Policy Gradient (DPG).
        # So we return a dummy log_prob of 0.
        dummy_log_prob = torch.zeros((batch_size, 1), device=device)
        
        # Apply squashing
        y_t = torch.tanh(xt)
        action = y_t * self.action_scale + self.action_bias
        
        return action, dummy_log_prob, action

    def compute_flow_loss(self, state: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
        """
        Flow matching objective: learn the vector field pointing from noise to target.
        Used if we want to pre-train or regularize with Behavior Cloning.
        """
        batch_size = state.size(0)
        device = state.device
        
        x0 = torch.randn_like(target_action)
        t = torch.rand((batch_size, 1), device=device)
        
        # Linear interpolation
        xt = t * target_action + (1 - t) * x0
        
        # Target velocity is simply (x1 - x0)
        target_v = target_action - x0
        
        pred_v = self.forward(state, xt, t)
        return F.mse_loss(pred_v, target_v)
        
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

class ContinuousSACAgent:
    """Soft Actor-Critic agent for continuous QEC Calibration."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        action_space=None,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr: float = 3e-4,
        hidden_dim: int = 256,
        use_diffusion: bool = False,
        device: str = "cpu"
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device(device)

        self.critic = SACQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.use_diffusion = use_diffusion
        if use_diffusion:
            self.policy = FlowMatchingPolicy(state_dim, action_dim, hidden_dim, action_space).to(device)
        else:
            self.policy = SACGaussianPolicy(state_dim, action_dim, hidden_dim, action_space).to(device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

        # Automatic Entropy Tuning
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(
        self, 
        state_batch: torch.Tensor, 
        action_batch: torch.Tensor, 
        reward_batch: torch.Tensor, 
        next_state_batch: torch.Tensor, 
        mask_batch: torch.Tensor
    ) -> dict:
        """Update SAC networks from a batch of transitions."""
        state_batch = state_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device).unsqueeze(1)
        mask_batch = mask_batch.to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # Critic update
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Actor update
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        if self.use_diffusion:
            policy_loss = -min_qf_pi.mean() # DPG for flow
        else:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Alpha update (Automatic Entropy Tuning)
        if self.use_diffusion:
            alpha_loss = torch.tensor(0.0, device=self.device)
        else:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        self.alpha = self.log_alpha.exp().item()

        # Target network soft update
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            "q_loss": qf_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha
        }
