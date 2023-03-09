from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import FloatTensor, Tensor
from torch.optim import Adam
from gym_derk import ActionKeys
from gym.spaces import Box, Discrete
from scipy.signal import lfilter

from .diagnostics import Diagnostics
from .parameters import Parameters
from .model import (
    MultiDiscreteDistribution,
    MultiDiscretePolicyModel,
    CentralizedValueModel,
    IndependentValueModel,
)

class Batch(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    log_likelihood: np.ndarray
    rewards: np.ndarray


class MiniBatch(NamedTuple):
    observations: FloatTensor
    actions: FloatTensor
    log_likelihood: FloatTensor
    returns: FloatTensor
    advantages: FloatTensor


def discounted_return(reward: np.ndarray, discount: float = 0.99) -> np.ndarray:
    """
    Computes the discounted return of a trajectory using a single coefficient
    IIR filter.

    R_t = r_t+1 + γ r_t+2 + γ^2 r_t+3 + ···
        = 𝚺_k γ^k r_t+k+1
        = r_t+1 + γ R_t+1
    """
    return lfilter([1], [1, -discount], reward[::-1], axis=0)[::-1]


def clipped_surrogate(
    policy: MultiDiscretePolicyModel,
    batch: MiniBatch,
    epsilon: float = 0.1,
) -> FloatTensor:
    """
    Computes the PPO clipped surrogate function.

    L(θ) = 𝔼[min(π(a|s) / π_old(a|s) A(s,a), clip(π(a|s) / π_old(a|s), 1 - ε, 1 + ε) A(s,a))]
    """
    # Compute new policy probabilities for each action branch
    log_likelihood = policy.log_likelihood(
        batch.actions[..., 0],
        batch.actions[..., 1],
        batch.actions[..., 3],
        batch.actions[..., 4],
    )

    # Compute clipped surrogate function for each action branch
    likelihood_ratio = torch.exp(log_likelihood - batch.log_likelihood)
    clipped_likelihood_ratio = torch.clamp(likelihood_ratio, 1 - epsilon, 1 + epsilon)
    clipped_surrogate = torch.min(
        likelihood_ratio * batch.advantages, clipped_likelihood_ratio * batch.advantages
    )

    # Optimize each action branch independently
    clipped_surrogate = clipped_surrogate.sum(-1)

    # Appoximate gradient by averaging over batch
    clipped_surrogate = clipped_surrogate.mean()

    return clipped_surrogate


def minibatch_iterator(
    minibatches: int,
    observations: FloatTensor,
    actions: Tensor,
    log_likelihood,
    returns: FloatTensor,
    advantages: FloatTensor,
) -> Iterator[MiniBatch]:
    T, B, *_ = observations.size()
    batch_size = T * B
    mini_batch_size = batch_size // minibatches
    assert batch_size % mini_batch_size == 0
    indices = np.random.permutation(batch_size)
    indices = np.split(
        indices[: batch_size // mini_batch_size * mini_batch_size], minibatches
    )
    for index in indices:
        t_index = index % T
        b_index = index // T
        minibatch = MiniBatch(
            observations[t_index, b_index],
            actions[t_index, b_index],
            log_likelihood[t_index, b_index],
            returns[t_index, b_index],
            advantages[t_index, b_index],
        )
        yield minibatch


class Trainer:
    """A Dr. Derk's Mutant Battlegrounds independent PPO trainer"""

    num_agents_per_team: int = 3

    def __init__(
        self,
        parameters: Parameters,
        action_space: Tuple[Box, Box, Box, Discrete, Discrete],
        observation_space: Union[Box, int] = 64,
        device_type: Optional[str] = None,
    ) -> None:
        """"""

        self.params = parameters

        # Create a uniform grid for the discretization of continuous actions
        self.discretization_grid = {
            action.name: np.linspace(
                action_space[action.value].low,
                action_space[action.value].high,
                self.params.discretization_bins,
            )
            for action in (ActionKeys.MoveX, ActionKeys.Rotate)
        }

        # Instantiate policy module and optimizer
        action_size = []
        for space, action in zip(action_space, ActionKeys):
            if action is ActionKeys.MoveX or action is ActionKeys.Rotate:
                size = self.params.discretization_bins
            elif action is ActionKeys.ChaseFocus:
                continue
            else:
                size = space.n
            action_size.append(size)

        if isinstance(observation_space, Box):
            (self.observation_size,) = observation_space.shape
        else:
            self.observation_size = observation_space

        if device_type is None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.policy = MultiDiscretePolicyModel(
            self.observation_size, action_size, self.params.hidden_size[0]
        )
        self.policy.to(self.device)
        self.policy.eval()

        self.value = IndependentValueModel(
            self.observation_size, self.params.hidden_size[1]
        )
        self.value.to(self.device)
        self.value.eval()

        self.policy_optimizer = Adam(
            self.policy.parameters(),
            lr=self.params.policy_learning_rate,
        )

        self.value_optimizer = Adam(
            self.value.parameters(),
            lr=self.params.value_learning_rate,
        )

    def reset(self):
        pass

    def convert_action(
        self, movex: Tensor, rotate: Tensor, castslot: Tensor, focus: Tensor
    ):
        action = [
            movex.cpu().numpy(),
            rotate.cpu().numpy(),
            np.zeros((rotate.size(0),), dtype=np.int64),
            castslot.cpu().numpy(),
            focus.cpu().numpy(),
        ]

        for action_key in (ActionKeys.MoveX, ActionKeys.Rotate):
            action[action_key.value] = self.discretization_grid[action_key.name][
                action[action_key.value]
            ]

        return np.stack(action, axis=-1)

    @torch.no_grad()
    def act(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        observation = torch.as_tensor(
            observation, dtype=torch.float, device=self.device
        )

        policy: MultiDiscreteDistribution = self.policy(observation)

        movex, rotate, castslot, focus = policy.sample(deterministic)

        log_likelihood = policy.log_likelihood(movex, rotate, castslot, focus)

        action = self.convert_action(movex, rotate, castslot, focus)

        return action, log_likelihood.cpu().numpy()

    def step(self, batch: Batch) -> Diagnostics:
        """
        L(θ) = 𝔼[min(π(a|s) / π_old(a|s) A(s,a), clip(π(a|s) / π_old(a|s), 1 - ε, 1 + ε) A(s,a))]
        """

        observations = torch.as_tensor(
            batch.observations,
            dtype=torch.float,
            device=self.device,
        )

        actions = batch.actions.astype(np.int64)
        for action_key in (ActionKeys.MoveX, ActionKeys.Rotate):
            actions[..., action_key.value] = np.int64(
                np.digitize(
                    batch.actions[..., action_key.value],
                    self.discretization_grid[action_key.name][1:],
                )
            )
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)

        log_likelihood = torch.as_tensor(
            batch.log_likelihood, dtype=torch.float, device=self.device
        )

        # Compute discounted returns and transfer to compute device
        returns = discounted_return(batch.rewards, self.params.gamma)
        returns = torch.as_tensor(
            np.ascontiguousarray(returns), dtype=torch.float, device=self.device
        )

        # View all tensors as batches of teams
        observations = observations.view(
            observations.size(0), -1, self.num_agents_per_team, self.observation_size
        )
        actions = actions.view(
            actions.size(0), -1, self.num_agents_per_team, actions.size(-1)
        )
        log_likelihood = log_likelihood.view(
            log_likelihood.size(0),
            -1,
            self.num_agents_per_team,
            log_likelihood.size(-1),
        )
        returns = returns.view(returns.size(0), -1, self.num_agents_per_team, 1)

        # Compute advantages by subtracting baseline from discounted returns
        baseline: FloatTensor = self.value(observations)
        advantages = returns - baseline.detach()

        # Batch normalize advantages
        advantages -= advantages.mean()
        advantages /= advantages.std() + torch.finfo(torch.float32).eps

        # Keep track of losses and latest policy for diagnostics
        policy_losses = []
        value_losses = []
        policy: Optional[MultiDiscreteDistribution] = None

        for epoch in range(self.params.epochs):
            for minibatch in minibatch_iterator(
                self.params.minibatches,
                observations,
                actions,
                log_likelihood,
                returns,
                advantages,
            ):
                policy: MultiDiscreteDistribution = self.policy(minibatch.observations)

                # Compute PPO clipped surrogate loss
                policy_loss = -clipped_surrogate(policy, minibatch, self.params.epsilon)

                # Add entropy regularization
                policy_loss -= self.params.beta * policy.mean_entropy()

                policy_losses.append(policy_loss)

                # Update policy parameters
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Update value parameters
                value: Tensor = self.value(minibatch.observations)

                value_loss = 0.5 * F.mse_loss(value, minibatch.returns)

                value_losses.append(value_loss)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

        diagnostics = Diagnostics()
        if self.params.debug:
            diagnostics.policy_loss = sum(
                policy_loss.item() for policy_loss in policy_losses
            ) / len(policy_losses)
            diagnostics.value_loss = sum(
                value_loss.item() for value_loss in value_losses
            ) / len(value_losses)
            diagnostics.movex_entropy = policy.movex_dist.entropy().mean().item()
            diagnostics.rotate_entropy = policy.rotate_dist.entropy().mean().item()
            diagnostics.castslot_entropy = policy.castslot_dist.entropy().mean().item()
            diagnostics.focus_entropy = policy.focus_dist.entropy().mean().item()

        return diagnostics

    def save(self, path: Path) -> None:
        checkpoint = {
            "params": self.params,
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path)
        self.params = checkpoint["params"]
        self.policy.load_state_dict(checkpoint["policy"])
        self.value.load_state_dict(checkpoint["value"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
