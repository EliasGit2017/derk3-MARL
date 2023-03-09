from pathlib import Path

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from gym.spaces import Box, Discrete
from gym_derk import ActionKeys

from .parameters import Parameters
from .model import MultiDiscreteDistribution, MultiDiscretePolicyModel


class Agent:
    def __init__(
        self,
        parameters: Parameters,
        action_space: Tuple[Box, Box, Box, Discrete, Discrete],
        observation_space: Union[Box, int] = 64,
        device_type: Optional[str] = None,
    ) -> None:
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

        # Instantiate policy module
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
            (observation_size,) = observation_space.shape
        else:
            observation_size = observation_space

        if device_type is None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.policy = MultiDiscretePolicyModel(
            observation_size, action_size, self.params.hidden_size[0]
        )
        self.policy.to(self.device)
        self.policy.eval()

        for param in self.policy.parameters():
            param.requires_grad = False

    def reset(self) -> None:
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
    ) -> np.ndarray:
        """"""
        observation = torch.as_tensor(
            observation, dtype=torch.float, device=self.device,
        )

        policy: MultiDiscreteDistribution = self.policy(observation)

        # Sample actions from policies
        movex, rotate, castslot, focus = policy.sample(deterministic)

        # Convert the sampled actions
        action = self.convert_action(movex, rotate, castslot, focus)

        return action

    def load(self, path: Path, device_type: Optional[str] = None) -> None:
        if device_type is None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
        checkpoint = torch.load(path, map_location=device)
        self.restore(checkpoint)

    def restore(self, checkpoint: Dict) -> None:
        self.params = checkpoint["params"]
        self.policy.load_state_dict(checkpoint["policy"])
