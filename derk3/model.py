from typing import Tuple, NamedTuple

import torch
from torch.autograd import Function
from torch.nn import (
    Linear,
    Module,
    ReLU,
    Sequential,
    ModuleList,
    LogSoftmax,
    Softplus,
    Tanh,
)
from torch import FloatTensor, Tensor
from torch.distributions import Normal, Categorical

from .distribution import HybridDistribution, MultiDiscreteDistribution


class ScaleGradBackward(Function):
    @staticmethod
    def forward(ctx, input: Tensor, scale: Tensor) -> Tensor:
        ctx.save_for_backward(scale)
        return input

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (scale,) = ctx.saved_tensors
        grad_input = scale * grad_output if ctx.needs_input_grad[0] else None
        return grad_input, None


scale_grad = ScaleGradBackward.apply


class ScaleGrad(Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = torch.as_tensor(scale)

    def forward(self, input: Tensor) -> Tensor:
        return scale_grad(input, self.scale)

    def __repr__(self) -> str:
        return "{}(scale={:.2f})".format(self.__class__.__name__, self.scale.item())


class HybridPolicyModel(Module):
    def __init__(
        self,
        observation_size: int = 64,
        action_size: Tuple[int, int, int] = (2, 4, 8),
        hidden_size: Tuple[int, int, int] = (512, 256, 256),
    ):
        super().__init__()

        # Create shared network for all policy heads
        self.shared = Sequential(
            Linear(observation_size, hidden_size[0]),
            ReLU(),
            Linear(hidden_size[0], hidden_size[1]),
            ReLU(),
        )

        # Policy head for MoveX and Rotate branch (continuous)
        self.move_mu = Sequential(
            Linear(hidden_size[1], hidden_size[2]),
            ReLU(),
            Linear(hidden_size[2], action_size[0]),
        )

        self.move_std = Sequential(
            Linear(hidden_size[1], hidden_size[2]),
            ReLU(),
            Linear(hidden_size[2], action_size[0]),
            Softplus(),
        )

        list(self.move_mu.parameters())[-1].data /= 100
        list(self.move_std.parameters())[-1].data /= 100

        # Policy head for CastSlot branch (discrete)
        self.cast = Sequential(
            Linear(hidden_size[1], hidden_size[2]),
            ReLU(),
            Linear(hidden_size[2], action_size[1]),
            LogSoftmax(dim=-1),
        )

        # Policy head for Focus branch (discrete)
        self.focus = Sequential(
            Linear(hidden_size[1], hidden_size[2]),
            ReLU(),
            Linear(hidden_size[2], action_size[2]),
            LogSoftmax(dim=-1),
        )

    def forward(self, observation: FloatTensor) -> HybridDistribution:
        shared = self.shared(observation)

        move_mu = self.move_mu(shared)

        move_std = self.move_std(shared)
        move = Normal(loc=move_mu, scale=move_std)

        cast = Categorical(logits=self.cast(shared))

        focus = Categorical(logits=self.focus(shared))

        return HybridDistribution(move, cast, focus)


class MultiDiscretePolicyModel(Module):
    def __init__(
        self,
        observation_size: int = 64,
        action_size: Tuple[int, int, int, int] = (9, 9, 4, 8),
        hidden_size: Tuple[int, int, int] = (512, 256, 256),
    ):
        super().__init__()

        self.action_size = action_size


        self.shared = Sequential(
            Linear(observation_size, hidden_size[0]),
            ReLU(),
            Linear(hidden_size[0], hidden_size[1]),
            ReLU(),
        )


        self.policy = ModuleList(
            Sequential(
                Linear(hidden_size[1], hidden_size[2]),
                ReLU(),
                Linear(hidden_size[2], size),
                LogSoftmax(-1),
            )
            for size in action_size
        )

    def forward(self, observation: FloatTensor) -> MultiDiscreteDistribution:
        shared = self.shared(observation)
        return MultiDiscreteDistribution(
            *(Categorical(logits=policy(shared)) for policy in self.policy)
        )


class IndependentValueModel(Module):
    def __init__(
        self,
        observation_size: int = 64,
        hidden_size: Tuple[int, int, int] = (512, 512, 256),
    ):
        super().__init__()

        self.observation_size = observation_size
        self.hidden_size = hidden_size

        self.value = Sequential(
            Linear(self.observation_size, self.hidden_size[0]),
            ReLU(),
            Linear(self.hidden_size[0], self.hidden_size[1]),
            ReLU(),
            Linear(self.hidden_size[1], self.hidden_size[2]),
            ReLU(),
            Linear(self.hidden_size[2], 1),
        )

    def forward(self, observation: FloatTensor) -> FloatTensor:
        return self.value(observation)


class CentralizedValueModel(Module):
    def __init__(
        self,
        observation_size: int = 64,
        hidden_size: Tuple[int, int, int] = (512, 512, 256),
    ):
        super().__init__()

        self.observation_size = observation_size
        self.hidden_size = hidden_size

        self.value = Sequential(
            Linear(3 * self.observation_size, self.hidden_size[0]),
            ReLU(),
            Linear(self.hidden_size[0], self.hidden_size[1]),
            ReLU(),
            Linear(self.hidden_size[1], self.hidden_size[2]),
            ReLU(),
            Linear(self.hidden_size[2], 1),
        )

    def forward(self, observation: FloatTensor) -> FloatTensor:
        observation = observation.flatten(start_dim=-2)
        value =  self.value(observation)
        return value
