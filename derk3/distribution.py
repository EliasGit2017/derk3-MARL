from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.distributions import Normal, Categorical


class Distribution(ABC):
    @abstractmethod
    def sample(self, deterministic: bool = False) -> Tensor:
        ...

    @abstractmethod
    def mean_entropy(self) -> Tensor:
        ...

    def log_likelihood(
        self, movex: Tensor, rotate: Tensor, castslot: Tensor, focus: Tensor
    ) -> Tensor:
        ...


class HybridDistribution(Distribution):
    movex_rotate_dist: Normal
    cast_dist: Categorical
    focus_dist: Categorical

    def __init__(
        self, movex_rotate: Normal, cast: Categorical, focus: Categorical
    ) -> None:
        self.movex_rotate_dist = movex_rotate
        self.cast_dist = cast
        self.focus_dist = focus

    def sample(self, deterministic: bool = False):
        if deterministic:
            move = self.movex_rotate_dist.mean
            cast = self.cast_dist.probs.argmax(-1)
            focus = self.focus_dist.probs.argmax(-1)
        else:
            move = self.movex_rotate_dist.sample()
            cast = self.cast_dist.sample()
            focus = self.focus_dist.sample()
        return move, cast, focus

    def mean_entropy(self):
        joint_independent_entropy = (
            self.movex_rotate_dist.entropy().sum(-1)
            + self.cast_dist.entropy()
            + self.focus_dist.entropy()
        )
        return torch.mean(joint_independent_entropy / 4)

    def log_likelihood(
        self, movex: Tensor, rotate: Tensor, cast: Tensor, focus: Tensor
    ):
        movex_rotate = torch.stack([movex, rotate], dim=-1)
        return torch.cat(
            [
                self.movex_rotate_dist.log_prob(movex_rotate),
                self.cast_dist.log_prob(cast).unsqueeze(-1),
                self.focus_dist.log_prob(focus).unsqueeze(-1),
            ],
            dim=-1,
        )


class MultiDiscreteDistribution:
    movex_dist: Categorical
    rotate_dist: Categorical
    castslot_dist: Categorical
    focus_dist: Categorical

    def __init__(
        self,
        movex: Categorical,
        rotate: Categorical,
        castslot: Categorical,
        focus: Categorical,
    ) -> None:
        self.movex_dist = movex
        self.rotate_dist = rotate
        self.castslot_dist = castslot
        self.focus_dist = focus

    def sample(self, deterministic: bool = False):
        if deterministic:
            movex = self.movex_dist.probs.argmax(-1)
            rotate = self.rotate_dist.probs.argmax(-1)
            castslot = self.castslot_dist.probs.argmax(-1)
            focus = self.focus_dist.probs.argmax(-1)
        else:
            movex = self.movex_dist.sample()
            rotate = self.rotate_dist.sample()
            castslot = self.castslot_dist.sample()
            focus = self.focus_dist.sample()
        return movex, rotate, castslot, focus

    def mean_entropy(self):
        joint_independent_entropy = (
            self.movex_dist.entropy()
            + self.rotate_dist.entropy()
            + self.castslot_dist.entropy()
            + self.focus_dist.entropy()
        )
        return torch.mean(joint_independent_entropy / 4)

    def log_likelihood(
        self, movex: Tensor, rotate: Tensor, castslot: Tensor, focus: Tensor
    ):
        return torch.stack(
            [
                self.movex_dist.log_prob(movex),
                self.rotate_dist.log_prob(rotate),
                self.castslot_dist.log_prob(castslot),
                self.focus_dist.log_prob(focus),
            ],
            dim=-1,
        )
