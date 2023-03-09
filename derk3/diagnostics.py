from typing import Optional
from dataclasses import dataclass


@dataclass
class Diagnostics:
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    movex_entropy: Optional[float] = None
    rotate_entropy: Optional[float] = None
    castslot_entropy: Optional[float] = None
    focus_entropy: Optional[float] = None
    policy_grad_norm: Optional[float] = None
    value_grad_norm: Optional[float] = None
