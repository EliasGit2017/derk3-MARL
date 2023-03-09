from pathlib import Path
from typing import Tuple

from gym.spaces import Box, Discrete
import torch

from derk3.agent import Agent


class DerkPlayer:
    """
    """

    NUM_AGENTS_PER_TEAM = 3

    def __init__(self, n_agents: int, action_space: Tuple[Box, Box, Box, Discrete, Discrete]):
        """
        Parameters:
         - n_agents: TOTAL number of agents being controlled (= #arenas * #agents per arena)
        """
        self.action_space = action_space

        self.observation_size = 64

        parent_path = Path(__file__).parent

        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(parent_path / "checkpoint.pt", map_location=device_type)

        params = checkpoint["params"]

        self.agent = Agent(params, action_space, device_type=device_type)
        self.agent.restore(checkpoint)

    def signal_env_reset(self, obs):
        """
        env.reset() was called
        """
        self.agent.reset()

    @torch.no_grad()
    def take_action(self, env_step_ret):
        """
        Parameters:
         - env_step_ret: whatever env.step() returned (obs_n, rew_n, done_n, info_n)

        Returns: action for each agent for each arena

        Actions:
         - MoveX: A number between -1 and 1
         - Rotate: A number between -1 and 1
         - ChaseFocus: A number between 0 and 1
         - CastingSlot:
                        0 = don't cast
                    1 - 3 = cast corresponding ability
         - ChangeFocus:
                        0 = keep current focus
                        1 = focus home statue
                    2 - 3 = focus teammates
                        4 = focus enemy statue
                    5 - 7 = focus enemy
        """
        observation, *_ = env_step_ret

        return self.agent.act(observation)
