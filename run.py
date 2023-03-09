from argparse import ArgumentParser, Namespace

import asyncio
import numpy as np
from gym_derk import DerkSession, DerkAgentServer, DerkAppInstance, ObservationKeys
from agent.bot import DerkPlayer
from tqdm.asyncio import tarange


def print_items(observation: np.ndarray) -> None:
    observation = observation.reshape(-1, 3, 64)
    for agent in range(3):
        for index in range(32, 47):
            key = ObservationKeys(index)
            if observation[0, agent, key.value] == 1:
                print(f"Agent {agent} has {key.name}")


async def random(env: DerkSession, episodes: int) -> None:
    """
    """
    for _ in range(episodes):
        await env.reset()

        while not env.done:
            action = [env.action_space.sample() for _ in range(env.n_agents)]
            await env.step(action)


async def player(env: DerkSession, episodes: int) -> None:
    """
    Runs a DerkPlayer
    """
    player = DerkPlayer(env.n_agents, env.action_space)

    async for _ in tarange(episodes):
        obs = await env.reset()

        print_items(obs)

        player.signal_env_reset(obs)
        ordi = await env.step()

        while not env.done:
            actions = player.take_action(ordi)
            ordi = await env.step(actions)


async def main(config: Namespace) -> None:
    """
    Runs the game in n arenas between p1 and p2
    """
    agent0 = DerkAgentServer(player, args={"episodes": config.episodes}, port=8795)
    if config.random:
        agent1 = DerkAgentServer(random, args={"episodes": config.episodes}, port=8796)
    else:
        agent1 = DerkAgentServer(player, args={"episodes": config.episodes}, port=8796)

    await agent0.start()
    await agent1.start()

    app = DerkAppInstance()
    await app.start()

    await app.run_session(
        n_arenas=config.arenas,
        turbo_mode=config.turbo,
        agent_hosts=[
            {"uri": agent0.uri, "regions": [{"sides": "home"}]},
            {"uri": agent1.uri, "regions": [{"sides": "away"}]},
        ],
        reward_function={
            "damageEnemyStatue": 0,
            "damageEnemyUnit": 0,
            "killEnemyStatue": 4,
            "killEnemyUnit": 1,
            "healFriendlyStatue": 0,
            "healTeammate1": 0,
            "healTeammate2": 0,
            "timeSpentHomeBase": 0,
            "timeSpentHomeTerritory": 0,
            "timeSpentAwayTerritory": 0,
            "timeSpentAwayBase": 0,
            "damageTaken": 0,
            "friendlyFire": 0,
            "healEnemy": 0,
            "fallDamageTaken": 0,
            "statueDamageTaken": 0,
            "manualBonus": 0,
            "victory": 0,
            "loss": 0,
            "tie": 0,
            "teamSpirit": 0.0,
            "timeScaling": 1.0,
        },
    )
    await app.print_team_stats()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--episodes", type=int, default=10,
    )
    parser.add_argument(
        "--arenas", help="Number of arenas to run, Defaults to 1", type=int, default=1,
    )
    parser.add_argument(
        "--turbo", action="store_true",
    )
    parser.add_argument(
        "--random", action="store_true",
    )
    config = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(main(config))
