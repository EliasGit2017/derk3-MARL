import itertools
import os
from argparse import ArgumentParser
from pathlib import Path
import dataclasses

import numpy as np
from gym_derk.envs import DerkEnv
from gym_derk import TeamStatsKeys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from derk3.trainer import Trainer, Batch
from derk3.parameters import Parameters

reward_function = {
    "damageEnemyStatue": 0.9,
    "damageEnemyUnit": 0.5,
    "killEnemyStatue": 3,
    "killEnemyUnit": 2,
    "healFriendlyStatue": 10,
    "healTeammate1": 6,
    "healTeammate2": 6,
    "timeSpentHomeBase": 3,
    "timeSpentHomeTerritory": 1,
    "timeSpentAwayTerritory": 0,
    "timeSpentAwayBase": 0,
    "damageTaken": -1.5,
    "friendlyFire": -5,
    "healEnemy": -6.0,
    "fallDamageTaken": -100, # highest penalty
    "statueDamageTaken": -10,
    "manualBonus": 0,
    "victory": 5,
    "loss": -5,
    "tie": -1,
    "teamSpirit": 1.0,
    "timeScaling": 1.0,
}


def parse_config():
    parser = ArgumentParser()
    parser.add_argument(
        "parameters", type=str, help="set path to the parameters JSON file"
    )
    parser.add_argument(
        "--device",
        type=str,
        help='set the compute device type (default: "cuda" or "cpu")',
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8788,
        help="set the server port number {default: 8788)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="set the number of episodes (default: 1000)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="set the path to save a checkpoint (default: none)",
    )
    parser.add_argument(
        "--restore",
        type=str,
        help="set the path to load a checkpoint (default: none)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="set the output path (default: .)",
    )

    config = parser.parse_args()

    return config


def main():
    config = parse_config()

    output_path = Path(config.output)
    output_path.mkdir(parents=True, exist_ok=True)

    os.environ["DERK_CHROMEDATA"] = str(output_path / "chromedata")

    param = Parameters(config.parameters)

    env = DerkEnv(
        n_arenas=param.arenas,
        turbo_mode=True,
        reward_function=reward_function,
        agent_server_args={"port": config.port},
    )

    trainer = Trainer(
        param,
        env.action_space,
        env.observation_space,
        device_type=config.device,
    )

    if config.restore is not None:
        trainer.load(config.restore)

    print(trainer.policy)
    print(trainer.value)

    num_param = sum(
        param.numel()
        for param in itertools.chain(
            trainer.policy.parameters(), trainer.value.parameters()
        )
        if param.requires_grad
    )
    print("Parameters:", num_param)

    writer = SummaryWriter()

    with tqdm(range(config.episodes), total=config.episodes) as progress:
        for episode in progress:
            observation = env.reset()
            trainer.reset()

            observations = [observation]
            actions = []
            log_likelihoods = []
            rewards = []

            while True:
                action, log_likelihood = trainer.act(observation)

                observation, reward, done, _ = env.step(action)

                observations.append(observation)
                actions.append(action)
                log_likelihoods.append(log_likelihood)
                rewards.append(reward)

                if all(done):
                    break

            team_stats = env.team_stats.reshape(2, -1, len(TeamStatsKeys))[0]

            # Print team statistics
            avg_team_stats = team_stats.mean(0)
            progress.write(
                "Episode {}: {}".format(
                    episode,
                    ", ".join(
                        "Average{}: {:.2f}".format(key.name, avg_team_stats[key.value])
                        for key in (
                            TeamStatsKeys.Hitpoints,
                            TeamStatsKeys.AliveTime,
                            TeamStatsKeys.CumulativeHitpoints,
                        )
                    ),
                )
            )
            progress.write(
                "Episode {}: {}".format(
                    episode,
                    ", ".join(
                        "AverageLocal{}: {:.2f}".format(
                            key.name, avg_team_stats[key.value]
                        )
                        for key in (TeamStatsKeys.Reward, TeamStatsKeys.OpponentReward)
                    ),
                )
            )

            # Log team statistics
            for key in TeamStatsKeys:
                writer.add_scalar(key.name, avg_team_stats[key.value], episode)

            # Subtract mean score from each agent for zero-sum game
            rewards = np.array(rewards)

            home_rewards = rewards[:, : env.n_agents // 2]
            home_rewards = home_rewards.reshape(
                -1, env.n_teams // 2, env.n_agents_per_team
            )
            away_rewards = rewards[:, env.n_agents // 2 :]
            away_rewards = away_rewards.reshape(
                -1, env.n_teams // 2, env.n_agents_per_team
            )
            home_zero_sum_rewards = home_rewards - away_rewards.mean(-1, keepdims=True)
            away_zero_sum_rewards = away_rewards - home_rewards.mean(-1, keepdims=True)

            home_score = home_zero_sum_rewards.sum(axis=(0, -1))
            away_score = away_zero_sum_rewards.sum(axis=(0, -1))

            # Print and log episode stats
            episode_stats = {}
            episode_stats["AverageHomeScore"] = home_score.mean()
            episode_stats["AverageAwayScore"] = away_score.mean()
            episode_stats["HomeWins"] = np.sum(home_score > away_score)
            episode_stats["AwayWins"] = np.sum(home_score < away_score)
            episode_stats["Ties"] = np.sum(home_score == away_score)

            progress.write(
                "Episode {}: {}".format(
                    episode,
                    ", ".join(
                        "{}: {:.2f}".format(key, value)
                        for key, value in episode_stats.items()
                    ),
                )
            )

            for key, value in episode_stats.items():
                writer.add_scalar(key, value, episode)

            zero_sum_rewards = np.concatenate(
                (
                    home_zero_sum_rewards.reshape(-1, env.n_agents // 2),
                    away_zero_sum_rewards.reshape(-1, env.n_agents // 2),
                ),
                axis=-1,
            )

            # Process batch and update parameters
            diagnostics = trainer.step(
                Batch(
                    np.array(observations[:-1]),
                    np.array(actions),
                    np.array(log_likelihoods),
                    zero_sum_rewards,
                ),
            )

            # Write to tensorboard
            for name, diagnostic in dataclasses.asdict(diagnostics).items():
                if diagnostic is None:
                    continue
                if isinstance(diagnostic, float):
                    writer.add_scalar(name, diagnostic, episode)
                else:
                    for key, value in diagnostic._asdict().items():
                        writer.add_scalar(name + "_" + key, value, episode)

            if config.checkpoint is not None and (
                (episode + 1) % param.checkpoint_interval == 0
                or (episode + 1) == config.episodes
            ):
                checkpoint_path = output_path / config.checkpoint
                progress.write("Saving checkpoint: {}".format(checkpoint_path))
                trainer.save(checkpoint_path)


if __name__ == "__main__":
    main()
