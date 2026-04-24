"""
Training loop for Q-learning agents with Social Value Orientation (SVO).

Each agent maintains its own Q-table and SVO angle. After every env step the
raw rewards are SVO-transformed before the Q-update, so two agents with
different phi values experience the same environment differently.

Example
-------
    # Prosocial agent 1 vs selfish agent 2
    agent_1, agent_2, metrics = train_rl_agent(
        num_episodes=2000,
        agent_1_svo=SVO_PROSOCIAL,
        agent_2_svo=SVO_SELFISH,
    )
"""

import numpy as np

from env.intersection_env import IntersectionEnv
from agents.rl_agent import RLAgent
from env.rewards import apply_svo_transform, SVO_SELFISH, SVO_PROSOCIAL


def train_rl_agent(
    num_episodes: int = 2000,
    agent_1_svo: float = SVO_PROSOCIAL,
    agent_2_svo: float = SVO_SELFISH,
    use_svo: bool = True,
    seed: int = 0,
    env_kwargs: dict = None,
    reward_kwargs: dict = None,
    agent_kwargs: dict = None,
    scenario_split: str = "train",
    save_path: str = None,
    verbose: bool = True,
    log_interval: int = 100,
):
    """
    Train two Q-learning agents with given SVO angles.

    Args:
        num_episodes: Number of training episodes.
        agent_1_svo: SVO angle for agent 1 in radians.
        agent_2_svo: SVO angle for agent 2 in radians.
        use_svo: If True (default), apply the SVO transform before each
            Q-update so agents blend self and other rewards according to
            their SVO angles.  If False, each agent updates on its own
            raw reward only — equivalent to purely self-interested Q-learning
            regardless of the svo_angle values set above.
        save_path: Optional .npz path to save Q-tables after training.
        verbose: Print progress every log_interval episodes.
        log_interval: Episodes between progress prints.

    Returns:
        Tuple of (agent_1, agent_2, metrics_dict) where metrics_dict
        contains lists logged every log_interval episodes:
            collision_rate, success_rate, avg_reward_1,
            avg_reward_2, avg_steps.
    """
    np.random.seed(seed)
    env_config = dict(env_kwargs or {})
    env_config.setdefault("scenario_split", scenario_split)
    env = IntersectionEnv(**env_config)
    if reward_kwargs is not None:
        env.reward_fn = env.reward_fn.__class__(**reward_kwargs)
    agent_cfg = agent_kwargs or {}
    agent_1 = RLAgent("agent_1", svo_angle=agent_1_svo, **agent_cfg)
    agent_2 = RLAgent("agent_2", svo_angle=agent_2_svo, **agent_cfg)

    metrics = {
        "collision_rate": [],
        "success_rate": [],
        "avg_reward_1": [],
        "avg_reward_2": [],
        "avg_steps": [],
    }

    # Accumulators for the current logging window
    win_collisions = 0
    win_successes = 0
    win_rewards_1 = []
    win_rewards_2 = []
    win_steps = []

    for episode in range(num_episodes):
        if episode == 0:
            obs, info = env.reset(seed=seed)
        else:
            obs, info = env.reset()
        total_reward_1 = 0.0
        total_reward_2 = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            obs_1 = obs["agent_1"]
            obs_2 = obs["agent_2"]

            action_1 = agent_1.get_action(obs_1, training=True)
            action_2 = agent_2.get_action(obs_2, training=True)

            next_obs, reward, terminated, truncated, info = env.step(
                np.array([action_1, action_2])
            )
            done = terminated or truncated

            raw_r1 = reward["agent_1"]
            raw_r2 = reward["agent_2"]

            if use_svo:
                # SVO transform: each agent blends its own reward with the other's
                upd_r1 = apply_svo_transform(raw_r1, raw_r2, agent_1.svo_angle)
                upd_r2 = apply_svo_transform(raw_r2, raw_r1, agent_2.svo_angle)
            else:
                # No SVO: each agent updates on its own raw reward only
                upd_r1 = raw_r1
                upd_r2 = raw_r2

            agent_1.update(obs_1, action_1, upd_r1, next_obs["agent_1"], done)
            agent_2.update(obs_2, action_2, upd_r2, next_obs["agent_2"], done)

            obs = next_obs
            total_reward_1 += raw_r1
            total_reward_2 += raw_r2

        agent_1.decay_epsilon()
        agent_2.decay_epsilon()

        # Accumulate window metrics
        win_collisions += int(info.get("collision", False))
        win_successes += int(
            info.get("done_1", False) and info.get("done_2", False)
        )
        win_rewards_1.append(total_reward_1)
        win_rewards_2.append(total_reward_2)
        win_steps.append(info["steps"])

        if (episode + 1) % log_interval == 0:
            metrics["collision_rate"].append(win_collisions / log_interval)
            metrics["success_rate"].append(win_successes / log_interval)
            metrics["avg_reward_1"].append(float(np.mean(win_rewards_1)))
            metrics["avg_reward_2"].append(float(np.mean(win_rewards_2)))
            metrics["avg_steps"].append(float(np.mean(win_steps)))

            if verbose:
                print(
                    f"Episode {episode + 1:>5} | "
                    f"Collision: {metrics['collision_rate'][-1]:.2f} | "
                    f"Success: {metrics['success_rate'][-1]:.2f} | "
                    f"R1: {metrics['avg_reward_1'][-1]:>7.2f} | "
                    f"R2: {metrics['avg_reward_2'][-1]:>7.2f} | "
                    f"Steps: {metrics['avg_steps'][-1]:.1f} | "
                    f"eps: {agent_1.epsilon:.3f}"
                )

            win_collisions = 0
            win_successes = 0
            win_rewards_1 = []
            win_rewards_2 = []
            win_steps = []

    if save_path is not None:
        np.savez(
            save_path,
            q_table_1=agent_1.q_table,
            q_table_2=agent_2.q_table,
            svo_1=agent_1.svo_angle,
            svo_2=agent_2.svo_angle,
        )
        if verbose:
            print(f"Saved Q-tables to {save_path}")

    return agent_1, agent_2, metrics


def load_rl_agents(npz_path: str):
    """
    Reconstruct a greedy (evaluation-ready) agent pair from a saved .npz file.

    The .npz must contain keys: q_table_1, q_table_2, svo_1, svo_2
    (as written by train_rl_agent with save_path set).

    Args:
        npz_path: Path to the .npz file produced by train_rl_agent.

    Returns:
        Tuple of (agent_1, agent_2) with epsilon=0 for greedy evaluation.
    """
    data = np.load(npz_path)
    agent_1 = RLAgent("agent_1", svo_angle=float(data["svo_1"]))
    agent_2 = RLAgent("agent_2", svo_angle=float(data["svo_2"]))
    agent_1.q_table = data["q_table_1"]
    agent_2.q_table = data["q_table_2"]
    agent_1.epsilon = 0.0
    agent_2.epsilon = 0.0
    return agent_1, agent_2


if __name__ == "__main__":
    print("Training prosocial agent 1 (φ=π/4) vs selfish agent 2 (φ=0) ...")
    train_rl_agent(num_episodes=2000, verbose=True)
