"""Basic validation tests for the current marl-av codebase."""

import numpy as np

from agents.heuristic import create_heuristic_pair
from env.intersection_env import IntersectionEnv


def test_environment():
    env = IntersectionEnv(dt=0.1, max_steps=50)

    obs, info = env.reset(seed=7)
    assert "agent_1" in obs and "agent_2" in obs
    assert obs["agent_1"].shape == (11,)
    assert obs["agent_2"].shape == (11,)
    assert info["scenario"]["scenario_split"] == "train"

    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
    assert "agent_1" in reward and "agent_2" in reward
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "scenario" in info


def test_heuristic_agents():
    env = IntersectionEnv()
    agent_1, agent_2 = create_heuristic_pair("priority", "yield")
    obs, _ = env.reset(seed=3)

    action_1 = agent_1.get_action(obs["agent_1"])
    action_2 = agent_2.get_action(obs["agent_2"])
    assert -1.0 <= action_1 <= 1.0
    assert -1.0 <= action_2 <= 1.0

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(np.array([action_1, action_2]))
        if terminated or truncated:
            break
        action_1 = agent_1.get_action(obs["agent_1"])
        action_2 = agent_2.get_action(obs["agent_2"])


def test_goal_reward_awarded_once():
    env = IntersectionEnv(max_steps=5)
    scenario = {
        "priority": "balanced",
        "center": (0.0, 0.0),
        "spawn_distance_1": 1.0,
        "spawn_distance_2": 1.0,
        "initial_speed_1": 0.0,
        "initial_speed_2": 0.0,
        "arrival_offset": 0.0,
        "start_pos_1": np.array([0.0, 0.0], dtype=np.float32),
        "goal_pos_1": (0.0, 0.0),
        "start_vel_1": np.array([0.0, 0.0], dtype=np.float32),
        "start_pos_2": np.array([5.0, 5.0], dtype=np.float32),
        "goal_pos_2": (5.0, 5.0),
        "start_vel_2": np.array([0.0, 0.0], dtype=np.float32),
    }
    env.reset(options={"scenario": scenario})
    _, reward_1, _, _, info_1 = env.step(np.array([0.0, 0.0]))
    _, reward_2, _, _, info_2 = env.step(np.array([0.0, 0.0]))
    assert info_1["reached_goal_1"] is True
    assert info_2["reached_goal_1"] is False
    assert reward_1["agent_1"] > reward_2["agent_1"]


if __name__ == "__main__":
    test_environment()
    test_heuristic_agents()
    test_goal_reward_awarded_once()
    print("Basic tests passed.")
