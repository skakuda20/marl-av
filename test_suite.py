"""
Regression suite aligned with the current marl-av implementation.
"""

import math
import unittest

import numpy as np


def _make_episode(collision=False, done_1=True, done_2=True, steps=55, r1=40.0, r2=40.0):
    state_1 = [[-10.0, 0.0, 8.0, 0.0], [0.0, 0.0, 8.0, 0.0]]
    state_2 = [[0.0, -10.0, 0.0, 8.0], [0.0, 0.0, 0.0, 8.0]]
    return {
        "collision": collision,
        "done_1": done_1,
        "done_2": done_2,
        "steps": steps,
        "total_reward_1": r1,
        "total_reward_2": r2,
        "actions_1": [0.0, 0.0, 0.0],
        "actions_2": [0.0, 0.0, 0.0],
        "states_1": state_1,
        "states_2": state_2,
        "scenario": {"scenario_split": "train"},
    }


class TestSVOTransform(unittest.TestCase):
    def setUp(self):
        from env.rewards import (
            SVO_ALTRUISTIC,
            SVO_COMPETITIVE,
            SVO_PROSOCIAL,
            SVO_SELFISH,
            apply_svo_transform,
        )
        self.fn = apply_svo_transform
        self.selfish = SVO_SELFISH
        self.prosocial = SVO_PROSOCIAL
        self.altruistic = SVO_ALTRUISTIC
        self.competitive = SVO_COMPETITIVE

    def test_selfish_returns_own_reward(self):
        self.assertAlmostEqual(self.fn(10.0, 4.0, self.selfish), 10.0)

    def test_altruistic_returns_other_reward(self):
        self.assertAlmostEqual(self.fn(10.0, 4.0, self.altruistic), 4.0)

    def test_prosocial_angle_constant(self):
        self.assertAlmostEqual(self.prosocial, math.pi / 4)

    def test_competitive_angle_constant(self):
        self.assertAlmostEqual(self.competitive, -math.pi / 4)


class TestEnvironment(unittest.TestCase):
    def test_reset_randomizes_scenarios_by_split(self):
        from env.intersection_env import IntersectionEnv

        env = IntersectionEnv(scenario_split="train")
        _, info_train = env.reset(seed=11, options={"scenario_split": "train"})
        _, info_test = env.reset(seed=11, options={"scenario_split": "test"})

        self.assertEqual(info_train["scenario"]["scenario_split"], "train")
        self.assertEqual(info_test["scenario"]["scenario_split"], "test")
        self.assertNotEqual(
            info_train["scenario"]["spawn_distance_1"],
            info_test["scenario"]["spawn_distance_1"],
        )

    def test_goal_reward_awarded_once(self):
        from env.intersection_env import IntersectionEnv

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
            "start_pos_2": np.array([10.0, 10.0], dtype=np.float32),
            "goal_pos_2": (10.0, 10.0),
            "start_vel_2": np.array([0.0, 0.0], dtype=np.float32),
        }
        env.reset(options={"scenario": scenario})
        _, reward_1, _, _, info_1 = env.step(np.array([0.0, 0.0]))
        _, reward_2, _, _, info_2 = env.step(np.array([0.0, 0.0]))

        self.assertTrue(info_1["reached_goal_1"])
        self.assertFalse(info_2["reached_goal_1"])
        self.assertGreater(reward_1["agent_1"], reward_2["agent_1"])


class TestRLAgent(unittest.TestCase):
    def setUp(self):
        from agents.rl_agent import DISCRETE_ACTIONS, NUM_ACTIONS, NUM_STATES, RLAgent, _discretize_obs

        self.actions = DISCRETE_ACTIONS
        self.num_actions = NUM_ACTIONS
        self.num_states = NUM_STATES
        self.agent = RLAgent("agent_1")
        self.discretize = _discretize_obs

    def test_q_table_shape(self):
        self.assertEqual(self.agent.q_table.shape, (self.num_states, self.num_actions))

    def test_discretization_uses_agent_partition(self):
        obs = np.zeros(11, dtype=np.float32)
        self.assertNotEqual(self.discretize(obs, 0), self.discretize(obs, 1))

    def test_update_changes_value(self):
        obs = np.zeros(11, dtype=np.float32)
        next_obs = np.ones(11, dtype=np.float32)
        state = self.discretize(obs, 0)
        before = self.agent.q_table[state].copy()
        self.agent.update(obs, 0.0, 5.0, next_obs, False)
        self.assertFalse(np.array_equal(before, self.agent.q_table[state]))


class TestTrainingAndEvaluation(unittest.TestCase):
    def test_training_supports_seed_and_svo_toggle(self):
        from training.train_rl import train_rl_agent

        a1, a2, metrics = train_rl_agent(
            num_episodes=5,
            use_svo=False,
            seed=5,
            verbose=False,
            log_interval=5,
        )
        self.assertEqual(len(metrics["collision_rate"]), 1)
        self.assertLess(a1.epsilon, 1.0)
        self.assertLess(a2.epsilon, 1.0)

    def test_run_eval_episodes_returns_trace_data(self):
        from agents.heuristic import create_heuristic_pair
        from env.intersection_env import IntersectionEnv
        from evaluation.compare_methods import run_eval_episodes

        env = IntersectionEnv()
        a1, a2 = create_heuristic_pair("cautious", "priority")
        episodes = run_eval_episodes(a1, a2, env, n_episodes=2, seed=3)

        self.assertEqual(len(episodes), 2)
        self.assertIn("actions_1", episodes[0])
        self.assertIn("states_1", episodes[0])
        self.assertIn("scenario", episodes[0])

    def test_compare_methods_reports_extended_metrics(self):
        from agents.heuristic import create_heuristic_pair
        from evaluation.compare_methods import compare_methods

        results = compare_methods(
            num_eval_episodes=2,
            method_dict={"priority_vs_yield": create_heuristic_pair("priority", "yield")},
        )
        metrics = results["priority_vs_yield"]
        for key in ("min_time_to_collision", "safety_margin", "avg_jerk", "reward_fairness"):
            self.assertIn(key, metrics)


class TestMetrics(unittest.TestCase):
    def test_summary_contains_extended_metrics(self):
        from evaluation.metrics import summarize_episodes

        summary = summarize_episodes([_make_episode()], dt=0.1)
        for key in (
            "success_rate",
            "collision_rate",
            "avg_time_to_clear",
            "min_time_to_collision",
            "safety_margin",
            "avg_jerk",
            "reward_fairness",
            "avg_reward_1",
        ):
            self.assertIn(key, summary)

    def test_generalization_gap(self):
        from evaluation.metrics import compute_generalization_gap

        gap = compute_generalization_gap({"success_rate": 0.8}, {"success_rate": 0.5})
        self.assertAlmostEqual(gap, 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
