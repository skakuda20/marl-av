"""
Comprehensive test suite for marl-av.

Covers every implemented module:
    env/rewards.py          — SVO transform and constants
    agents/rl_agent.py      — discretization, Q-table, epsilon-greedy, Bellman update
    training/train_rl.py    — training loop smoke test
    evaluation/metrics.py   — all three metric functions
    evaluation/compare_methods.py — eval episode loop and comparison table

Run with either:
    python3 test_suite.py
    pytest   test_suite.py -v
"""

import math
import sys
import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode(collision=False, done_1=True, done_2=True, steps=55,
                  r1=40.0, r2=40.0):
    return {
        "collision": collision,
        "done_1": done_1,
        "done_2": done_2,
        "steps": steps,
        "total_reward_1": r1,
        "total_reward_2": r2,
    }


# ===========================================================================
# 1.  SVO Transform  (env/rewards.py)
# ===========================================================================

class TestSVOTransform(unittest.TestCase):

    def setUp(self):
        from env.rewards import (
            apply_svo_transform,
            SVO_SELFISH, SVO_PROSOCIAL, SVO_ALTRUISTIC, SVO_COMPETITIVE,
        )
        self.fn = apply_svo_transform
        self.SELFISH     = SVO_SELFISH
        self.PROSOCIAL   = SVO_PROSOCIAL
        self.ALTRUISTIC  = SVO_ALTRUISTIC
        self.COMPETITIVE = SVO_COMPETITIVE

    def test_selfish_returns_own_reward(self):
        self.assertAlmostEqual(self.fn(10.0, 5.0, self.SELFISH), 10.0)

    def test_altruistic_returns_other_reward(self):
        self.assertAlmostEqual(self.fn(10.0, 5.0, self.ALTRUISTIC), 5.0, places=9)

    def test_prosocial_equal_weight(self):
        expected = 10.0 * math.sqrt(2)
        self.assertAlmostEqual(self.fn(10.0, 10.0, self.PROSOCIAL), expected, places=9)

    def test_competitive_subtracts_other(self):
        # cos(-pi/4)*10 + sin(-pi/4)*0 = 10/sqrt(2)
        result = self.fn(10.0, 0.0, self.COMPETITIVE)
        self.assertAlmostEqual(result, 10.0 / math.sqrt(2), places=9)

    def test_negative_other_reward_altruistic(self):
        # Altruistic agent fully absorbs other's negative reward
        self.assertAlmostEqual(self.fn(50.0, -100.0, self.ALTRUISTIC), -100.0, places=9)

    def test_returns_float(self):
        result = self.fn(1, 1, 0.0)
        self.assertIsInstance(result, float)

    def test_selfish_constant_value(self):
        self.assertEqual(self.SELFISH, 0.0)

    def test_prosocial_constant_value(self):
        self.assertAlmostEqual(self.PROSOCIAL, math.pi / 4, places=9)

    def test_altruistic_constant_value(self):
        self.assertAlmostEqual(self.ALTRUISTIC, math.pi / 2, places=9)

    def test_competitive_constant_value(self):
        self.assertAlmostEqual(self.COMPETITIVE, -math.pi / 4, places=9)


# ===========================================================================
# 2.  State Discretization  (agents/rl_agent.py)
# ===========================================================================

class TestStateDiscretization(unittest.TestCase):

    def setUp(self):
        from agents.rl_agent import _discretize_obs, NUM_STATES
        self._disc = _discretize_obs
        self.NUM_STATES = NUM_STATES

    def _obs(self, vx=8.0, goal_dist=10.0, other_dist=10.0):
        obs = np.zeros(10, dtype=np.float32)
        obs[2] = vx       # own vx → speed
        obs[8] = goal_dist
        obs[9] = other_dist
        return obs

    def test_output_in_valid_range(self):
        for _ in range(50):
            obs = self._obs(
                vx=np.random.uniform(0, 15),
                goal_dist=np.random.uniform(0, 40),
                other_dist=np.random.uniform(0, 40),
            )
            idx = self._disc(obs)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, self.NUM_STATES)

    def test_different_goals_give_different_states(self):
        s1 = self._disc(self._obs(goal_dist=2.0))   # bin 0
        s2 = self._disc(self._obs(goal_dist=20.0))  # bin 4
        self.assertNotEqual(s1, s2)

    def test_different_other_dist_give_different_states(self):
        s1 = self._disc(self._obs(other_dist=2.0))
        s2 = self._disc(self._obs(other_dist=20.0))
        self.assertNotEqual(s1, s2)

    def test_different_speed_gives_different_states(self):
        s1 = self._disc(self._obs(vx=1.0))   # low speed bin
        s2 = self._disc(self._obs(vx=12.0))  # high speed bin
        self.assertNotEqual(s1, s2)

    def test_zero_observation_is_valid(self):
        idx = self._disc(np.zeros(10))
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, self.NUM_STATES)

    def test_num_states_is_100(self):
        self.assertEqual(self.NUM_STATES, 100)


# ===========================================================================
# 3.  RLAgent — init, get_action, update, decay_epsilon
# ===========================================================================

class TestRLAgent(unittest.TestCase):

    def setUp(self):
        from agents.rl_agent import RLAgent, NUM_STATES, NUM_ACTIONS, DISCRETE_ACTIONS
        from env.rewards import SVO_PROSOCIAL
        self.RLAgent = RLAgent
        self.NUM_STATES = NUM_STATES
        self.NUM_ACTIONS = NUM_ACTIONS
        self.DISCRETE_ACTIONS = DISCRETE_ACTIONS
        self.svo = SVO_PROSOCIAL
        self.agent = RLAgent("agent_1", svo_angle=self.svo)

    def test_q_table_shape(self):
        self.assertEqual(self.agent.q_table.shape,
                         (self.NUM_STATES, self.NUM_ACTIONS))

    def test_q_table_initialised_to_zero(self):
        self.assertTrue(np.all(self.agent.q_table == 0.0))

    def test_svo_angle_stored(self):
        self.assertAlmostEqual(self.agent.svo_angle, self.svo)

    def test_greedy_action_in_discrete_set(self):
        obs = np.zeros(10, dtype=np.float32)
        action = self.agent.get_action(obs, training=False)
        self.assertIn(action, self.DISCRETE_ACTIONS)

    def test_explore_action_in_discrete_set(self):
        np.random.seed(0)
        obs = np.zeros(10, dtype=np.float32)
        for _ in range(20):
            action = self.agent.get_action(obs, training=True)
            self.assertIn(action, self.DISCRETE_ACTIONS)

    def test_update_changes_q_value(self):
        obs = np.zeros(10, dtype=np.float32)
        next_obs = np.zeros(10, dtype=np.float32)
        next_obs[8] = 5.0  # different state
        from agents.rl_agent import _discretize_obs
        s = _discretize_obs(obs)
        before = self.agent.q_table[s].copy()
        self.agent.update(obs, 0.0, 10.0, next_obs, False)
        after = self.agent.q_table[s]
        self.assertFalse(np.array_equal(before, after),
                         "Q-table should change after update with non-zero reward")

    def test_update_terminal_ignores_next_state(self):
        """Terminal update should not bootstrap from next state."""
        obs = np.zeros(10, dtype=np.float32)
        next_obs = np.full(10, 999.0, dtype=np.float32)  # out-of-range values
        from agents.rl_agent import _discretize_obs
        s = _discretize_obs(obs)
        a_idx = 2  # action 0.0
        before = self.agent.q_table[s, a_idx]
        self.agent.update(obs, 0.0, 5.0, next_obs, done=True)
        # target = 5.0 (no gamma * max Q)
        expected = before + self.agent.lr * (5.0 - before)
        self.assertAlmostEqual(self.agent.q_table[s, a_idx], expected, places=9)

    def test_decay_epsilon_reduces_epsilon(self):
        eps_before = self.agent.epsilon
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, eps_before)

    def test_decay_epsilon_floor(self):
        self.agent.epsilon = self.agent.epsilon_min
        self.agent.decay_epsilon()
        self.assertEqual(self.agent.epsilon, self.agent.epsilon_min)

    def test_default_svo_is_selfish(self):
        agent = self.RLAgent("x")
        self.assertEqual(agent.svo_angle, 0.0)


# ===========================================================================
# 4.  Training Loop  (training/train_rl.py)
# ===========================================================================

class TestTrainingLoop(unittest.TestCase):

    def setUp(self):
        from training.train_rl import train_rl_agent
        from env.rewards import SVO_PROSOCIAL, SVO_SELFISH
        self.train = train_rl_agent
        self.SVO_PROSOCIAL = SVO_PROSOCIAL
        self.SVO_SELFISH = SVO_SELFISH

    def _run(self, n=10):
        return self.train(
            num_episodes=n,
            agent_1_svo=self.SVO_PROSOCIAL,
            agent_2_svo=self.SVO_SELFISH,
            verbose=False,
            log_interval=n,
        )

    def test_returns_two_agents_and_metrics(self):
        a1, a2, m = self._run()
        from agents.rl_agent import RLAgent
        self.assertIsInstance(a1, RLAgent)
        self.assertIsInstance(a2, RLAgent)
        self.assertIsInstance(m, dict)

    def test_metrics_keys_present(self):
        _, _, m = self._run()
        for key in ("collision_rate", "success_rate", "avg_reward_1",
                    "avg_reward_2", "avg_steps"):
            self.assertIn(key, m)

    def test_metrics_lists_have_one_entry_per_log_interval(self):
        _, _, m = self._run(n=10)
        self.assertEqual(len(m["collision_rate"]), 1)

    def test_collision_rate_in_valid_range(self):
        _, _, m = self._run()
        for v in m["collision_rate"]:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_agents_have_different_svo_angles(self):
        a1, a2, _ = self._run()
        self.assertAlmostEqual(a1.svo_angle, self.SVO_PROSOCIAL)
        self.assertAlmostEqual(a2.svo_angle, self.SVO_SELFISH)

    def test_q_tables_modified_after_training(self):
        a1, a2, _ = self._run(n=20)
        self.assertFalse(np.all(a1.q_table == 0.0),
                         "agent_1 Q-table should be non-zero after training")
        self.assertFalse(np.all(a2.q_table == 0.0),
                         "agent_2 Q-table should be non-zero after training")

    def test_epsilon_decayed_after_training(self):
        a1, _, _ = self._run(n=20)
        self.assertLess(a1.epsilon, 1.0)


# ===========================================================================
# 5.  Evaluation Metrics  (evaluation/metrics.py)
# ===========================================================================

class TestMetrics(unittest.TestCase):

    def setUp(self):
        from evaluation.metrics import (
            compute_success_rate, compute_collision_rate, compute_efficiency
        )
        self.success_rate  = compute_success_rate
        self.collision_rate = compute_collision_rate
        self.efficiency     = compute_efficiency

    # --- success_rate ---

    def test_all_success(self):
        data = [_make_episode(collision=False, done_1=True, done_2=True)] * 4
        self.assertAlmostEqual(self.success_rate(data), 1.0)

    def test_no_success(self):
        data = [_make_episode(collision=False, done_1=True, done_2=False)] * 4
        self.assertEqual(self.success_rate(data), 0.0)

    def test_partial_success(self):
        data = (
            [_make_episode(collision=False, done_1=True, done_2=True)] * 3 +
            [_make_episode(collision=True,  done_1=False, done_2=False)] * 1
        )
        self.assertAlmostEqual(self.success_rate(data), 0.75)

    def test_collision_counts_as_failure(self):
        data = [_make_episode(collision=True, done_1=True, done_2=True)]
        self.assertEqual(self.success_rate(data), 0.0)

    def test_empty_returns_zero(self):
        self.assertEqual(self.success_rate([]), 0.0)

    # --- collision_rate ---

    def test_collision_rate_all_collide(self):
        data = [_make_episode(collision=True)] * 5
        self.assertAlmostEqual(self.collision_rate(data), 1.0)

    def test_collision_rate_none_collide(self):
        data = [_make_episode(collision=False)] * 5
        self.assertEqual(self.collision_rate(data), 0.0)

    def test_collision_rate_mixed(self):
        data = (
            [_make_episode(collision=True)]  * 2 +
            [_make_episode(collision=False)] * 8
        )
        self.assertAlmostEqual(self.collision_rate(data), 0.2)

    def test_collision_rate_empty_returns_zero(self):
        self.assertEqual(self.collision_rate([]), 0.0)

    # --- efficiency ---

    def test_avg_steps_correct(self):
        data = [_make_episode(steps=60), _make_episode(steps=80)]
        eff = self.efficiency(data)
        self.assertAlmostEqual(eff["avg_steps"], 70.0)

    def test_avg_time_to_clear_only_successful(self):
        data = [
            _make_episode(collision=False, done_1=True, done_2=True, steps=60),
            _make_episode(collision=True,  done_1=False, done_2=False, steps=30),
        ]
        eff = self.efficiency(data)
        # Only the successful episode (60 steps * 0.1 = 6.0 s)
        self.assertAlmostEqual(eff["avg_time_to_clear"], 6.0)

    def test_avg_time_to_clear_none_when_no_success(self):
        data = [_make_episode(collision=True, done_1=False, done_2=False)]
        eff = self.efficiency(data)
        self.assertIsNone(eff["avg_time_to_clear"])

    def test_avg_delay_positive_when_slower_than_baseline(self):
        data = [_make_episode(steps=80)]  # 80 > 50 baseline
        eff = self.efficiency(data)
        self.assertGreater(eff["avg_delay"], 0.0)

    def test_avg_delay_negative_when_faster_than_baseline(self):
        data = [_make_episode(steps=20)]  # 20 < 50 baseline
        eff = self.efficiency(data)
        self.assertLess(eff["avg_delay"], 0.0)

    def test_efficiency_keys_present(self):
        eff = self.efficiency([_make_episode()])
        for k in ("avg_time_to_clear", "avg_delay", "avg_steps"):
            self.assertIn(k, eff)

    def test_efficiency_empty_input(self):
        eff = self.efficiency([])
        self.assertIsNone(eff["avg_time_to_clear"])
        self.assertEqual(eff["avg_delay"], 0.0)
        self.assertEqual(eff["avg_steps"], 0.0)


# ===========================================================================
# 6.  Compare Methods  (evaluation/compare_methods.py)
# ===========================================================================

class TestRunEvalEpisodes(unittest.TestCase):

    def setUp(self):
        from evaluation.compare_methods import run_eval_episodes
        from env.intersection_env import IntersectionEnv
        from agents.heuristic import create_heuristic_pair
        self.run = run_eval_episodes
        self.env = IntersectionEnv()
        self.a1, self.a2 = create_heuristic_pair("cautious", "cautious")

    def test_returns_list_of_correct_length(self):
        results = self.run(self.a1, self.a2, self.env, n_episodes=5)
        self.assertEqual(len(results), 5)

    def test_each_result_has_required_keys(self):
        results = self.run(self.a1, self.a2, self.env, n_episodes=3)
        for r in results:
            for key in ("collision", "done_1", "done_2", "steps",
                        "total_reward_1", "total_reward_2"):
                self.assertIn(key, r, f"Missing key: {key}")

    def test_steps_at_least_one(self):
        results = self.run(self.a1, self.a2, self.env, n_episodes=3)
        for r in results:
            self.assertGreater(r["steps"], 0)

    def test_collision_is_bool(self):
        results = self.run(self.a1, self.a2, self.env, n_episodes=3)
        for r in results:
            self.assertIsInstance(r["collision"], bool)


class TestCompareMethods(unittest.TestCase):

    def setUp(self):
        from evaluation.compare_methods import compare_methods
        self.compare = compare_methods

    def test_returns_dict_with_all_labels(self):
        from agents.heuristic import create_heuristic_pair
        methods = {
            "cautious vs cautious": create_heuristic_pair("cautious", "cautious"),
        }
        results = self.compare(num_eval_episodes=5, method_dict=methods)
        self.assertIn("cautious vs cautious", results)

    def test_result_has_metric_keys(self):
        from agents.heuristic import create_heuristic_pair
        methods = {"c vs c": create_heuristic_pair("cautious", "cautious")}
        results = self.compare(num_eval_episodes=5, method_dict=methods)
        m = results["c vs c"]
        for key in ("success_rate", "collision_rate",
                    "avg_time_to_clear", "avg_delay", "avg_steps"):
            self.assertIn(key, m)

    def test_rates_in_valid_range(self):
        from agents.heuristic import create_heuristic_pair
        methods = {"c vs a": create_heuristic_pair("cautious", "aggressive")}
        results = self.compare(num_eval_episodes=10, method_dict=methods)
        m = results["c vs a"]
        self.assertGreaterEqual(m["success_rate"], 0.0)
        self.assertLessEqual(m["success_rate"], 1.0)
        self.assertGreaterEqual(m["collision_rate"], 0.0)
        self.assertLessEqual(m["collision_rate"], 1.0)


# ===========================================================================
# 7.  Integration — train then evaluate
# ===========================================================================

class TestIntegrationTrainAndEvaluate(unittest.TestCase):

    def test_rl_agents_work_in_eval_pipeline(self):
        """Train briefly then pass RL agents into compare_methods."""
        from training.train_rl import train_rl_agent
        from evaluation.compare_methods import compare_methods
        from env.rewards import SVO_PROSOCIAL, SVO_SELFISH

        a1, a2, _ = train_rl_agent(
            num_episodes=20,
            agent_1_svo=SVO_PROSOCIAL,
            agent_2_svo=SVO_SELFISH,
            verbose=False,
            log_interval=20,
        )

        results = compare_methods(
            num_eval_episodes=10,
            method_dict={"RL trained": (a1, a2)},
        )

        self.assertIn("RL trained", results)
        m = results["RL trained"]
        self.assertGreaterEqual(m["collision_rate"], 0.0)
        self.assertLessEqual(m["collision_rate"], 1.0)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("marl-av test suite")
    print("=" * 70)
    loader = unittest.TestLoader()
    # Preserve definition order so groups print together
    suite = unittest.TestSuite()
    for cls in [
        TestSVOTransform,
        TestStateDiscretization,
        TestRLAgent,
        TestTrainingLoop,
        TestMetrics,
        TestRunEvalEpisodes,
        TestCompareMethods,
        TestIntegrationTrainAndEvaluate,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
