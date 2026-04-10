"""
Tabular Q-learning agent with Social Value Orientation (SVO).

State space is discretized from the continuous 10-D observation using three
features: distance to own goal, distance to the other vehicle, and own speed.
Each agent stores its own Q-table and SVO angle, so heterogeneous populations
(e.g. one selfish, one prosocial) can be trained without changing the env.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

DISCRETE_ACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
NUM_ACTIONS = len(DISCRETE_ACTIONS)

# ---------------------------------------------------------------------------
# Observation discretization
# ---------------------------------------------------------------------------

# Bin edges for each feature (values above the last edge → highest bin)
_GOAL_DIST_BINS = np.array([4.0, 8.0, 12.0, 16.0])   # → 5 bins
_OTHER_DIST_BINS = np.array([4.0, 8.0, 12.0, 16.0])  # → 5 bins
_SPEED_BINS = np.array([3.0, 6.0, 9.0])               # → 4 bins

NUM_STATES = 5 * 5 * 4  # 100


def _discretize_obs(obs: np.ndarray) -> int:
    """
    Map a continuous observation to an integer state index.

    Observation layout (indices):
        0-1  own position (x, y)
        2-3  own velocity (vx, vy)
        4-5  other position (x, y)
        6-7  other velocity (vx, vy)
        8    distance to own goal
        9    distance to other vehicle

    Returns:
        Integer in [0, NUM_STATES).
    """
    goal_bin = int(np.searchsorted(_GOAL_DIST_BINS, obs[8]))
    other_bin = int(np.searchsorted(_OTHER_DIST_BINS, obs[9]))
    speed = np.sqrt(obs[2] ** 2 + obs[3] ** 2)
    speed_bin = int(np.searchsorted(_SPEED_BINS, speed))
    return goal_bin * 20 + other_bin * 4 + speed_bin


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RLAgent:
    """
    Tabular Q-learning agent with Social Value Orientation (SVO).

    The SVO angle phi is applied externally (in the training loop) to blend
    self and other rewards before the Q-update:

        r_svo = r_self * cos(phi) + r_other * sin(phi)

    Import SVO_* constants from env.rewards for common presets.
    """

    def __init__(
        self,
        agent_id: str,
        svo_angle: float = 0.0,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        """
        Args:
            agent_id: Identifier for this agent ("agent_1" or "agent_2").
            svo_angle: SVO angle in radians. 0 = selfish, π/4 = prosocial.
            learning_rate: Q-learning step size (alpha).
            discount: Discount factor (gamma).
            epsilon_start: Initial exploration probability.
            epsilon_min: Floor for epsilon after decay.
            epsilon_decay: Multiplicative decay per episode.
        """
        self.agent_id = agent_id
        self.svo_angle = svo_angle
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

    def get_action(self, observation: np.ndarray, training: bool = False) -> float:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            observation: Continuous observation array of shape (10,).
            training: If True, apply epsilon-greedy exploration.

        Returns:
            Continuous action value in [-1, 1].
        """
        state = _discretize_obs(observation)
        if training and np.random.random() < self.epsilon:
            return float(np.random.choice(DISCRETE_ACTIONS))
        action_idx = int(np.argmax(self.q_table[state]))
        return float(DISCRETE_ACTIONS[action_idx])

    def update(
        self,
        obs: np.ndarray,
        action: float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Apply one-step Bellman Q-update.

        The reward passed in should already be SVO-transformed by the caller
        (via env.rewards.apply_svo_transform) before being passed here.

        Args:
            obs: Observation before the action, shape (10,).
            action: Continuous action taken (one of DISCRETE_ACTIONS).
            reward: SVO-transformed reward received.
            next_obs: Observation after the action, shape (10,).
            done: Whether the episode terminated or was truncated.
        """
        state = _discretize_obs(obs)
        next_state = _discretize_obs(next_obs)
        action_idx = int(np.argmin(np.abs(DISCRETE_ACTIONS - action)))

        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])

        self.q_table[state, action_idx] += self.lr * (
            target - self.q_table[state, action_idx]
        )

    def decay_epsilon(self) -> None:
        """Decay the exploration rate. Call once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


if __name__ == "__main__":
    agent = RLAgent("agent_1")
    dummy_obs = np.zeros(10)
    print(f"Action (greedy): {agent.get_action(dummy_obs)}")
    print(f"Q-table shape: {agent.q_table.shape}  ({NUM_STATES} states × {NUM_ACTIONS} actions)")
