"""
Reward structure for the intersection environment.
Balances safety, efficiency, and goal achievement.
"""

import numpy as np
from typing import Dict, Tuple


class RewardFunction:
    """Compute rewards for the intersection environment."""
    
    def __init__(
        self,
        collision_penalty: float = -100.0,
        goal_reward: float = 50.0,
        time_penalty: float = -0.1,
        efficiency_weight: float = 0.5,
        min_safe_distance: float = 5.0,
        progress_weight: float = 0.1,
    ):
        """
        Initialize reward function parameters.
        
        Args:
            collision_penalty: Large negative reward for collisions
            goal_reward: Positive reward for reaching goal
            time_penalty: Small negative reward per time step
            efficiency_weight: Weight for speed/efficiency component
            min_safe_distance: Minimum safe distance between vehicles (meters)
            progress_weight: Weight for dense progress-toward-goal reward
        """
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.time_penalty = time_penalty
        self.efficiency_weight = efficiency_weight
        self.min_safe_distance = min_safe_distance
        self.progress_weight = progress_weight
    
    def compute(
        self,
        state1: np.ndarray,
        state2: np.ndarray,
        goal1: Tuple[float, float],
        goal2: Tuple[float, float],
        collision: bool,
        done1: bool,
        done2: bool,
        reached_goal_1: bool = False,
        reached_goal_2: bool = False,
        prev_goal_dist_1: float = 0.0,
        prev_goal_dist_2: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Compute rewards for both vehicles.
        
        Args:
            state1: State of vehicle 1 [x, y, vx, vy]
            state2: State of vehicle 2 [x, y, vx, vy]
            goal1: Goal position for vehicle 1
            goal2: Goal position for vehicle 2
            collision: Whether collision occurred
            done1: Whether vehicle 1 reached goal
            done2: Whether vehicle 2 reached goal
            reached_goal_1: Whether vehicle 1 reached goal on this transition
            reached_goal_2: Whether vehicle 2 reached goal on this transition
            prev_goal_dist_1: Distance to goal for vehicle 1 at the previous step
            prev_goal_dist_2: Distance to goal for vehicle 2 at the previous step
        
        Returns:
            Tuple of (reward1, reward2)
        """
        reward1 = 0.0
        reward2 = 0.0
        
        # Collision penalty (shared by both vehicles)
        if collision:
            reward1 += self.collision_penalty
            reward2 += self.collision_penalty
            return reward1, reward2
        
        # Goal reward is awarded once on the transition into the goal state.
        if reached_goal_1:
            reward1 += self.goal_reward
        if reached_goal_2:
            reward2 += self.goal_reward
        
        # Time penalty (encourages faster completion)
        if not done1:
            reward1 += self.time_penalty
        if not done2:
            reward2 += self.time_penalty
        
        # Efficiency reward: bonus for maintaining good speed
        speed1 = np.sqrt(state1[2]**2 + state1[3]**2)
        speed2 = np.sqrt(state2[2]**2 + state2[3]**2)
        
        if not done1:
            # Reward for moving at reasonable speed (normalized to [0, 1])
            reward1 += self.efficiency_weight * (speed1 / 15.0)
        if not done2:
            reward2 += self.efficiency_weight * (speed2 / 15.0)
        
        # Safety reward: penalty for getting too close
        distance = np.sqrt((state1[0] - state2[0])**2 + (state1[1] - state2[1])**2)
        if distance < self.min_safe_distance:
            proximity_penalty = -1.0 * (1.0 - distance / self.min_safe_distance)
            reward1 += proximity_penalty
            reward2 += proximity_penalty

        # Progress reward: dense shaping toward goal (prev_dist - curr_dist > 0 when closer)
        curr_goal_dist_1 = np.sqrt((state1[0] - goal1[0])**2 + (state1[1] - goal1[1])**2)
        curr_goal_dist_2 = np.sqrt((state2[0] - goal2[0])**2 + (state2[1] - goal2[1])**2)
        if not done1 and prev_goal_dist_1 > 0.0:
            reward1 += self.progress_weight * (prev_goal_dist_1 - curr_goal_dist_1)
        if not done2 and prev_goal_dist_2 > 0.0:
            reward2 += self.progress_weight * (prev_goal_dist_2 - curr_goal_dist_2)

        return reward1, reward2
    
    def compute_single(
        self,
        state: np.ndarray,
        goal: Tuple[float, float],
        other_state: np.ndarray,
        collision: bool,
        done: bool
    ) -> float:
        """
        Compute reward for a single vehicle (convenience method).
        
        Args:
            state: Vehicle state
            goal: Goal position
            other_state: Other vehicle state
            collision: Whether collision occurred
            done: Whether vehicle reached goal
        
        Returns:
            Reward value
        """
        # Determine which vehicle this is and compute both rewards
        # Then return the appropriate one
        dummy_goal = (0.0, 0.0)
        r1, r2 = self.compute(state, other_state, goal, dummy_goal, collision, done, False)
        return r1


# ---------------------------------------------------------------------------
# Social Value Orientation (SVO)
# ---------------------------------------------------------------------------

# Common SVO angle presets (radians)
SVO_SELFISH: float = 0.0           # cares only about own reward
SVO_PROSOCIAL: float = np.pi / 4   # equal weight to self and other
SVO_ALTRUISTIC: float = np.pi / 2  # cares only about other's reward
SVO_COMPETITIVE: float = -np.pi / 4  # maximises own reward minus other's


def apply_svo_transform(r_self: float, r_other: float, phi: float) -> float:
    """
    Blend self-interested and other-regarding rewards via an SVO angle.

    Formula:
        r_svo = r_self * cos(phi) + r_other * sin(phi)

    Args:
        r_self: The agent's own raw reward.
        r_other: The other agent's raw reward.
        phi: SVO angle in radians. Use the SVO_* module constants for
             common presets (SVO_SELFISH, SVO_PROSOCIAL, etc.).

    Returns:
        Scalar SVO-weighted reward.
    """
    return float(r_self * np.cos(phi) + r_other * np.sin(phi))
