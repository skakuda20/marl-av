"""
Two-vehicle intersection environment using Gymnasium.
Vehicles approach an intersection from perpendicular directions.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional

from env.dynamics import VehicleDynamics
from env.rewards import RewardFunction
from env.scenarios import (
    DEFAULT_SCENARIO_DISTRIBUTIONS,
    ScenarioDistribution,
    build_scenario_distributions,
    priority_value,
    sample_scenario,
)


class IntersectionEnv(gym.Env):
    """
    Two-vehicle intersection environment.
    
    Observation Space (per vehicle):
        - Own position (x, y)
        - Own velocity (vx, vy)
        - Other vehicle position (x, y)
        - Other vehicle velocity (vx, vy)
        - Distance to goal
        - Distance to other vehicle
    
    Action Space (per vehicle):
        - Continuous acceleration in [-1, 1]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(
        self,
        dt: float = 0.1,
        max_steps: int = 200,
        goal_threshold: float = 2.0,
        render_mode: Optional[str] = None,
        scenario_split: str = "train",
        scenario_distributions: Optional[Dict[str, ScenarioDistribution]] = None,
        localize_observations: bool = True,
    ):
        """
        Initialize the intersection environment.
        
        Args:
            dt: Time step for dynamics (seconds)
            max_steps: Maximum episode length
            goal_threshold: Distance to goal for completion (meters)
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.dt = dt
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold
        self.render_mode = render_mode
        self.scenario_split = scenario_split
        self.scenario_distributions = build_scenario_distributions(scenario_distributions) if isinstance(scenario_distributions, dict) else (scenario_distributions or DEFAULT_SCENARIO_DISTRIBUTIONS)
        self.localize_observations = localize_observations
        
        # Initialize dynamics and rewards
        self.dynamics = VehicleDynamics(dt=dt)
        self.reward_fn = RewardFunction()
        
        # Define action and observation spaces
        # Action: acceleration for each vehicle
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation: [own_x, own_y, own_vx, own_vy, other_x, other_y, 
        #                other_vx, other_vy, goal_dist, other_dist, own_priority]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2, 11), dtype=np.float32
        )
        
        # Define intersection layout
        # Vehicle 1 starts from the left, moves right
        # Vehicle 2 starts from the bottom, moves up
        self.start_pos_1 = np.array([-20.0, 0.0])
        self.start_pos_2 = np.array([0.0, -20.0])
        self.goal_pos_1 = (20.0, 0.0)
        self.goal_pos_2 = (0.0, 20.0)
        
        # Initial velocities (moving toward intersection)
        self.start_vel_1 = np.array([8.0, 0.0])
        self.start_vel_2 = np.array([0.0, 8.0])
        
        # State variables
        self.state_1 = None
        self.state_2 = None
        self.steps = 0
        # Persistent goal-reached flags (latch True, never reset mid-episode)
        self._done_1: bool = False
        self._done_2: bool = False
        # Previous goal distances for progress reward computation
        self._prev_goal_dist_1: float = 0.0
        self._prev_goal_dist_2: float = 0.0
        self.current_scenario: Optional[Dict[str, object]] = None

    def _scenario_center(self) -> np.ndarray:
        if self.current_scenario is None:
            return np.zeros(2, dtype=np.float32)
        return np.array(self.current_scenario.get("center", (0.0, 0.0)), dtype=np.float32)

    def _localize_state(self, state: np.ndarray) -> np.ndarray:
        localized = np.array(state, dtype=np.float32, copy=True)
        if self.localize_observations:
            localized[:2] -= self._scenario_center()
        return localized

    def _apply_other_agent_noise(self, obs: np.ndarray) -> np.ndarray:
        noisy = np.array(obs, dtype=np.float32, copy=True)
        noise_std = float(self.current_scenario.get("observation_noise_std", 0.0))
        occlusion_probability = float(self.current_scenario.get("occlusion_probability", 0.0))

        if noise_std > 0.0:
            noisy[4:8] += self.np_random.normal(0.0, noise_std, size=4).astype(np.float32)
            noisy[9] = max(0.0, float(noisy[9] + self.np_random.normal(0.0, noise_std * 2.0)))

        if occlusion_probability > 0.0 and float(self.np_random.random()) < occlusion_probability:
            noisy[4:8] = 0.0
            noisy[9] = max(noisy[9], 20.0)

        return noisy
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Dictionary with observations for both agents
            info: Dictionary with additional information
        """
        super().reset(seed=seed)

        split = self.scenario_split
        if options is not None:
            split = options.get("scenario_split", split)

        scenario = None if options is None else options.get("scenario")
        if scenario is None:
            distribution = self.scenario_distributions[split]
            scenario = sample_scenario(self.np_random, distribution)

        self.current_scenario = dict(scenario)
        self.current_scenario["scenario_split"] = split

        self.start_pos_1 = np.array(scenario["start_pos_1"], dtype=np.float32)
        self.start_pos_2 = np.array(scenario["start_pos_2"], dtype=np.float32)
        self.goal_pos_1 = tuple(scenario["goal_pos_1"])
        self.goal_pos_2 = tuple(scenario["goal_pos_2"])
        self.start_vel_1 = np.array(scenario["start_vel_1"], dtype=np.float32)
        self.start_vel_2 = np.array(scenario["start_vel_2"], dtype=np.float32)

        # Reset vehicle states
        self.state_1 = np.concatenate([self.start_pos_1, self.start_vel_1])
        self.state_2 = np.concatenate([self.start_pos_2, self.start_vel_2])
        self.steps = 0
        self._done_1 = False
        self._done_2 = False
        self._prev_goal_dist_1 = self.dynamics.distance_to_point(self.state_1, self.goal_pos_1)
        self._prev_goal_dist_2 = self.dynamics.distance_to_point(self.state_2, self.goal_pos_2)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """
        Execute one time step.
        
        Args:
            action: Array of shape (2,) with actions for both vehicles
        
        Returns:
            observation: New observations for both agents
            reward: Rewards for both agents
            terminated: Whether episode ended due to goal/collision
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        # Extract actions
        action_1 = action[0]
        action_2 = action[1]

        # Only update physics for vehicles that haven't reached their goal yet
        if not self._done_1:
            self.state_1 = self.dynamics.step(self.state_1, action_1)
        if not self._done_2:
            self.state_2 = self.dynamics.step(self.state_2, action_2)

        self.steps += 1

        # Check termination conditions
        collision = self.dynamics.check_collision(self.state_1, self.state_2)

        dist_to_goal_1 = self.dynamics.distance_to_point(self.state_1, self.goal_pos_1)
        dist_to_goal_2 = self.dynamics.distance_to_point(self.state_2, self.goal_pos_2)

        prev_done_1 = self._done_1
        prev_done_2 = self._done_2

        # Latch goal flags — once True they stay True for the rest of the episode
        if dist_to_goal_1 < self.goal_threshold:
            self._done_1 = True
        if dist_to_goal_2 < self.goal_threshold:
            self._done_2 = True

        done_1 = self._done_1
        done_2 = self._done_2
        reached_goal_1 = (not prev_done_1) and done_1
        reached_goal_2 = (not prev_done_2) and done_2

        # Episode terminates if collision or both vehicles have reached their goals
        terminated = collision or (done_1 and done_2)
        truncated = self.steps >= self.max_steps

        # Compute rewards, passing previous goal distances for progress shaping
        reward_1, reward_2 = self.reward_fn.compute(
            self.state_1, self.state_2,
            self.goal_pos_1, self.goal_pos_2,
            collision, done_1, done_2,
            reached_goal_1=reached_goal_1,
            reached_goal_2=reached_goal_2,
            prev_goal_dist_1=self._prev_goal_dist_1,
            prev_goal_dist_2=self._prev_goal_dist_2,
        )

        # Update previous distances for next step
        self._prev_goal_dist_1 = dist_to_goal_1
        self._prev_goal_dist_2 = dist_to_goal_2

        obs = self._get_obs()
        reward = {"agent_1": reward_1, "agent_2": reward_2}
        info = self._get_info()
        info["collision"] = collision
        info["done_1"] = done_1
        info["done_2"] = done_2
        info["reached_goal_1"] = reached_goal_1
        info["reached_goal_2"] = reached_goal_2

        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Get observations for both agents.
        
        Returns:
            Dictionary with observations for each agent
        """
        # Calculate distances
        dist_to_goal_1 = self.dynamics.distance_to_point(self.state_1, self.goal_pos_1)
        dist_to_goal_2 = self.dynamics.distance_to_point(self.state_2, self.goal_pos_2)
        dist_between = np.sqrt(
            (self.state_1[0] - self.state_2[0])**2 + 
            (self.state_1[1] - self.state_2[1])**2
        )
        
        priority_1 = priority_value(self.current_scenario["priority"], "agent_1")
        priority_2 = priority_value(self.current_scenario["priority"], "agent_2")
        state_1_obs = self._localize_state(self.state_1)
        state_2_obs = self._localize_state(self.state_2)

        # Agent 1 observation: own state + other state + distances
        obs_1 = np.array([
            state_1_obs[0], state_1_obs[1],  # own position
            state_1_obs[2], state_1_obs[3],  # own velocity
            state_2_obs[0], state_2_obs[1],  # other position
            state_2_obs[2], state_2_obs[3],  # other velocity
            dist_to_goal_1,  # distance to own goal
            dist_between,     # distance to other vehicle
            priority_1,       # own priority hint
        ], dtype=np.float32)
        
        # Agent 2 observation: own state + other state + distances
        obs_2 = np.array([
            state_2_obs[0], state_2_obs[1],  # own position
            state_2_obs[2], state_2_obs[3],  # own velocity
            state_1_obs[0], state_1_obs[1],  # other position
            state_1_obs[2], state_1_obs[3],  # other velocity
            dist_to_goal_2,  # distance to own goal
            dist_between,     # distance to other vehicle
            priority_2,       # own priority hint
        ], dtype=np.float32)

        return {
            "agent_1": self._apply_other_agent_noise(obs_1),
            "agent_2": self._apply_other_agent_noise(obs_2),
        }
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            "steps": self.steps,
            "state_1": self.state_1.copy(),
            "state_2": self.state_2.copy(),
            "scenario": dict(self.current_scenario) if self.current_scenario is not None else None,
        }
    
    def render(self):
        """Render the environment (placeholder for now)."""
        if self.render_mode == "human":
            print(f"Step {self.steps}:")
            print(f"  Vehicle 1: pos=({self.state_1[0]:.2f}, {self.state_1[1]:.2f}), "
                  f"vel=({self.state_1[2]:.2f}, {self.state_1[3]:.2f})")
            print(f"  Vehicle 2: pos=({self.state_2[0]:.2f}, {self.state_2[1]:.2f}), "
                  f"vel=({self.state_2[2]:.2f}, {self.state_2[3]:.2f})")
    
    def close(self):
        """Clean up resources."""
        pass
