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
        render_mode: Optional[str] = None
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
        
        # Initialize dynamics and rewards
        self.dynamics = VehicleDynamics(dt=dt)
        self.reward_fn = RewardFunction()
        
        # Define action and observation spaces
        # Action: acceleration for each vehicle
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation: [own_x, own_y, own_vx, own_vy, other_x, other_y, 
        #                other_vx, other_vy, goal_dist, other_dist]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2, 10), dtype=np.float32
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
        
        # Reset vehicle states
        self.state_1 = np.concatenate([self.start_pos_1, self.start_vel_1])
        self.state_2 = np.concatenate([self.start_pos_2, self.start_vel_2])
        self.steps = 0
        
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
        
        # Update states using dynamics
        self.state_1 = self.dynamics.step(self.state_1, action_1)
        self.state_2 = self.dynamics.step(self.state_2, action_2)
        
        self.steps += 1
        
        # Check termination conditions
        collision = self.dynamics.check_collision(self.state_1, self.state_2)
        
        dist_to_goal_1 = self.dynamics.distance_to_point(self.state_1, self.goal_pos_1)
        dist_to_goal_2 = self.dynamics.distance_to_point(self.state_2, self.goal_pos_2)
        
        done_1 = dist_to_goal_1 < self.goal_threshold
        done_2 = dist_to_goal_2 < self.goal_threshold
        
        # Episode terminates if collision or both reach goal
        terminated = collision or (done_1 and done_2)
        truncated = self.steps >= self.max_steps
        
        # Compute rewards
        reward_1, reward_2 = self.reward_fn.compute(
            self.state_1, self.state_2,
            self.goal_pos_1, self.goal_pos_2,
            collision, done_1, done_2
        )
        
        obs = self._get_obs()
        reward = {"agent_1": reward_1, "agent_2": reward_2}
        info = self._get_info()
        info["collision"] = collision
        info["done_1"] = done_1
        info["done_2"] = done_2
        
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
        
        # Agent 1 observation: own state + other state + distances
        obs_1 = np.array([
            self.state_1[0], self.state_1[1],  # own position
            self.state_1[2], self.state_1[3],  # own velocity
            self.state_2[0], self.state_2[1],  # other position
            self.state_2[2], self.state_2[3],  # other velocity
            dist_to_goal_1,  # distance to own goal
            dist_between      # distance to other vehicle
        ], dtype=np.float32)
        
        # Agent 2 observation: own state + other state + distances
        obs_2 = np.array([
            self.state_2[0], self.state_2[1],  # own position
            self.state_2[2], self.state_2[3],  # own velocity
            self.state_1[0], self.state_1[1],  # other position
            self.state_1[2], self.state_1[3],  # other velocity
            dist_to_goal_2,  # distance to own goal
            dist_between      # distance to other vehicle
        ], dtype=np.float32)
        
        return {"agent_1": obs_1, "agent_2": obs_2}
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            "steps": self.steps,
            "state_1": self.state_1.copy(),
            "state_2": self.state_2.copy()
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
