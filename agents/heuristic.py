"""
Heuristic policies for testing the intersection environment.
These provide simple baseline behaviors for autonomous vehicles.
"""

import numpy as np
from typing import Tuple


class HeuristicAgent:
    """Base class for heuristic agents."""
    
    def __init__(self, agent_id: str):
        """
        Initialize heuristic agent.
        
        Args:
            agent_id: Identifier for this agent ("agent_1" or "agent_2")
        """
        self.agent_id = agent_id
    
    def get_action(self, observation: np.ndarray) -> float:
        """
        Get action based on observation.
        
        Args:
            observation: Current observation [own_x, own_y, own_vx, own_vy, 
                                             other_x, other_y, other_vx, other_vy,
                                             goal_dist, other_dist]
        
        Returns:
            Action (acceleration) in [-1, 1]
        """
        raise NotImplementedError


class ConstantVelocityAgent(HeuristicAgent):
    """Agent that maintains constant velocity."""
    
    def __init__(self, agent_id: str, target_speed: float = 8.0):
        """
        Initialize constant velocity agent.
        
        Args:
            agent_id: Identifier for this agent
            target_speed: Desired speed in m/s
        """
        super().__init__(agent_id)
        self.target_speed = target_speed
    
    def get_action(self, observation: np.ndarray) -> float:
        """Maintain constant velocity."""
        # Extract own velocity
        vx, vy = observation[2], observation[3]
        current_speed = np.sqrt(vx**2 + vy**2)
        
        # Simple proportional control to maintain target speed
        speed_error = self.target_speed - current_speed
        action = np.clip(speed_error / 5.0, -1.0, 1.0)
        
        return action


class CautiousAgent(HeuristicAgent):
    """Agent that slows down when other vehicle is nearby."""
    
    def __init__(
        self, 
        agent_id: str, 
        target_speed: float = 8.0,
        safe_distance: float = 8.0,
        critical_distance: float = 5.0
    ):
        """
        Initialize cautious agent.
        
        Args:
            agent_id: Identifier for this agent
            target_speed: Desired speed when safe
            safe_distance: Distance to start slowing down
            critical_distance: Distance to brake hard
        """
        super().__init__(agent_id)
        self.target_speed = target_speed
        self.safe_distance = safe_distance
        self.critical_distance = critical_distance
    
    def get_action(self, observation: np.ndarray) -> float:
        """Slow down when other vehicle is near."""
        # Extract relevant information
        vx, vy = observation[2], observation[3]
        current_speed = np.sqrt(vx**2 + vy**2)
        distance_to_other = observation[9]
        
        # Determine target speed based on proximity
        if distance_to_other < self.critical_distance:
            # Emergency braking
            target = 0.0
        elif distance_to_other < self.safe_distance:
            # Gradual slowdown
            slowdown_factor = (distance_to_other - self.critical_distance) / \
                            (self.safe_distance - self.critical_distance)
            target = self.target_speed * slowdown_factor
        else:
            # Maintain normal speed
            target = self.target_speed
        
        # Proportional control
        speed_error = target - current_speed
        action = np.clip(speed_error / 5.0, -1.0, 1.0)
        
        return action


class AggressiveAgent(HeuristicAgent):
    """Agent that maintains high speed unless collision is imminent."""
    
    def __init__(
        self, 
        agent_id: str, 
        target_speed: float = 12.0,
        critical_distance: float = 3.0
    ):
        """
        Initialize aggressive agent.
        
        Args:
            agent_id: Identifier for this agent
            target_speed: Desired high speed
            critical_distance: Distance to emergency brake
        """
        super().__init__(agent_id)
        self.target_speed = target_speed
        self.critical_distance = critical_distance
    
    def get_action(self, observation: np.ndarray) -> float:
        """Maintain high speed, only brake if collision imminent."""
        # Extract relevant information
        vx, vy = observation[2], observation[3]
        current_speed = np.sqrt(vx**2 + vy**2)
        distance_to_other = observation[9]
        
        # Only brake if very close
        if distance_to_other < self.critical_distance:
            return -1.0  # Full braking
        
        # Otherwise maintain high speed
        speed_error = self.target_speed - current_speed
        action = np.clip(speed_error / 5.0, -1.0, 1.0)
        
        return action


class YieldAgent(HeuristicAgent):
    """Agent that yields to the other vehicle at the intersection."""
    
    def __init__(
        self, 
        agent_id: str,
        target_speed: float = 8.0,
        intersection_zone: float = 5.0,
        safe_distance: float = 10.0
    ):
        """
        Initialize yielding agent.
        
        Args:
            agent_id: Identifier for this agent
            target_speed: Normal driving speed
            intersection_zone: Radius around origin considered as intersection
            safe_distance: Distance to other vehicle to consider safe
        """
        super().__init__(agent_id)
        self.target_speed = target_speed
        self.intersection_zone = intersection_zone
        self.safe_distance = safe_distance
    
    def get_action(self, observation: np.ndarray) -> float:
        """Yield if other vehicle is at intersection first."""
        # Extract information
        own_x, own_y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        other_x, other_y = observation[4], observation[5]
        current_speed = np.sqrt(vx**2 + vy**2)
        distance_to_other = observation[9]
        
        # Distance to intersection center (origin)
        own_dist_to_intersection = np.sqrt(own_x**2 + own_y**2)
        other_dist_to_intersection = np.sqrt(other_x**2 + other_y**2)
        
        # Check if we're approaching intersection and other is at least as close.
        # Use a wider detection zone (zone+13) and >= comparison so the yield
        # agent brakes even when both vehicles are equidistant — which happens
        # whenever they start at the same distance and travel at the same speed.
        # The wider safe_distance (×2) ensures braking starts early enough to
        # actually slow before the conflict zone is reached.
        if own_dist_to_intersection < self.intersection_zone + 13:
            if other_dist_to_intersection <= own_dist_to_intersection:
                # Other vehicle is at least as close to intersection; yield
                if distance_to_other < self.safe_distance * 2:
                    return -0.5  # Moderate braking
        
        # Otherwise maintain speed
        speed_error = self.target_speed - current_speed
        action = np.clip(speed_error / 5.0, -1.0, 1.0)
        
        return action


def create_heuristic_pair(
    policy_1: str = "constant",
    policy_2: str = "cautious"
) -> Tuple[HeuristicAgent, HeuristicAgent]:
    """
    Create a pair of heuristic agents for testing.
    
    Args:
        policy_1: Policy type for agent 1 
                  ("constant", "cautious", "aggressive", "yield")
        policy_2: Policy type for agent 2
    
    Returns:
        Tuple of (agent_1, agent_2)
    """
    policy_map = {
        "constant": ConstantVelocityAgent,
        "cautious": CautiousAgent,
        "aggressive": AggressiveAgent,
        "yield": YieldAgent
    }
    
    agent_1 = policy_map[policy_1]("agent_1")
    agent_2 = policy_map[policy_2]("agent_2")
    
    return agent_1, agent_2
