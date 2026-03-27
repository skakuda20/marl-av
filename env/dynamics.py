"""
Simple deterministic vehicle dynamics for intersection simulation.
Uses a kinematic bicycle model with discrete time steps.
"""

import numpy as np
from typing import Tuple


class VehicleDynamics:
    """Deterministic kinematic vehicle model."""
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize vehicle dynamics.
        
        Args:
            dt: Time step in seconds
        """
        self.dt = dt
        self.max_acceleration = 3.0  # m/s^2
        self.max_deceleration = 5.0  # m/s^2
        self.max_velocity = 15.0  # m/s (~54 km/h)
        self.min_velocity = 0.0  # m/s
        self.vehicle_length = 4.5  # meters
        self.vehicle_width = 2.0  # meters
    
    def step(self, state: np.ndarray, action: float) -> np.ndarray:
        """
        Update vehicle state given an action.
        
        State: [x, y, vx, vy] where:
            x, y: position (meters)
            vx, vy: velocity (m/s)
        
        Action: acceleration in [-1, 1] where:
            -1: maximum braking
            1: maximum acceleration
            0: maintain speed
        
        Args:
            state: Current state [x, y, vx, vy]
            action: Acceleration command in [-1, 1]
        
        Returns:
            New state [x, y, vx, vy]
        """
        x, y, vx, vy = state
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Convert action to acceleration
        if action >= 0:
            acceleration = action * self.max_acceleration
        else:
            acceleration = action * self.max_deceleration
        
        # Current speed and direction
        speed = np.sqrt(vx**2 + vy**2)
        
        if speed > 1e-6:  # Avoid division by zero
            direction_x = vx / speed
            direction_y = vy / speed
        else:
            # If stationary, maintain current direction or default to x-axis
            direction_x = 1.0 if abs(vx) > abs(vy) else 0.0
            direction_y = 0.0 if abs(vx) > abs(vy) else 1.0
        
        # Update speed
        new_speed = speed + acceleration * self.dt
        new_speed = np.clip(new_speed, self.min_velocity, self.max_velocity)
        
        # Update velocity components
        new_vx = new_speed * direction_x
        new_vy = new_speed * direction_y
        
        # Update position
        new_x = x + new_vx * self.dt
        new_y = y + new_vy * self.dt
        
        return np.array([new_x, new_y, new_vx, new_vy])
    
    def get_bounding_box(self, state: np.ndarray) -> np.ndarray:
        """
        Get axis-aligned bounding box for collision detection.
        
        Args:
            state: Vehicle state [x, y, vx, vy]
        
        Returns:
            Bounding box [x_min, y_min, x_max, y_max]
        """
        x, y, vx, vy = state
        
        # Simplified: axis-aligned box centered at position
        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2
        
        return np.array([
            x - half_width,
            y - half_width,
            x + half_width,
            y + half_width
        ])
    
    def check_collision(self, state1: np.ndarray, state2: np.ndarray) -> bool:
        """
        Check if two vehicles collide.
        
        Args:
            state1: First vehicle state
            state2: Second vehicle state
        
        Returns:
            True if collision detected
        """
        box1 = self.get_bounding_box(state1)
        box2 = self.get_bounding_box(state2)
        
        # Check for overlap in both axes
        x_overlap = box1[2] >= box2[0] and box2[2] >= box1[0]
        y_overlap = box1[3] >= box2[1] and box2[3] >= box1[1]
        
        return x_overlap and y_overlap
    
    def get_speed(self, state: np.ndarray) -> float:
        """Get current speed of vehicle."""
        return np.sqrt(state[2]**2 + state[3]**2)
    
    def distance_to_point(self, state: np.ndarray, point: Tuple[float, float]) -> float:
        """Calculate Euclidean distance from vehicle to a point."""
        return np.sqrt((state[0] - point[0])**2 + (state[1] - point[1])**2)
