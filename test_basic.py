"""Quick test to validate the environment implementation."""

import sys
import numpy as np
from env.intersection_env import IntersectionEnv
from agents.heuristic import create_heuristic_pair


def test_environment():
    """Test basic environment functionality."""
    print("Testing IntersectionEnv...")
    
    # Create environment
    env = IntersectionEnv(dt=0.1, max_steps=50)
    
    # Test reset
    obs, info = env.reset()
    assert "agent_1" in obs
    assert "agent_2" in obs
    assert obs["agent_1"].shape == (10,)
    assert obs["agent_2"].shape == (10,)
    print("✓ Environment reset successful")
    
    # Test step
    action = np.array([0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    assert "agent_1" in reward
    assert "agent_2" in reward
    print("✓ Environment step successful")
    
    # Test episode
    obs, info = env.reset()
    for _ in range(10):
        action = np.array([0.5, 0.5])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print("✓ Episode execution successful")
    
    print("\nEnvironment test passed!")


def test_heuristic_agents():
    """Test heuristic agents."""
    print("\nTesting Heuristic Agents...")
    
    env = IntersectionEnv()
    agent_1, agent_2 = create_heuristic_pair("cautious", "cautious")
    
    obs, info = env.reset()
    
    # Test action generation
    action_1 = agent_1.get_action(obs["agent_1"])
    action_2 = agent_2.get_action(obs["agent_2"])
    
    assert -1.0 <= action_1 <= 1.0
    assert -1.0 <= action_2 <= 1.0
    print("✓ Heuristic agents generate valid actions")
    
    # Run short episode
    for _ in range(10):
        action_1 = agent_1.get_action(obs["agent_1"])
        action_2 = agent_2.get_action(obs["agent_2"])
        action = np.array([action_1, action_2])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    print("✓ Heuristic agents can control environment")
    print("\nHeuristic agents test passed!")


def test_collision_detection():
    """Test collision detection."""
    print("\nTesting Collision Detection...")
    
    from env.dynamics import VehicleDynamics
    
    dynamics = VehicleDynamics()
    
    # Test no collision
    state1 = np.array([0.0, 0.0, 1.0, 0.0])
    state2 = np.array([10.0, 0.0, 1.0, 0.0])
    assert not dynamics.check_collision(state1, state2)
    print("✓ Correctly detects no collision")
    
    # Test collision
    state1 = np.array([0.0, 0.0, 1.0, 0.0])
    state2 = np.array([1.0, 0.0, -1.0, 0.0])
    assert dynamics.check_collision(state1, state2)
    print("✓ Correctly detects collision")
    
    print("\nCollision detection test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING VALIDATION TESTS")
    print("=" * 60)
    
    try:
        test_environment()
        test_heuristic_agents()
        test_collision_detection()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now run: python demo/render_episode.py")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
