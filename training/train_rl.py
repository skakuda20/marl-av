"""
Training script for RL agents (placeholder for future work).

This module will contain:
- Training loop
- Experience collection
- Model checkpointing
- Tensorboard logging
"""

from env.intersection_env import IntersectionEnv


def train_rl_agent(
    num_episodes: int = 1000,
    save_path: str = "models/rl_agent.pt"
):
    """
    Train RL agent on intersection environment.
    
    Args:
        num_episodes: Number of training episodes
        save_path: Path to save trained model
    """
    # TODO: Initialize environment
    # TODO: Initialize RL agent
    # TODO: Implement training loop
    # TODO: Add logging and checkpointing
    raise NotImplementedError("RL training not yet implemented")


if __name__ == "__main__":
    print("Training script - to be implemented")
    print("Future work: implement PPO/SAC training pipeline")
