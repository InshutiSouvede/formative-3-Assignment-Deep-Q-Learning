"""
DQN Agent Training Script for Demon Attack (Atari)
This script trains a DQN agent using Stable Baselines3 on the Demon Attack environment.
"""

from datetime import datetime
import json
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import DQN
import os
import gymnasium as gym
import ale_py

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to log training metrics including rewards and episode lengths.
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0
        self.current_length = 0

    def _on_step(self) -> bool:
        # Track rewards and episode info
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    if self.verbose > 0:
                        print(f"Episode {len(self.episode_rewards)}: "
                              f"Reward = {info['episode']['r']:.2f}, "
                              f"Length = {info['episode']['l']}")
        return True

    def _on_training_end(self) -> None:
        # Save metrics to file
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'mean_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            'std_reward': float(np.std(self.episode_rewards)) if self.episode_rewards else 0,
            'mean_length': float(np.mean(self.episode_lengths)) if self.episode_lengths else 0,
        }
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n=== Training Complete ===")
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(
            f"Mean Reward: {metrics['mean_reward']:.2f} (+/- {metrics['std_reward']:.2f})")
        print(f"Mean Episode Length: {metrics['mean_length']:.2f}")


def make_atari_env(env_id: str = "ALE/DemonAttack-v5", render_mode: str = None):
    """
    Create and wrap the Atari environment with proper preprocessing.
    """
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = AtariWrapper(env)
        env = Monitor(env)
        return env
    return _init


def train_dqn(
    policy: str = "CnnPolicy",
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    batch_size: int = 32,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    exploration_fraction: float = 0.1,
    buffer_size: int = 100000,
    learning_starts: int = 50000,
    target_update_interval: int = 10000,
    total_timesteps: int = 1000000,
    log_dir: str = "./logs",
    model_save_path: str = "./dqn_model",
    experiment_name: str = "default",
    verbose: int = 1
):
    """
    Train a DQN agent with specified hyperparameters.

    Args:
        policy: "CnnPolicy" or "MlpPolicy"
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor for future rewards
        batch_size: Number of samples per gradient update
        exploration_initial_eps: Initial epsilon for exploration
        exploration_final_eps: Final epsilon for exploration
        exploration_fraction: Fraction of training for epsilon decay
        buffer_size: Size of the replay buffer
        learning_starts: How many steps before learning starts
        target_update_interval: Update target network every N steps
        total_timesteps: Total training timesteps
        log_dir: Directory for tensorboard logs
        model_save_path: Path to save the trained model
        experiment_name: Name for this experiment
        verbose: Verbosity level

    Returns:
        Trained DQN model and training metrics
    """

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Log hyperparameters
    hyperparams = {
        "policy": policy,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "batch_size": batch_size,
        "exploration_initial_eps": exploration_initial_eps,
        "exploration_final_eps": exploration_final_eps,
        "exploration_fraction": exploration_fraction,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "target_update_interval": target_update_interval,
        "total_timesteps": total_timesteps,
        "experiment_name": experiment_name
    }

    with open(os.path.join(exp_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2)

    print("\n" + "="*60)
    print(f"Training DQN Agent - Experiment: {experiment_name}")
    print("="*60)
    print(f"Policy: {policy}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Batch Size: {batch_size}")
    print(f"Epsilon: {exploration_initial_eps} -> {exploration_final_eps} "
          f"(decay fraction: {exploration_fraction})")
    print(f"Total Timesteps: {total_timesteps}")
    print("="*60 + "\n")

    # Create the environment
    env = DummyVecEnv([make_atari_env("ALE/DemonAttack-v5")])
    env = VecFrameStack(env, n_stack=4)  # Stack 4 frames for temporal info

    # Create evaluation environment
    eval_env = DummyVecEnv([make_atari_env("ALE/DemonAttack-v5")])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # Initialize the DQN agent
    model = DQN(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exploration_fraction,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        target_update_interval=target_update_interval,
        tensorboard_log=exp_dir,
        verbose=verbose,
        device="auto"  # Will use GPU if available
    )

    # Create callbacks
    metrics_callback = TrainingMetricsCallback(log_dir=exp_dir, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=exp_dir,
        log_path=exp_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[metrics_callback, eval_callback],
        progress_bar=True
    )

    # Save the final model
    final_model_path = f"{model_save_path}.zip"
    model.save(model_save_path)
    print(f"\nModel saved to: {final_model_path}")

    # Clean up
    env.close()
    eval_env.close()

    return model, metrics_callback.episode_rewards, metrics_callback.episode_lengths


def run_hyperparameter_experiments():
    """
    Run multiple experiments with different hyperparameter configurations.
    This function tests various combinations and logs results.
    """

    # Define hyperparameter configurations to test
    experiments = [
        # Experiment 1: Baseline configuration
        {
            "name": "baseline",
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 2: Higher learning rate
        {
            "name": "high_lr",
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 3: Lower learning rate
        {
            "name": "low_lr",
            "learning_rate": 1e-5,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 4: Lower gamma (more short-term focused)
        {
            "name": "low_gamma",
            "learning_rate": 1e-4,
            "gamma": 0.95,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 5: Higher gamma (more long-term focused)
        {
            "name": "high_gamma",
            "learning_rate": 1e-4,
            "gamma": 0.999,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 6: Larger batch size
        {
            "name": "large_batch",
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 7: Smaller batch size
        {
            "name": "small_batch",
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 16,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 8: Slower exploration decay
        {
            "name": "slow_exploration",
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.3,
            "total_timesteps": 500000
        },
        # Experiment 9: Higher final epsilon (more exploration)
        {
            "name": "high_final_eps",
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.1,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 10: Optimized configuration
        {
            "name": "optimized",
            "learning_rate": 2.5e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.01,
            "exploration_fraction": 0.15,
            "total_timesteps": 500000
        }
    ]

    results = []

    for exp in experiments:
        print(f"\n{'#'*60}")
        print(f"Running Experiment: {exp['name']}")
        print(f"{'#'*60}")

        try:
            model, rewards, lengths = train_dqn(
                policy="CnnPolicy",  # CNNPolicy is better for image-based Atari games
                learning_rate=exp["learning_rate"],
                gamma=exp["gamma"],
                batch_size=exp["batch_size"],
                exploration_initial_eps=exp["exploration_initial_eps"],
                exploration_final_eps=exp["exploration_final_eps"],
                exploration_fraction=exp["exploration_fraction"],
                total_timesteps=exp["total_timesteps"],
                experiment_name=exp["name"]
            )

            result = {
                "experiment": exp["name"],
                "hyperparameters": exp,
                "mean_reward": float(np.mean(rewards)) if rewards else 0,
                "std_reward": float(np.std(rewards)) if rewards else 0,
                "max_reward": float(np.max(rewards)) if rewards else 0,
                "total_episodes": len(rewards),
                "mean_episode_length": float(np.mean(lengths)) if lengths else 0
            }
            results.append(result)

        except Exception as e:
            print(f"Experiment {exp['name']} failed: {e}")
            results.append({
                "experiment": exp["name"],
                "error": str(e)
            })

    # Save all results
    with open('./logs/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<20} {'Mean Reward':<15} {'Max Reward':<15} {'Episodes':<10}")
    print("-"*80)
    for r in results:
        if 'error' not in r:
            print(
                f"{r['experiment']:<20} {r['mean_reward']:<15.2f} {r['max_reward']:<15.2f} {r['total_episodes']:<10}")

    return results


def compare_policies():
    """
    Compare MLPPolicy vs CNNPolicy performance.
    Note: For Atari games with image observations, CNNPolicy is typically better.
    MLPPolicy would require flattened observations and is not recommended for images.
    """
    print("\n" + "="*60)
    print("POLICY COMPARISON: CNNPolicy vs MlpPolicy")
    print("="*60)
    print("\nNote: For Atari games with image-based observations:")
    print("- CNNPolicy: Designed for image inputs, uses convolutional layers")
    print("- MlpPolicy: Designed for vector inputs, would require flattening images")
    print("\nCNNPolicy is the recommended choice for Atari environments.")
    print("MlpPolicy would result in poor performance due to loss of spatial information.")
    print("="*60 + "\n")

    # Train with CNNPolicy (recommended)
    print("Training with CNNPolicy (Recommended for Atari)...")
    model_cnn, rewards_cnn, lengths_cnn = train_dqn(
        policy="CnnPolicy",
        total_timesteps=200000,
        experiment_name="cnn_policy"
    )

    return {
        "cnn_policy": {
            "mean_reward": float(np.mean(rewards_cnn)) if rewards_cnn else 0,
            "max_reward": float(np.max(rewards_cnn)) if rewards_cnn else 0
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train DQN Agent for Demon Attack")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "experiments", "compare"],
                        help="Training mode: single run, hyperparameter experiments, or policy comparison")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Total timesteps for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--eps-start", type=float, default=1.0,
                        help="Initial exploration epsilon")
    parser.add_argument("--eps-end", type=float, default=0.05,
                        help="Final exploration epsilon")
    parser.add_argument("--eps-fraction", type=float, default=0.1,
                        help="Fraction of training for epsilon decay")

    args = parser.parse_args()

    if args.mode == "single":
        # Single training run with specified hyperparameters
        print("Running single training session...")
        model, rewards, lengths = train_dqn(
            policy="CnnPolicy",
            learning_rate=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            exploration_initial_eps=args.eps_start,
            exploration_final_eps=args.eps_end,
            exploration_fraction=args.eps_fraction,
            total_timesteps=args.timesteps,
            experiment_name="single_run"
        )

    elif args.mode == "experiments":
        # Run all hyperparameter experiments
        print("Running hyperparameter tuning experiments...")
        results = run_hyperparameter_experiments()

    elif args.mode == "compare":
        # Compare policies
        print("Running policy comparison...")
        results = compare_policies()
