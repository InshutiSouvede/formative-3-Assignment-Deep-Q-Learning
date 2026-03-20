# DQN Hyperparameter Tuning Documentation

## Environment: Demon Attack (ALE/DemonAttack-v5)

### Policy Comparison: CNNPolicy vs MlpPolicy

| Policy        | Suitability        | Reason                                                                                                         |
| ------------- | ------------------ | -------------------------------------------------------------------------------------------------------------- |
| **CNNPolicy** | ✅ Recommended     | Designed for image-based inputs; uses convolutional layers to extract spatial features from game frames        |
| **MlpPolicy** | ❌ Not Recommended | Designed for vector inputs; would flatten images and lose spatial relationships, resulting in poor performance |

**Conclusion**: For Atari games like Demon Attack that use pixel-based observations, **CNNPolicy** is the only sensible choice as it preserves spatial information through convolutional operations.

---

### Hyperparameter Tuning Experiments

| #   | Hyperparameter Set                                                                       | Noted Behavior                                                                                                                                                                                                                                     |
| --- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1    | **Baseline configuration.** Balanced learning with moderate exploration. Model learns basic enemy avoidance and shooting patterns. Stable training with gradual reward increase. Good starting point for comparison.                               |
| 2   | lr=5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1    | **Higher learning rate.** Faster initial learning but may become unstable. Can lead to overshooting optimal Q-values. May exhibit reward oscillation in later training stages. Better for quick experiments but less consistent final performance. |
| 3   | lr=1e-5, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1    | **Lower learning rate.** Very slow but stable learning. Requires more timesteps to see improvement. Lower risk of divergence. Good for fine-tuning but impractical for initial training. Agent takes longer to learn basic survival.               |
| 4   | lr=1e-4, gamma=0.95, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1    | **Lower gamma (short-term focus).** Agent prioritizes immediate rewards over future rewards. Good for quick point-scoring but may not develop long-term survival strategies. Less planning ahead, more reactive gameplay.                          |
| 5   | lr=1e-4, gamma=0.999, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1   | **Higher gamma (long-term focus).** Agent considers distant future rewards. Can lead to better strategic planning. May require longer training. More consistent high scores but slower initial progress. Better for survival-focused play.         |
| 6   | lr=1e-4, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1    | **Larger batch size.** More stable gradient updates due to larger sample size. Reduces variance in learning. May require more memory. Slightly slower per-update but more reliable learning progression.                                           |
| 7   | lr=1e-4, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1    | **Smaller batch size.** Faster updates but higher variance. Can lead to noisier learning curves. May escape local minima better due to stochasticity. Less memory usage but potentially unstable.                                                  |
| 8   | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.3    | **Slower exploration decay.** Agent explores longer before exploiting learned policy. Better state-space coverage. May discover better strategies but takes longer to converge. Good for complex environments.                                     |
| 9   | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.1     | **Higher final epsilon.** Maintains more exploration even after training. Prevents getting stuck in suboptimal policies. May result in slightly lower peak performance but more robust behavior. Good for dynamic environments.                    |
| 10  | lr=2.5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.15 | **Optimized configuration.** Balanced learning rate (slightly higher for faster learning). Very low final epsilon for maximum exploitation. Moderate exploration decay. Expected to achieve best overall performance with stable learning.         |

---

### Hyperparameter Descriptions

| Parameter              | Description                               | Effect on Training                                                                              |
| ---------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Learning Rate (lr)** | Step size for gradient updates            | Higher = faster but potentially unstable learning; Lower = slower but more stable               |
| **Gamma (γ)**          | Discount factor for future rewards        | Higher = more weight on future rewards (long-term planning); Lower = focus on immediate rewards |
| **Batch Size**         | Number of experiences sampled per update  | Larger = more stable gradients but slower; Smaller = faster but noisier updates                 |
| **Epsilon Start**      | Initial exploration probability           | Usually 1.0 (100% random actions initially)                                                     |
| **Epsilon End**        | Final exploration probability             | Lower = more exploitation; Higher = maintains some exploration                                  |
| **Epsilon Decay**      | Fraction of training for epsilon to decay | Smaller = faster transition to exploitation; Larger = longer exploration phase                  |

---

### Recommendations for Demon Attack

1. **Best Starting Configuration**: Use the optimized configuration (#10) for best results
2. **Memory-Constrained Systems**: Use smaller batch size (16-32) with lower learning rate
3. **Quick Experiments**: Use higher learning rate (5e-4) with fewer timesteps
4. **Thorough Training**: Use lower learning rate (1e-4 or less) with more timesteps (1M+)

---

### Training Tips

1. **Frame Stacking**: We use 4 stacked frames to provide temporal information (motion detection)
2. **Atari Wrappers**: Essential for proper preprocessing (grayscale, resize, frame skip)
3. **Buffer Size**: Larger replay buffers (100K+) allow for more diverse experience sampling
4. **Target Network**: Updated every 10K steps to stabilize Q-learning
5. **Learning Starts**: Wait for 50K steps before learning to fill replay buffer

---

### Running Experiments

```bash
# Single training run with default parameters
python train.py --mode single --timesteps 500000

# Run all hyperparameter experiments
python train.py --mode experiments

# Compare policies (CNNPolicy vs MlpPolicy analysis)
python train.py --mode compare

# Custom hyperparameters
python train.py --mode single --lr 0.0001 --gamma 0.99 --batch-size 32 --timesteps 1000000
```

### Playing with Trained Model

```bash
# Play with default model
python play.py --episodes 5

# Play with specific model
python play.py --model ./logs/optimized_*/best_model.zip --episodes 3

# Evaluate without rendering (faster)
python play.py --evaluate --episodes 20

# Find and play with best model
python play.py --best --episodes 5
```
