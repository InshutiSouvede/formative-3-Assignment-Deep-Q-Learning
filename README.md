# DQN Agent for Demon Attack (Atari)

A Deep Q-Network (DQN) implementation using Stable Baselines3 to play the Demon Attack Atari game.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Accept Atari ROM License (First Time Only)

```bash
pip install gymnasium[accept-rom-license]
```

## Usage

### Training the Agent

```bash
# Basic training (500,000 timesteps)
python train.py --mode single --timesteps 500000

# Run hyperparameter experiments (10 configurations)
python train.py --mode experiments

# Custom training with specific hyperparameters
python train.py --mode single --lr 0.0001 --gamma 0.99 --batch-size 32 --timesteps 1000000
```

### Playing with Trained Agent

```bash
# Play 5 episodes with rendering
python play.py --episodes 5

# Load specific model
python play.py --model ./dqn_model.zip --episodes 3

# Evaluate without rendering (faster)
python play.py --evaluate --episodes 20

# Find and play with best model from experiments
python play.py --best
```

## Files

- `train.py` - Training script with hyperparameter tuning
- `play.py` - Playing script with greedy policy evaluation
- `HYPERPARAMETER_DOCUMENTATION.md` - Detailed hyperparameter analysis
- `requirements.txt` - Project dependencies

## Model Architecture

- **Policy**: CNNPolicy (Convolutional Neural Network)
- **Algorithm**: Deep Q-Network (DQN)
- **Frame Stacking**: 4 frames for temporal information
- **Preprocessing**: Atari wrappers (grayscale, resize, frame skip)

## Key Features

- Automatic hyperparameter experiment logging
- TensorBoard integration for training visualization
- Best model checkpoint saving
- Greedy Q-policy for evaluation (deterministic action selection)
