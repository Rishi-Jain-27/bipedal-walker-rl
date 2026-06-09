# bipedal-walker-rl
This is a PyTorch implementation of the Deep Deterministic Policy Gradient (DDPG), Twin Delayed Deep Deterministic (TD3), and Soft Actor-Critic (SAC) algorithms. The algorithms learn to play **Bipedal Walker** from the [Bipedal Walker Gymnasium environment](https://gymnasium.farama.org/environments/box2d/bipedal_walker/).
This implementation includes a configurable training pipeline, models, logs of training data, and graphs of the returns during training for each model.

## Demos
| DDPG | TD3 | SAC
|-------------|----------|----------|
| ![DDPG](demos/ddpg_demo.mov) | ![TD3](demos/td3_demo.mov) | ![SAC](demos/sac_demo.mov) |

Trained agents and reward curves are saved under `runs/` in the respective `DDPG`, `TD3`, and `SAC` directories.

## Features
- **DDPG**, **TD3**, and **SAC** implementations from scratch, including a replay buffer highly optimized to conserve memory.
- **Vectorized environments** for quick and efficient training.
- **Hyperparameters** configurable per each experiment.
- **Automatic logging**, **best-model checkpointing**, and **live reward/epsilon plots**.

## Project Layout
```
bipedal-walker-rl/
├── DDPG/
│   ├── ddpg.py # ActorCritic networks + action rescaling
│   ├── ddpg_agent.py # DDPG Agent training loop, run method, CLI entry point
│   ├── hyperparameters.yml # Hyperparameters loaded in ddpg_agent.py
│   ├── runs/ # Logs, reward plots, and best model checkpoint
├── TD3
│   ├── td3.py # ActorCritic networks + action rescaling
│   ├── td3_agent.py # TD3 Agent, training loop, run method, CLI entry point
│   ├── hyperparameters.yml # Hyperparameters loaded in td3_agent.py
│   ├── runs/ # Logs, reward plots, and best model checkpoint
├── SAC
│   ├── sac.py # ActorCritic network, action rescaling, get_action method, .to override
│   ├── sac_agent.py # SAC Agent, training loop, run method, CLI entry point
│   ├── hyperparameters.yml # Hyperparameters loaded in sac_agent.py
│   ├── runs/ # Logs, reward plots, and best model checkpoint
├── demos # Recorded gameplay of trained agents
├── requirements.txt
└── README.md
```

## Setup
Note: requires **Python 3.11** 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train an agent using a named hyperparameter set from `hyperparameters.yml`:

```bash
cd ALGORITHM_DIR
python ALGORITHM_agent.py HYPERPARAM_SET_NAME --train
```

Watch a trained agent play (loads `runs/<set>.pt`, renders to screen):

```bash
cd ALGORITHM_DIR
python agent.py HYPERPARAM_SET_NAME
```

While training, the script writes:
- `runs/<set>.log`. The timestamped log of new best rewards.
- `runs/<set>.pt`. The best model weights so far.
- `runs/<set>.png`. Curves of mean-reward and epsilon decay.

## Hyperparameter sets
Defined in `hyperparameters.yml` for each algorithm's directory. Add a new key to define your own experiment.
