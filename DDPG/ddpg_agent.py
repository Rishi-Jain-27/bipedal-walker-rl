# Note: weird stuff with self.envs and all happening but I think it works

# Import environment
import gymnasium as gym
from gymnasium import spaces

# Import ML libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import stuff from other files
from ddpg import Actor, Critic
from replay_buffer import ReplayBuffer

# Import yaml for hyperparams
import yaml

# Itertools for indefinite looping
import itertools

# os for directories
import os

# MPL for plotting
import matplotlib.pyplot as plt

# datetime for datetime
from datetime import datetime, timedelta

# argparse for CLI training
import argparse



# for printing date and time
DATE_FORMAT = "%y-%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, hyperparameter_set):
        # Get hyperparameters
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set

        # Environment params
        self.env_id = hyperparameters['env_id']
        self.env_make_params = hyperparameters.get('env_make_params', {})

        # Training relevant params
        self.num_envs = hyperparameters['num_envs']
        self.total_timesteps = hyperparameters['total_timesteps']
        self.learning_rate = hyperparameters['learning_rate']
        self.buffer_size = hyperparameters['buffer_size']
        self.gamma = hyperparameters['gamma']
        self.tau = hyperparameters['tau']
        self.batch_size = hyperparameters['batch_size']
        self.exploration_noise = hyperparameters['exploration_noise']
        self.learning_starts = hyperparameters['learning_starts']
        self.policy_frequency = hyperparameters['policy_frequency']

        # Path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
    
    def train(self, render=False):
        # Build the environments (vectorized envs)
        self.envs = gym.vector.SyncVectorEnv([make_env(self.env_id, render, **self.env_make_params)  for _ in range(self.num_envs)])
        
        # Create the Actor-Critic networks, target networks, optimizers, and loss function
        actor = Actor(self.envs).to(device)
        target_actor = Actor(self.envs).to(device)
        target_actor.load_state_dict(actor.state_dict())

        critic = Critic(self.envs).to(device)
        target_critic = Critic(self.envs).to(device)
        target_critic.load_state_dict(critic.state_dict())

        self.actor_optimizer = optim.Adam(params=actor.parameters(),
                                      lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(params=critic.parameters(),
                                      lr=self.learning_rate)
        
        self.loss_fn = nn.MSELoss()
        
        # Create ReplayBuffer
        rb = ReplayBuffer(self.buffer_size, self.envs, device)

        # Create episodic returns and best return tracking
        episodic_returns = []
        best_return = float('-inf')

        # Begin logging
        start_time = datetime.now()
        last_graph_update_time = start_time
        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')
        
        obs, _ = self.envs.reset()

        # This makes sense: num iterations is number of timesteps
        # needed to complete total timesteps timesteps across all environments running
        # I can imagine this! So cool.
        for iteration in range(self.total_timesteps // self.num_envs):
            global_step = iteration * self.num_envs
            if global_step < self.learning_starts:
                # If learning hasn't started yet, just get some random actions
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                with torch.no_grad():
                    # The actor is a deterministic function, so we'll get the best action
                    actions = actor(torch.tensor(obs).to(device))

                    # Then, we inject Gaussian noise into it
                    actions += torch.normal(0, actor.action_scale * self.exploration_noise)

                    # After adding noise, clip to the bounds of the environment
                    # Add an assertion to prevent PyLance from freaking out
                    assert isinstance(self.envs.single_action_space, spaces.Box), "This agent requires a Box action space"
                    actions = actions.cpu().numpy().clip(self.envs.single_action_space.low, self.envs.single_action_space.high)
            
            # Execute the game
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

            # Logging & saving model logic
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        r = float(info["episode"]["r"]) # float bc it's usually a length-1 array
                        episodic_returns.append((global_step, r))
                        self._log(f"global_step={global_step}, episodic_return={r:.1f}")
            
                    # Save the model if it's good
                    recent = [r for _, r in episodic_returns[-100:]]
                    mean_return = sum(recent)/len(recent)
                    if mean_return > best_return:
                        best_return = mean_return
                        torch.save(actor.state_dict(), self.MODEL_FILE)
            
            # Update graph every 30 seconds
            if datetime.now() - last_graph_update_time > timedelta(seconds=30):
                self.save_graph(episodic_returns)
                last_graph_update_time = datetime.now()
            
            # Save data to replay buffer
            # Vector envs auto-reset any sub-env that finishes
            # When an env terminates, the next_obs it hands is already the first obs of the new episode, not the real final state of the episode that just ended
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, actions, rewards, real_next_obs, terminations)

            # Crucial. Else, the agent feeds the old observation back into the actor network, getting stuck in a loop of the same state
            obs = next_obs

            # Train!
            if global_step > self.learning_starts:
                states, actions, rewards, next_states, dones = rb.sample(self.batch_size)
                self.optimize(states, actions, rewards, next_states, dones, target_actor, actor, target_critic, critic, iteration)

    def optimize(self, states, actions, rewards, next_states, dones, target_actor, actor, target_critic, critic, iteration):
        # Get predicted Q value from target
        with torch.no_grad():
            next_state_actions = target_actor(next_states)
            critic_next_target = target_critic(next_states, next_state_actions)
            next_q_values = rewards.flatten() + (1 - dones.flatten()) * self.gamma * critic_next_target.view(-1)
        
        # Now get it from critic
        q_values = critic(states, actions).view(-1) # .view resolves shaping issues here

        # Find loss
        critic_loss = self.loss_fn(q_values, next_q_values)

        # optimize
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Check if we also optimize our actor network and update it
        # Loop by iterations becaue we want to count gradient updates, and there is exactly one per loop iteration
        if iteration % self.policy_frequency == 0:
            # Calc loss
            actor_loss = -critic(states, actor(states)).mean() # have the critic evaluate the actions the actor is taking
            
            # Optimize
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the target networks — soft update
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)        

    def run(self):
        # Build the env (single)
        self.envs = gym.vector.SyncVectorEnv([make_env(self.env_id, render=True, **self.env_make_params)])

        # Build an Actor (no Critic bc not training)
        actor = Actor(self.envs).to(device)
        actor.load_state_dict(torch.load(self.MODEL_FILE, weights_only=True))
        actor.eval()

        state, _ = self.envs.reset() # Note that vector envs auto-reset

        # Loop infinitely. No exploration noise, buffer, and gradients.
        with torch.no_grad():
            for i in itertools.count():
                done = False
                while not done:
                    action = actor(torch.tensor(state).to(device))
                    state, _, terminated, truncated, _ = self.envs.step(action.cpu().numpy())
                    done = terminated[0] or truncated[0] # because this is a vector env — terminated and truncated are shape (1,)


    def save_graph(self, episodic_returns):
        if not episodic_returns: return

        # Unzip the pairs
        steps, returns = zip(*episodic_returns)

        plt.plot(steps, returns)
        plt.xlabel('Global step')
        plt.ylabel('Episodic returns')
        plt.savefig(self.GRAPH_FILE)
        plt.close() # so figures don't pile up 
    
    # Log helper function — for appending.
    def _log(self, msg):
        print(msg)
        with open(self.LOG_FILE, 'a') as f:
            f.write(msg + '\n')

# Make environment helper function for vectorized envs
def make_env(env_id, render=False, **env_make_params):
    def thunk():
        env = gym.make(env_id, render_mode="human" if render else None, **env_make_params)
        return env
    return thunk

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or test?")
    parser.add_argument('hyperparameters', help='Enter the name of the set of hyperparameters to test/train')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    ddpg = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        ddpg.train()
    else:
        ddpg.run()
