# Note: weird stuff with self.envs and all happening but I think it works

# Import environment
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AutoresetMode

# Import ML libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import stuff from other files
from td3 import Actor, Critic
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
        self.total_timesteps = int(hyperparameters['total_timesteps'])
        self.learning_rate = float(hyperparameters['learning_rate'])
        self.buffer_size = float(hyperparameters['buffer_size']) # ReplayBuffer int-casts this later
        self.gamma = hyperparameters['gamma']
        self.tau = hyperparameters['tau']
        self.batch_size = hyperparameters['batch_size']
        self.exploration_noise = hyperparameters['exploration_noise']
        self.learning_starts = float(hyperparameters['learning_starts'])
        self.policy_frequency = hyperparameters['policy_frequency']
        self.policy_noise = hyperparameters['policy_noise']
        self.noise_clip = hyperparameters['noise_clip']
        self.death_penalty = hyperparameters.get('death_penalty') # None is the default's default, it can be omitted here (would've been 'death_penalty', None then)

        # Path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
    
    def train(self, render=False):
        # Build the environments (vectorized envs)
        self.envs = gym.vector.SyncVectorEnv([make_env(self.env_id, render, **self.env_make_params)  for _ in range(self.num_envs)], autoreset_mode=AutoresetMode.SAME_STEP)
        
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
            if "final_info" in infos: # final info only where at least one subenv finished an episode
                ep = infos["final_info"]["episode"] # ep is stats dict that RecordEpisodeStatistics produced, stored column-wise
                finished = infos["final_info"]["_episode"] # boolean mask — true if env i completed an episode this step
                for i in range(self.envs.num_envs): # walk each env
                    if finished[i]: # Act only on the envs that finished
                        # Logging
                        r = float(ep["r"][i]) # float bc it's usually a length-1 array
                        episodic_returns.append((global_step, r))
                        self._log(f"global_step={global_step}, episodic_return={r:.1f}")
            
                        # Best model check & save
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
                    real_next_obs[idx] = infos["final_obs"][idx]
            
            # Implement reward clipping bc it's -100 for falling
            # Do this here so it doesn't affect logs upstream
            if self.death_penalty is not None:
                clipped_rewards = rewards.copy()
                mask = terminations & (clipped_rewards < -90)
                clipped_rewards[mask] = self.death_penalty
            
                # Add data with clippd reward to buffer
                rb.add(obs, actions, clipped_rewards, real_next_obs, terminations)
            else:
                # Else, add data with normal reward to buffer
                rb.add(obs, actions, rewards, real_next_obs, terminations)

            # Crucial. Else, the agent feeds the old observation back into the actor network, getting stuck in a loop of the same state
            obs = next_obs

            # Train!
            if global_step > self.learning_starts:
                for i in range(self.envs.num_envs):
                    states, actions, rewards, next_states, dones = rb.sample(self.batch_size)
                    self.optimize(states, actions, rewards, next_states, dones, target_actor, actor, target_critic, critic, global_step + i)

    def optimize(self, states, actions, rewards, next_states, dones, target_actor, actor, target_critic, critic, update_step):
        # Get predicted Q value from target
        with torch.no_grad():
            # Compute the target action
            next_state_actions = target_actor(next_states)

            # Build smoothing noise
            smoothing_noise = (torch.randn_like(next_state_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) * actor.action_scale # clamp first in raw units, then multiply by action scale

            # Add noise to the action, then clip the action to env bounds
            next_state_actions = (next_state_actions + smoothing_noise).clamp(actor.action_bias - actor.action_scale, actor.action_bias + actor.action_scale)

            # Take the min of the two critic's Q values to correct against optimism
            q1_t, q2_t = target_critic(next_states, next_state_actions)
            critic_next_target = torch.min(q1_t, q2_t)

            # Find Bellman target
            next_q_values = rewards.flatten() + (1 - dones.flatten()) * self.gamma * critic_next_target.view(-1)
        
        # Now get the q1 and q2 on the current states/actions
        q1, q2 = critic(states, actions)

        # .view resolves shaping issues here — q1, q2 are shape (batch,), same for rewards/dones
        q1 = q1.view(-1)
        q2 = q2.view(-1)

        # Find loss
        critic_loss = self.loss_fn(q1, next_q_values) + self.loss_fn(q2, next_q_values)

        # optimize — trains both bc both Q-nets are in one module under one optimizer
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Check if we also optimize our actor network and update it
        if update_step % self.policy_frequency == 0:
            # Calc loss
            actor_loss = -critic.Q1(states, actor(states)).mean() # have the critic evaluate the actions the actor is taking, using Q1
            
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
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or test?")
    parser.add_argument('hyperparameters', help='Enter the name of the set of hyperparameters to test/train')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    td3 = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        td3.train() # python td3_agent.py --train td3bipedalwalker_hard
    else:
        td3.run() # python td3_agent.py td3bipedalwalker_hard
