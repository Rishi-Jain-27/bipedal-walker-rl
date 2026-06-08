
# Import environment
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AutoresetMode

# Import ML libraries
import numpy as np
import torch
import torch.nn as nn

# Import stuff from other files
from sac import Actor, Critic
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
        self.learning_starts = hyperparameters['learning_starts']
        self.policy_frequency = hyperparameters['policy_frequency']
        self.death_penalty = hyperparameters['death_penalty']
        self.entropy_reg_coeff = hyperparameters['entropy_reg_coeff']
        self.autotune = hyperparameters['autotune']

        # Path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def train(self, render=False):
        envs = gym.vector.SyncVectorEnv([make_env(self.env_id, render, **self.env_make_params)  for _ in range(self.num_envs)], autoreset_mode=AutoresetMode.SAME_STEP)

        actor = Actor(envs).to(device)
        actor_optimizer = torch.optim.Adam(
            params=actor.parameters(),
            lr=self.learning_rate)

        critic = Critic(envs).to(device)
        target_critic = Critic(envs).to(device)
        target_critic.load_state_dict(critic.state_dict())
        critic_optimizer = torch.optim.Adam(
            params=critic.parameters(),
            lr=self.learning_rate
        )

        if self.autotune:
            # Target entropy = negative dim action space to encourage exploring enough to cover the action space
            target_entropy = -torch.prod(torch.tensor(envs.single_action_space.shape, device=device)).item()

            # Optimize the log of alpha to make alpha positive after exponentiation
            # init to 0 (e^0 = 1)
            log_alpha = torch.zeros(1, requires_grad=True, device=device)

            alpha = log_alpha.exp().item()

            # Optimizer based on how far entropy is from target entropy
            alpha_optimizer = torch.optim.Adam(
                params=[log_alpha],
                lr=self.learning_rate
            )
        else:
            # Make alpha a tensor to avoid tensor specific crashes in optimize
            alpha = torch.tensor(self.entropy_reg_coeff, device=device)


        loss_fn = nn.MSELoss()

        rb = ReplayBuffer(self.buffer_size, envs, device)

        episodic_returns = []
        best_return = float('-inf')

        start_time = datetime.now()
        last_graph_update_time = start_time
        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as log_file:
            log_file.write(log_message + '\n')
        

        obs, _ = envs.reset()

        # We loop for however many iterations is needed to cover the total timesteps count
        # remembering that vectorized environments means we divide total timesteps by num envs
        # each loop iter steps all envs at once

        # note: floor div truncates, not 100% exact if total timesteps not perfectly divisible
        for iteration in range(self.total_timesteps // self.num_envs):
            # global step is the running count of all env steps taken
            global_step = iteration * self.num_envs

            if global_step < self.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(self.num_envs)])
            else:
                with torch.no_grad():
                    """
                    Action is a value sampled from the normal distribution
                    created when passing in state x to the Actor
                    so we can get the Action using actor.get_action

                    Tho we call it actions bc of batching and vectorized envs

                    and we convert the action to numpy
                    """
                    actions, _, _ = actor.get_action(torch.tensor(obs, device=device))
                    actions = actions.detach().cpu().numpy()
            
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # All the following is just logging, metrics, and data collection stuff

            # Logging and saving model logic — ripped straight from td3:
            if "final_info" in infos: # final info only where at least one subenv finished an episode
                ep = infos["final_info"]["episode"] # ep is stats dict that RecordEpisodeStatistics produced, stored column-wise
                finished = infos["final_info"]["_episode"] # boolean mask — true if env i completed an episode this step
                for i in range(self.num_envs): # walk each env
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
            
            # Fix auto-reset not giving the final state
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_obs"][idx]
            
            
            # Reward clipping + update replay buffer ripped from td3
            # this is bc it's -100 for losing the game
            if self.death_penalty is not None:
                clipped_rewards = rewards.copy()
                mask = terminations & (clipped_rewards < -90)
                clipped_rewards[mask] = self.death_penalty
            
                # Add data with clippd reward to buffer
                rb.add(obs, actions, clipped_rewards, real_next_obs, terminations)
            else:
                # Else, add data with normal reward to buffer
                rb.add(obs, actions, rewards, real_next_obs, terminations)
            
            obs = next_obs # so we move through the states and not infinity loop


            # Now that all that data collection metrics stuff is done, we can train
            if global_step > self.learning_starts:
                for i in range(self.num_envs):
                    states, actions, rewards, next_states, dones = rb.sample(self.batch_size)
                    if self.autotune:
                        alpha = log_alpha.exp()
                        self.optimize(
                            states=states,
                            actions=actions,
                            rewards=rewards,
                            next_states=next_states,
                            dones=dones,
                            actor=actor,
                            actor_optim=actor_optimizer,
                            target_critic=target_critic,
                            critic=critic,
                            critic_optim=critic_optimizer,
                            alpha=alpha,
                            log_alpha=log_alpha,
                            target_entropy=target_entropy,
                            alpha_optim=alpha_optimizer,
                            loss_fn=loss_fn,
                            update_step=global_step + i)
                    else:
                        self.optimize(
                            states=states,
                            actions=actions,
                            rewards=rewards,
                            next_states=next_states,
                            dones=dones,
                            actor=actor,
                            actor_optim=actor_optimizer,
                            target_critic=target_critic,
                            critic=critic,
                            critic_optim=critic_optimizer,
                            alpha=alpha,
                            log_alpha=None, # harmless bc optimize guards with self.autotune
                            target_entropy=None, # harmless bc optimize guards with self.autotune
                            alpha_optim=None, # harmless bc optimize guards with self.autotune
                            loss_fn=loss_fn,
                            update_step=global_step + i)



    # Maybe this isn't a very good function to write
    # because dear god this is a massive amount of params
    def optimize(self,
                 states,
                 actions,
                 rewards,
                 next_states,
                 dones,
                 actor,
                 actor_optim,
                 target_critic,
                 critic,
                 critic_optim,
                 alpha,
                 log_alpha,
                 target_entropy,
                 alpha_optim,
                 loss_fn,
                 update_step):
        with torch.no_grad():
            # next_state_log_prob is prob of action being chosen by state
            next_state_actions, next_state_log_prob, _ = actor.get_action(next_states)
            q1_next_targ, q2_next_targ = target_critic(next_states, next_state_actions)

            # subtract by temperature * log prob incentivizes visitng states where there is more entropy/higher uncertainty
            critic_next_target = torch.min(q1_next_targ, q2_next_targ) - alpha.detach() * next_state_log_prob
            # agent tries to maximize reward. high alpha decreases reward, so agent thus explores more to increase reward

            next_q_val = rewards.flatten() + (1 - dones.flatten()) * self.gamma * critic_next_target.view(-1)
        
        q1, q2 = critic(states, actions)
        # .view gives (batch_size, ) here
        q1 = q1.view(-1)
        q2 = q2.view(-1)
        # add losses so the optimizer works for both critic nets
        critic_loss = loss_fn(q1, next_q_val) + loss_fn(q2, next_q_val)

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        if update_step % self.policy_frequency == 0:
            actions, action_log_prob, _ = actor.get_action(states)
            q1, q2 = critic(states, actions)
            min_q = torch.min(q1, q2)
            actor_loss = ((alpha.detach() * action_log_prob) - min_q).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            if self.autotune:
                with torch.no_grad():
                    _, log_prob, _ = actor.get_action(states)
                # Calc loss: alpha - (log prob of action from the policy + target entropy)
                # log prob + target entropy is close to zero, but if log prob too negative
                # then too deterministic bc the chances of getting that action randomly are too high — thats my understanding
                alpha_loss = (-log_alpha.exp() * (log_prob + target_entropy)).mean()
                # use .mean() bc log_prob and thus alpha_loss is shape (batch_size, 1)
                # so .mean() it to get scalar (1)
                # bc gradients are only defined relative to a scalar

                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()

                # This ends up updating alpha
                # bc log alpha is a param and alpha gets recalculated before input into this function
            
            # Update the target networks — soft update
            # idk if this is correct or not but let's just pray it is
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def run(self):
        envs = gym.vector.SyncVectorEnv([make_env(self.env_id, render=True, **self.env_make_params)])

        actor = Actor(envs).to(device)
        actor.load_state_dict(torch.load(self.MODEL_FILE, weights_only=True))
        actor.eval()
        state, _ = envs.reset() # vector envs auto-reset, so this can be outside the loop

        with torch.inference_mode():
            for i in itertools.count():
                done = False
                while not done:
                    # third return is mean, the deterministic outcome of SAC
                    # make that the action for run so we're not doing stochastic stuff during inference
                    # which would be pretty bad ngl
                    _, _, action = actor.get_action(torch.tensor(state, device=device))
                    state, _, terminated, truncated, _ = envs.step(action.cpu().numpy()) # no detach bc inference mode
                    done = terminated[0] or truncated[0] # vector env — term and trunc shape (1,)

    def save_graph(self, episodic_returns):
        if not episodic_returns: return
        steps, returns = zip(*episodic_returns)
        plt.plot(steps, returns)
        plt.xlabel('Global step')
        plt.ylabel('Episodic returns')
        plt.savefig(self.GRAPH_FILE)
        plt.close()

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

    sac = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        sac.train() # python sac_agent.py --train sacbipedalwalker_hard
    else:
        sac.run()