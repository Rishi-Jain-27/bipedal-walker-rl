
import torch
import torch.nn as nn
import numpy as np

class Critic(nn.Module):
    def __init__(self, env, hidden_dim_1=400, hidden_dim_2=300):
        super().__init__()

        # We make two different Q-networks in one module

        # First Q network
        # Make input dim = action dim + observation space dim because need to see
        # both state and action to determine if that specific combination is good or bad
        self.q1 = nn.Sequential(
            nn.Linear(
                np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
                hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1) # bring result down to scalar
        )

        # Second Q network
        self.q2 = nn.Sequential(
            nn.Linear(
                np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
                hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1) # bring result down to scalar
        )

    def forward(self, x, a): # state = x, action = a
        x = torch.cat([x, a], 1)
        return self.q1(x), self.q2(x) # output valus represents Q-values for that specific state-action pair
    
    def Q1(self, x, a):
        x = torch.cat([x, a], 1)
        return self.q1(x)

# need these for preventing logstd from going crazy
LOG_STD_MAX = 2
LOG_STD_MIN = -5
class Actor(nn.Module):
    def __init__(self, env, hidden_dim_1=400, hidden_dim_2=300):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim_2, np.prod(env.single_action_space.shape))
        self.log_std = nn.Linear(hidden_dim_2, np.prod(env.single_action_space.shape))

        # Map network output to env

        # env.single_action_space because gives action_scale and action_bias shape (action_dim,) which broadcasts against any batch size
        self.action_scale = torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        self.action_bias = torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        
    
    def forward(self, x):
        x = self.layers(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.tanh(log_std) # tanh to map to a fixed range of values

        # rescale log_std — applies the linear interpolation formula
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std
    
    def get_action(self, x):
        mean, log_std = self(x) # does forward
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)

        # for reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        # Action scaled to match the env's action space
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)

        # Bc of tanh, probability density shifts. this corrects for tanh
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2) + 1e-6))
        log_prob = log_prob.sum(1, keepdim=True) # 0 is batch dim, 1 is where log probs are

        # Calculates deterministic action (mean of the policy) for testing
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean
    
    def to(self, device):
        # action scale and bias aren't registered buffers or params
        # so must override to so they also go to device
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)



