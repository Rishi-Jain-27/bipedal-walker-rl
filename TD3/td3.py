
import torch
import torch.nn as nn
import numpy as np

# Critic
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

    def forward(self, x, a):
        x = torch.cat([x, a], 1) # state = x. this glues state and action into a single vector
        return self.q1(x), self.q2(x) # output value represents Q-value for that specific state-action pair
    
    def Q1(self, x, a):
        x = torch.cat([x, a], 1)
        return self.q1(x)

# Actor
class Actor(nn.Module):
    def __init__(self, env, hidden_dim_1=400, hidden_dim_2=300):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_dim_1), # input dim = observation dim
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, np.prod(env.single_action_space.shape)), # output is the action to take
            nn.Tanh() # bind continuous actions with a specific, symmetrical range
        )

        # Action rescaling to map network's output to the environment
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        ) # we use env.single_action_space because that gives action_scale and action_bias shape (action_dim,) which broadcasts against any batch size
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
    
    def forward(self, x):
        # gets the output and shifts and scales to the range allowed by the env's action_space
        return self.layers(x) * self.action_scale + self.action_bias



