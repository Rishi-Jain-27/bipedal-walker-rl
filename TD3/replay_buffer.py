import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, envs, device):
        # Bc hyperparams makes it a float i think
        self.buffer_size = int(buffer_size)
        self.num_envs = envs.num_envs

        # Read dimensions off env
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)

        self.states = np.zeros((self.buffer_size, self.num_envs, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, self.num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)

        self.device = device

        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        # Update np arrays
        self.states[self.pos] = state
        self.next_states[self.pos] = next_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        # advance the pointer
        self.pos = (self.pos + 1) % self.buffer_size

        # Grow the fill counter
        self.size = min(self.size + 1, self.buffer_size)
        # Climbs until buffer is full, then pins at capacity
        # stops sample from drawing an unwritten slot.

    def sample(self, batch_size):
        # Draw random indices
        rand_row_idxs = np.random.randint(0, self.size, size=batch_size)
        rand_env_idxs = np.random.randint(0, self.num_envs, size=batch_size)

        # Index all five arrays & convert to tensors
        states = torch.as_tensor(self.states[rand_row_idxs, rand_env_idxs]).to(self.device) # gives (batch_size, obs_dim)
        next_states = torch.as_tensor(self.next_states[rand_row_idxs, rand_env_idxs]).to(self.device) # gives (batch_size, obs_dim)
        actions = torch.as_tensor(self.actions[rand_row_idxs, rand_env_idxs]).to(self.device)  # gives (batch_size, action_dim)
        rewards = torch.as_tensor(self.rewards[rand_row_idxs, rand_env_idxs]).to(self.device).reshape(batch_size, 1) # reshape so broadcasting against critic's output still works
        dones = torch.as_tensor(self.dones[rand_row_idxs, rand_env_idxs]).to(self.device).reshape(batch_size, 1) # reshape so broadcasting against critic's output still works

        # Return
        return states, actions, rewards, next_states, dones
