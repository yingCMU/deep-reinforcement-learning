import importlib
import torch

import random
import utilities
importlib.reload(utilities)
from collections import namedtuple, deque
import numpy as np


# class ReplayBuffer:
#     def __init__(self,size):
#         self.size = size
#         self.deque = deque(maxlen=self.size)

#     def push(self,transition):
#         """push into the buffer"""
#         print('transition[0]',transition[0].shape, transition[0])
#         input_to_buffer = utilities.transpose_list(transition)
#         print('input_to_buffe[3]r', input_to_buffer)

#         for item in input_to_buffer:
#             self.deque.append(item)
#         return transition

#     def sample(self, batchsize):
#         """sample from the buffer"""
#         samples = random.sample(self.deque, batchsize)

#         # transpose list of list
#         return utilities.transpose_list(samples)

#     def __len__(self):
#         return len(self.deque)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        print('ReplayBuffer-device:',device)
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["states","obs_full", "actions", "reward", "next_states","next_obs_full", "dones"])
        self.seed = random.seed(seed)

    def push(self, states, obs_full, actions, reward, next_states,next_obs_full, dones):
        """Add a new experience to memory."""
        e = self.experience(states,obs_full, actions, reward, next_states,next_obs_full, dones)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.array([e.states for e in experiences if e is not None])).float().to(device)
        obs_full = torch.from_numpy(np.array([e.obs_full for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_states for e in experiences if e is not None])).float().to(device)
        next_obs_full = torch.from_numpy(np.array([e.next_obs_full for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states.t(), obs_full.view(batch_size, -1), actions.t(), rewards.t(), next_states.t(),next_obs_full.view(batch_size, -1), dones.t())

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
