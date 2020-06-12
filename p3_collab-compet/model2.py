import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=0, fc1_units=512, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

#         self.fc1.weight.data.normal_(std=norm_init(self.fc1))
#         self.fc2.weight.data.normal_(std=norm_init(self.fc2))
#         self.fc3.weight.data.normal_(std=0.003)


    def forward(self, state,  should_sample=False):
        """Build an actor (policy) network that maps states -> actions."""
#         if state.dim() == 1:
#             state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        restuls= torch.tanh(self.fc3(x))
        return restuls

def norm_init(layer):
    fan_in = layer.weight.data.size()[0]
    return 1./np.sqrt(fan_in)

class ActorNoise(Actor):
    """Parameter noise for Actor policy model."""

    def __init__(self, state_size, action_size, seed):
        super(ActorNoise, self).__init__(state_size, action_size, seed)

    def reset_parameters(self):
        self.fc1.weight.data.normal_(std=norm_init(self.fc1))
        self.fc2.weight.data.normal_(std=norm_init(self.fc2))
        self.fc3.weight.data.normal_(std=0.003)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, num_agents = 2, fcs1_units=512, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        print('model_2 Critic')

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear((state_size+action_size)*num_agents, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.dropout = nn.Dropout(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
#         self.fcs1.weight.data.normal_(std=norm_init(self.fcs1))
#         self.fc2.weight.data.normal_(std=norm_init(self.fc2))
#         self.fc3.weight.data.normal_(std=0.003)


    def forward(self, state):
#         print('\n model state {} action {}'.format(state.shape, action.shape))
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        xs = F.relu(self.dropout(self.fcs1(state)))
#         xs = nn.LeakyReLU(self.fcs1(state))
#         xs = F.relu(self.dropout(self.fcs1(state)))
#         xs = F.relu(self.fcs1(state))
        x = F.relu(self.fc2(xs))
#         x = nn.LeakyReLU(self.fc2(xs))
        return self.fc3(x)
