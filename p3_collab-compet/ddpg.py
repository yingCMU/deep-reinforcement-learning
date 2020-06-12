# individual network settings for each actor + critic pair
# see networkforall for details

# from networkforall import Network
import importlib

from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np
# import model
# importlib.reload(model)
import model2 as model
importlib.reload(model)
# import model2Noisy as model
# importlib.reload(model)

import random
# add OU noise for exploration
import OUNoise
importlib.reload(OUNoise)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_size, action_size, random_seed, warm_up,  lr_actor, lr_critic, shared_critic_local, shared_critic_target,  shared_critic_optimizer):
        super(DDPGAgent, self).__init__()
        self.lr_actor = lr_actor
        print('DDPGAgent: lr_actor={} ; lr_critic={}'.format(lr_actor, lr_critic))
        self.actor_local = model.Actor(state_size, action_size, random_seed).to(device)
#         self.actor_local = model.Actor(state_size, action_size, random_seed, noise='linear').to(device)
#         self.actor_target = model.Actor(state_size, action_size, random_seed, noise='linear').to(device)
        self.actor_target = model.Actor(state_size, action_size, random_seed).to(device)
        self.critic_local = shared_critic_local
        self.critic_target = shared_critic_target
        self.critic_optimizer = shared_critic_optimizer

#         self.critic_local = model.Critic(state_size, action_size, random_seed).to(device)
#         self.critic_target = model.Critic(state_size, action_size, random_seed).to(device)
#         self.critic_optimizer = Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=0)#1.e-5)

        self.noise = OUNoise.OUNoise(action_dimension=action_size, sigma=0.5, scale=1)
        # initialize targets same as original networks
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)

        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=lr_actor)

    def reset(self):
        self.noise.reset()

    def reduce_LR(self, reduce_factor, min_lr_actor):
        if self.lr_actor>min_lr_actor:
            self.lr_actor *= reduce_factor
            self.lr_actor = max(self.lr_actor, min_lr_actor)
            print('!!! reduce_LR to ', self.lr_actor)
            self.actor_optimizer = Adam(self.actor_local.parameters(), lr=self.lr_actor)


    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        noise_tensor = self.noise.noise()
        should_sample = False
        if  random.random()<0.0005: #i_episode>=100 and
            should_sample = True
        self.actor_local.eval()
        ori_action= self.actor_local(obs, should_sample)

        action =ori_action + noise*noise_tensor
        clipped_action = torch.clamp(action, min=-1.0, max=1.0)
#         if  should_sample:
#             print('$$$$noise: {:.2f}, noise*noise_tensor={}; clipped_action={}'.format(noise, noise*noise_tensor.detach().cpu().numpy(), clipped_action.detach().cpu().numpy()))
        self.actor_local.train()

        return clipped_action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor_target(obs) + noise*self.noise.noise()
        return action
