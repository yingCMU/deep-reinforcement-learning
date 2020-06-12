# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network
import importlib
# from tensorboardX import SummaryWriter
from torch.optim import Adam

import ddpg
importlib.reload(ddpg)
import numpy as np
import random

# import model
# importlib.reload(model)
import model2 as model
importlib.reload(model)
# import model2Noisy as model
# importlib.reload(model)

import torch
import buffer
importlib.reload(buffer)
import torch.nn.functional as F

import utilities
importlib.reload(utilities)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256#128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2#1e-3              # for soft update of target parameters
LR_ACTOR =  5e-4         # learning rate of the actor
MIN_LR_ACTOR = 1e-4
LR_CRITIC = 1e-3 #3e-4        # learning rate of the critic
WEIGHT_DECAY = 0#0.0001   # L2 weight decay
# log_path = os.getcwd()+"/log"
# logger = SummaryWriter(log_dir=log_path)


class MADDPG:
    def __init__(self, state_size, action_size, random_seed, warm_up=BATCH_SIZE,  lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, num_agents=2,):
        super(MADDPG, self).__init__()
        # critic input = obs_full + actions = 14+2+2+2=20
        self.shared_critic_local =  model.Critic(state_size, action_size, random_seed).to(device)
        self.shared_critic_target = model.Critic(state_size, action_size, random_seed).to(device)
        self.shared_critic_optimizer = Adam(self.shared_critic_local.parameters(), lr=lr_critic, weight_decay=0)
        self.maddpg_agent = [ddpg.DDPGAgent(state_size, action_size, 12, warm_up, lr_actor,lr_critic, self.shared_critic_local, self.shared_critic_target, self.shared_critic_optimizer),
                             ddpg.DDPGAgent(state_size, action_size, 0, warm_up, lr_actor,lr_critic,self.shared_critic_local, self.shared_critic_target, self.shared_critic_optimizer)]

        self.discount_factor = GAMMA
        self.tau = TAU
        self.iter = 0
        self.num_agents = num_agents
        self.memory=buffer.ReplayBuffer(action_size, BUFFER_SIZE, random_seed)
#         self.memory_2=buffer.ReplayBuffer(action_size, BUFFER_SIZE, random_seed)


    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()

    def reduce_LR(self, reduce_factor=0.8, min_lr_actor=MIN_LR_ACTOR):
        for agent in self.maddpg_agent:
            agent.reduce_LR(reduce_factor,MIN_LR_ACTOR)

    def reduce_tau(self, reduce_factor=0.9, min_tau=1e-3):
        if self.tau > min_tau:
            self.tau = min(self.tau*reduce_factor, min_tau)
            print('!!! update self.tau', self.tau)

    def add_to_memory(self, states, actions, rewards, next_states, dones, i_episode):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        obs_full = np.reshape(states, newshape=(1, -1))
        next_obs_full = np.reshape(next_states, newshape=(1, -1))
        self.memory.push(states, obs_full, actions, rewards, next_states,next_obs_full, dones)

    def learn_and_update(self, episode_i, episode_t,  update_iteration, start_update_episode=0,ts_per_update=1 , updates_per_ts=1, episodes_per_update=1):
        if episode_t % ts_per_update != 0:
            return update_iteration

        if len(self.memory) > BATCH_SIZE and episode_i % episodes_per_update ==0 and episode_i > start_update_episode:
            samples = self.memory.sample(BATCH_SIZE)
#             if random.random() < 0.01:
#                 print('samples', samples)
            if updates_per_ts < 1:
                if random.random() < updates_per_ts:
                    update_iteration += 1
                    for agent_i in range(self.num_agents):
#                         samples = self.memory.sample(BATCH_SIZE)
                        self.update(samples, agent_i)
                        self.update_targets(True)

            else:
                for updat4e_i in range(updates_per_ts):
                    update_iteration += 1
                    for agent_i in range(self.num_agents):
#                         samples = self.memory.sample(BATCH_SIZE)
                        self.update(samples, agent_i)
                        self.update_targets(True)
#             self.update_targets(update_critic=True)

        return update_iteration


    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """
        get actions from all agents in the MADDPG object
        obs_all_agents : 2 * 24 nparray => tensor
        """
        actions = [agent.act(torch.tensor(obs, dtype=torch.float), noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples
#         obs_full = torch.stack(obs_full)
#         next_obs_full = torch.stack(next_obs_full)
        agent = self.maddpg_agent[agent_number]
        agent.actor_target.eval()
        agent.critic_target.eval()
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs) # a list of size 2, each torch inside Size([128, 2])
        target_actions = torch.cat(target_actions, dim=1)
        target_critic_input = torch.cat((next_obs_full.view(BATCH_SIZE, -1),target_actions.view(BATCH_SIZE, -1)), dim=1).to(device)

        with torch.no_grad():
            q_next = agent.critic_target(target_critic_input)

        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action=action.t()
        critic_input = torch.cat((obs_full.view(BATCH_SIZE, -1), action.view(BATCH_SIZE, -1)), dim=1).to(device)
        agent.actor_target.train()
        agent.critic_target.train()
        q = agent.critic_local(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())

#         critic_loss = huber_loss(q, y)
#         critic_loss = F.mse_loss(q, y)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1.0)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor_local(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor_local(ob).detach()
                   for i, ob in enumerate(obs) ]

        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full, q_input), dim=1)

        # get the policy gradient
        actor_loss = -agent.critic_local(q_input2).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
#         torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(), 0.5)
        agent.actor_optimizer.step()

#         al = actor_loss.cpu().detach().item()
#         cl = critic_loss.cpu().detach().item()
#         if random.random() < 0.008:
#             print('agent{} :  actor_loss {}'.format(agent_number, al))
#         logger.add_scalars('agent%i/losses' % agent_number,
#                            {'critic loss': cl,
#                             'actor_loss': al},
#                            self.iter)

    def update_targets(self, update_critic):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            utilities.soft_update(ddpg_agent.actor_target, ddpg_agent.actor_local, self.tau)
            if update_critic:
                utilities.soft_update(ddpg_agent.critic_target, ddpg_agent.critic_local, self.tau)
