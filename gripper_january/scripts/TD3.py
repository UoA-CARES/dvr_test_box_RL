
import random
import numpy as np

import copy
import torch
import torch.optim as optim

from Networks_Architectures import Actor_Normal as Actor
class TD3:

    def __init__(self, device, obs_dim, act_dim):
        self.device = device

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.max_action_value = 1
        self.min_action_value = -1

        # main networks
        self.actor = Actor(self.obs_dim, self.act_dim, self.max_action_value).to(self.device)

        # target networks
        self.actor_target  = copy.deepcopy(self.actor)

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.to(self.device)
            action = self.actor(state_tensor)
            action = action.cpu().data.numpy().flatten()
        return action

    def action_sample(self):
        action = []
        for i in range(0, self.act_dim):
            a = np.clip(random.uniform(-1, 1), self.min_action_value, self.max_action_value)
            action.append(a)
        return action

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)

        # Convert into tensor
        states  = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones       = torch.LongTensor(dones).to(self.device)

        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones   = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)






