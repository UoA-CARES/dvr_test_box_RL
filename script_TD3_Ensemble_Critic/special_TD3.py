
import os
import numpy as np
import torch.nn as nn

import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch.nn.functional as F

from Networks import Actor
from Networks import Special_Critic as Critic


class Special_Agent:
    def __init__(self, input_dim, action_num, device):

        self.input_dim   = input_dim
        self.action_num  = action_num
        self.device      = device

        self.gamma = 0.99
        self.tau   = 0.005

        self.learn_counter      = 0
        self.policy_update_freq = 2

        self.actor         = Actor(self.input_dim, self.action_num).to(self.device)
        self.actor_target  = Actor(self.input_dim, self.action_num).to(self.device)

        self.ensemble_size = 2

        self.ensemble_critic_network = nn.ModuleList()  # ModuleList have not a forward method
        networks = [Critic(self.input_dim, self.action_num) for _ in range(self.ensemble_size)]
        self.ensemble_critic_network.extend(networks)
        self.ensemble_critic_network.to(self.device)

        self.ensemble_critic_network_targets = nn.ModuleList()
        networks_targets = [Critic(self.input_dim, self.action_num) for _ in range(self.ensemble_size)]
        self.ensemble_critic_network_targets.extend(networks_targets)
        self.ensemble_critic_network_targets.to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        [self.ensemble_critic_network_targets[i].load_state_dict(self.ensemble_critic_network[i].state_dict()) for i in range(self.ensemble_size)]

        lr_actor   = 1e-4
        lr_critic  = 1e-3
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = [torch.optim.Adam(self.ensemble_critic_network_targets[i].parameters(), lr=lr_critic) for i in range(self.ensemble_size)]


    def get_action_from_policy(self, state, evaluation=False, noise_scale=0.1):
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor(state_tensor)
            action = action.cpu().data.numpy().flatten()
            if not evaluation:
                # this is part the TD3 too, add noise to the action
                noise = np.random.normal(0, scale=noise_scale, size=self.action_num)
                action = action + noise
                action = np.clip(action, -1, 1)
        self.actor.train()
        return action

    def train_policy(self, experiences):

        self.learn_counter += 1

        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)

        # Convert into tensor
        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards     = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones       = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones   = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_noise = 0.2 * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -0.5, 0.5)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_ens = []
            for target_critic_net in self.ensemble_critic_network_targets:
                t_q_value = target_critic_net(next_states, next_actions)
                target_q_values_ens.append(t_q_value)
            ens = torch.concat(target_q_values_ens, 1)
            target_q_values, _ = torch.min(ens, dim=1)
            target_q_values    = target_q_values.reshape(batch_size, 1)
            q_target           = rewards + self.gamma * (1 - dones) * target_q_values



        for critic_net, optimizer_net in zip(self.ensemble_critic_network, self.critic_optimizer):
            q_value     = critic_net(states, actions)
            critic_loss = F.mse_loss(q_value, q_target)
            # Update the Critic
            optimizer_net.zero_grad()
            critic_loss.backward()
            optimizer_net.step()


        # Update Actor
        if self.learn_counter % self.policy_update_freq == 0:
            actor_q_values_ens = []
            for critic_net in self.ensemble_critic_network:
                actor_q = critic_net(states, actions)
                actor_q_values_ens.append(actor_q)
            ens_actor = torch.concat(actor_q_values_ens, 1)
            actor_q_values, _ = torch.min(ens_actor, dim=1)
            actor_q_values    = actor_q_values.reshape(batch_size, 1)

            actor_loss = -actor_q_values.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


        #     # Update target network params
        #     for target_param, param in zip(self.critic_target.Q1.parameters(), self.critic.Q1.parameters()):
        #         target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        #
        #     for target_param, param in zip(self.critic_target.Q2.parameters(), self.critic.Q2.parameters()):
        #         target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        #
        #     for target_param, param in zip(self.actor_target.act_net.parameters(), self.actor.act_net.parameters()):
        #         target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
