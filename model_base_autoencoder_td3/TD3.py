"""
This is tradition TD3 and testing idea about sampling episodes
"""


import logging
import numpy as np
import torch
import torch.nn.functional as F

from Networks import Actor_Normal as Actor
from Networks import Critic_Normal as Critic

class TD3:

    def __init__(self, device, obs_dim, action_dim, max_action_value):

        # ------------------- Hyperparameters ---------------------- #
        actor_lr  = 1e-4
        critic_lr = 1e-3

        self.max_action_value = max_action_value
        self.tau   = 0.005
        self.gamma = 0.99

        self.update_counter     = 0
        self.policy_freq_update = 2

        self.device     = device
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        # ----------------------------------------------------------#

        # main networks RL Agent
        self.actor   = Actor(self.obs_dim, self.action_dim, self.max_action_value).to(self.device)
        self.critic  = Critic(self.obs_dim, self.action_dim).to(self.device)

        # target networks RL
        self.actor_target  = Actor(self.obs_dim, self.action_dim, self.max_action_value).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)

        # ----------------- copy weights and bias from main to target networks ----------#
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ----------------------------------------------------------------------------------------------- #
        self.actor.train(True)
        self.critic.train(True)
        self.critic_target.train(True)
        self.actor_target.train(True)

    def select_action_from_policy(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            action = self.actor(state_tensor)
            action = action.cpu().data.numpy().flatten()
        return action

    def train_policy(self, experiences):

        self.update_counter += 1

        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)

        # Convert into tensor
        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards     = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones       = torch.FloatTensor(np.asarray(dones)).to(self.device)

        # Reshape in the right order
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones   = dones.unsqueeze(0).reshape(batch_size, 1)

        # update the critic part
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_noise = 0.2 * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -0.5, 0.5)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-self.max_action_value, max=self.max_action_value)

            target_q_values_one, target_q_values_two = self.critic_target(next_states, next_actions)
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_vals_q1, q_vals_q2 = self.critic(states, actions)

        critic_loss_1 = F.mse_loss(q_vals_q1, q_target)
        critic_loss_2 = F.mse_loss(q_vals_q2, q_target)
        critic_loss_total = critic_loss_1 + critic_loss_2

        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
        self.critic_optimizer.step()

        # Update the actor and soft updates of targets networks
        if self.update_counter % self.policy_freq_update == 0:
            actor_action       = self.actor(states)
            actor_q1, actor_q2 = self.critic(states, actor_action)

            actor_q_min = torch.minimum(actor_q1, actor_q2)
            actor_loss  = - actor_q_min.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_optimizer.step()

            # Update target networks
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def save_models(self, filename):
        torch.save(self.actor.state_dict(), f'models/{filename}_actor_model.pht')
        torch.save(self.critic.state_dict(), f'models/{filename}_critic_mode.pht')
        logging.info("models has been saved...")

    def load_models(self, filename):
        self.actor.load_state_dict(torch.load(f'models/{filename}_actor_model.pht'))
        self.critic.load_state_dict(torch.load(f'models/{filename}_critic_mode.pht'))
        logging.info("models has been loaded...")