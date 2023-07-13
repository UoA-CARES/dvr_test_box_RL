
import os
import numpy as np

import torch
import torch.nn.functional as F

from networks import Encoder
from networks import Actor
from networks import Critic


class TD3_Pixel:
    def __init__(self, latent_size=50, action_num=1, device="cuda", k=3):

        self.latent_size = latent_size
        self.action_num  = action_num
        self.device      = device

        self.k = k*3  # number of stack frames, K*3  if I am using color images

        self.gamma = 0.99
        self.tau   = 0.005
        self.ensemble_size = 5

        self.learn_counter      = 0
        self.policy_update_freq = 2

        self.encoder = Encoder(latent_dim=self.latent_size, k=self.k).to(self.device)
        self.actor   = Actor(self.latent_size, self.action_num, self.encoder).to(self.device)
        self.critic  = Critic(self.latent_size, self.action_num, self.encoder).to(self.device)

        self.actor_target  = Actor(self.latent_size, self.action_num, self.encoder).to(self.device)
        self.critic_target = Critic(self.latent_size, self.action_num, self.encoder).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        lr_actor   = 1e-4
        lr_critic  = 1e-3
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),   lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),  lr=lr_critic)


    def select_action_from_policy(self, state, evaluation=False, noise_scale=0.1):
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
        self.encoder.train()
        self.actor.train()
        self.critic.train()

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

            target_q_values_one, target_q_values_two = self.critic_target(next_states, next_actions)
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic(states, actions)

        critic_loss_1 = F.mse_loss(q_values_one, q_target)
        critic_loss_2 = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_1 + critic_loss_2

        # Update the Critic
        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        self.critic_optimizer.step()

        # Update Actor
        if self.learn_counter % self.policy_update_freq == 0:
            actor_q_one, actor_q_two = self.critic(states, self.actor(states))
            actor_q_values = torch.minimum(actor_q_one, actor_q_two)
            actor_loss = -actor_q_values.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target network params
            for target_param, param in zip(self.critic_target.Q1.parameters(), self.critic.Q1.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target.Q2.parameters(), self.critic.Q2.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.actor_target.act_net.parameters(), self.actor.act_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_models(self, filename):
        dir_exists = os.path.exists("models")
        if not dir_exists:
            os.makedirs("models")
        torch.save(self.actor.state_dict(),   f'models/{filename}_actor.pht')
        torch.save(self.critic.state_dict(),  f'models/{filename}_critic.pht')
        torch.save(self.encoder.state_dict(), f'models/{filename}_encoder.pht')
        print("models has been saved...")
