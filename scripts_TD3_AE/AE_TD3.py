
"""
This is my version and  re-implementation of the paper  https://arxiv.org/pdf/1910.01741.pdf
I removed and changed many part here from the original implementation for example the original paper use SAC
"""

import os
import copy
import numpy as np

import torch
import torch.nn.functional as F


class AE_TD3:
    def __init__(self,
                 actor_network,
                 critic_network,
                 decoder_network,
                 gamma,
                 tau,
                 action_num,
                 latent_size,
                 device):

        self.device = device
        self.gamma  = gamma
        self.tau    = tau

        self.learn_counter      = 0
        self.policy_update_freq = 2
        self.action_num         = action_num
        self.latent_dim         = latent_size

        self.actor_net  = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        # tie encoders between actor and critic, with this, any changes in the critic encoder
        # will also be affecting the actor-encoder during the WHOLE training

        #self.actor_net.encoder_net.copy_conv_weights_from(self.critic_net.encoder_net) # original paper do this
        self.actor_net.encoder_net.copy_all_weights_from(self.critic_net.encoder_net) # I use this version

        self.actor_target_net  = copy.deepcopy(self.actor_net)
        self.critic_target_net = copy.deepcopy(self.critic_net)

        self.decoder_net = decoder_network.to(device)

        lr_actor   = 1e-4
        lr_critic  = 1e-3

        lr_encoder = 1e-3
        lr_decoder = 1e-3

        self.encoder_optimizer = torch.optim.Adam(self.critic_net.encoder_net.parameters(), lr=lr_encoder)
        self.decoder_optimizer = torch.optim.Adam(self.decoder_net.parameters(), lr=lr_decoder, weight_decay=1e-7)
        self.actor_optimizer   = torch.optim.Adam(self.actor_net.parameters(),   lr=lr_actor)
        self.critic_optimizer  = torch.optim.Adam(self.critic_net.parameters(),  lr=lr_critic)



    def get_action_from_policy(self, state, evaluation=False, noise_scale=0.1):
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action       = self.actor_net(state_tensor)
            action       = action.cpu().data.numpy().flatten()
            if not evaluation:
                # this is part the TD3 too, add noise to the action
                noise  = np.random.normal(0, scale=noise_scale, size=self.action_num)
                action = action + noise
                action = np.clip(action, -1, 1)
        self.actor_net.train()
        return action


    def train_policy(self, experiences):
        self.learn_counter += 1

        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)

        # Convert into tensor
        states  = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones   = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            next_actions = self.actor_target_net(next_states)
            target_noise = 0.2 * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -0.5, 0.5)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.critic_target_net(next_states, next_actions)
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)

        critic_loss_1 = F.mse_loss(q_values_one, q_target)
        critic_loss_2 = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_1 + critic_loss_2

        # Update the Critic
        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        self.critic_optimizer.step()

        # Update Autoencoder
        z_vector = self.critic_net.encoder_net(states)
        rec_obs  = self.decoder_net(z_vector)
        rec_loss    = F.mse_loss(states, rec_obs)
        latent_loss = (0.5 * z_vector.pow(2).sum(1)).mean()  # add L2 penalty on latent representation

        ae_loss = rec_loss + 1e-6 * latent_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        ae_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # Update Actor
        if self.learn_counter % self.policy_update_freq == 0:
            actor_q_one, actor_q_two = self.critic_net(states, self.actor_net(states, detach_encoder=True),  detach_encoder=True)
            actor_q_values = torch.minimum(actor_q_one, actor_q_two)
            actor_loss = -actor_q_values.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target network params
            for target_param, param in zip(self.critic_target_net.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.actor_target_net.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def save_models(self, filename):
        dir_exists = os.path.exists("models")

        if not dir_exists:
            os.makedirs("models")
        torch.save(self.actor_net.state_dict(),  f'models/{filename}_actor.pht')
        torch.save(self.critic_net.state_dict(), f'models/{filename}_critic.pht')

        torch.save(self.critic_net.encoder_net.state_dict(),  f'models/{filename}_encoder.pht')
        torch.save(self.decoder_net.state_dict(),  f'models/{filename}_decoder.pht')
        print("models has been saved...")

    def load_models(self, filename):
        self.actor_net.load_state_dict(torch.load(f'models/{filename}_actor.pht'))
        self.critic_net.load_state_dict(torch.load(f'models/{filename}_critic.pht'))

        self.critic_net.encoder_net.load_state_dict(torch.load(f'models/{filename}_encoder.pht'))
        self.decoder_net.load_state_dict(torch.load(f'models/{filename}_decoder.pht'))
        print("models has been loaded...")
