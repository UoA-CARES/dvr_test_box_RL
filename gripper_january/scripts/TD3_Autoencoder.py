
import logging

import random
import numpy as np

import torch
import torch.nn.functional as F

from Networks_Architectures import Actor_AE  as Actor
from Networks_Architectures import Critic_AE as Critic
from Networks_Architectures import Decoder

class TD3:
    def __init__(self, device, latent_dim, action_dim):

        # -------- Hyperparameters ----------------------
        hidden_size = [1024, 1024]

        encoder_lr = 1e-3
        decoder_lr = 1e-3

        actor_lr   = 1e-4
        critic_lr  = 1e-3

        self.tau   = 0.005
        self.gamma = 0.99

        self.update_counter     = 0
        self.policy_freq_update = 2

        self.device     = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        # ------------------------------------------------

        # main networks RL Agent
        self.actor   = Actor(self.latent_dim, self.action_dim, hidden_size).to(self.device)
        self.critic  = Critic(self.latent_dim, self.action_dim, hidden_size).to(self.device)
        self.decoder = Decoder(self.latent_dim).to(device)

        # target networks
        self.actor_target  = Actor(self.latent_dim, self.action_dim, hidden_size).to(self.device)
        self.critic_target = Critic(self.latent_dim, self.action_dim, hidden_size).to(self.device)

        # tie encoders between actor and critic
        # any changes in the critic encoder will also be affecting the actor-encoder during the whole training
        self.actor.encoder_net.copy_conv_weights_from(self.critic.encoder_net)

        # tie all net
        #self.actor.encoder_net.copy_all_weights_from(self.critic.encoder_net)


        # copy weights and bias from main to target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Optimizer.
        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder_net.parameters(), lr=encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=decoder_lr, weight_decay=1e-7)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor.train(True)
        self.critic.train(True)
        self.decoder.train(True)
        self.critic_target.train(True)
        self.actor_target.train(True)


    def select_action(self, state):
        with torch.no_grad():
            state_image_tensor = torch.FloatTensor(state)
            state_image_tensor = state_image_tensor.unsqueeze(0).to(self.device)  # torch.Size([1, 3, 84, 84])
            action = self.actor(state_image_tensor)
            action = action.cpu().data.numpy().flatten()
        return action

    def action_sample(self):
        # this function should be in the env file
        action = []
        for i in range(0, self.action_dim):
            a = np.clip(random.uniform(-1, 1), -1, 1)
            action.append(a)
        return action

    def learn(self, experiences):
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
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.critic_target(next_states, next_actions)
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_vals_q1, q_vals_q2 = self.critic(states, actions)

        critic_loss_1     = F.mse_loss(q_vals_q1, q_target)
        critic_loss_2     = F.mse_loss(q_vals_q2, q_target)
        critic_loss_total = critic_loss_1 + critic_loss_2

        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
        self.critic_optimizer.step()

        # Update the actor and soft updates of targets networks
        if self.update_counter % self.policy_freq_update == 0:

            actor_action       = self.actor(states, detach_encoder=True)
            actor_q1, actor_q2 = self.critic(states, actor_action, detach_encoder=True)

            actor_q_min = torch.minimum(actor_q1, actor_q2)
            actor_loss  = - actor_q_min.mean()

            #---------------------------
            # idea: Saturation Penalty
            # upper_saturation =  2.0
            # lower_saturation = -2.0
            # k_saturation     = 1000 # this number is empirical value could be changed
            #
            # saturation_penalty_up = pre_activation - upper_saturation
            # saturation_penalty_up = torch.maximum(saturation_penalty_up, torch.tensor(0))
            #
            # saturation_penalty_down = -pre_activation + lower_saturation
            # saturation_penalty_down = torch.maximum(saturation_penalty_down, torch.tensor(0))
            #
            # saturation_penalty = saturation_penalty_up + saturation_penalty_down
            # saturation_penalty = torch.square(saturation_penalty).mean()
            # saturation_penalty = k_saturation * saturation_penalty
            # total_actor_loss = actor_loss + saturation_penalty
            # print(f"Actor Loss={actor_loss} Penalty={saturation_penalty}")
            # ---------------------------

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
            self.actor_optimizer.step()

            # ------------------------------------- Update target networks --------------- #
            # since I will use the same tau for encoder and actor-critic i can just updata like this
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


        # Update the autoencoder part
        z_vector = self.critic.encoder_net(states)
        rec_obs  = self.decoder(z_vector)

        rec_loss    = F.mse_loss(states, rec_obs)
        latent_loss = (0.5 * z_vector.pow(2).sum(1)).mean()  # add L2 penalty on latent representation

        ae_loss     = rec_loss + 1e-6 * latent_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        ae_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


    def save_models(self, filename):
        torch.save(self.actor.state_dict(),  f'models/{filename}_actor.pht')
        torch.save(self.critic.state_dict(), f'models/{filename}_critic.pht')

        torch.save(self.critic.encoder_net.state_dict(), f'models/{filename}_encoder.pht')
        torch.save(self.decoder.state_dict(), f'models/{filename}_decoder.pht')

        logging.info("models has been saved...")


    def load_models(self, filename):

        self.actor.load_state_dict(torch.load(f'models/{filename}_actor.pht'))
        self.critic.load_state_dict(torch.load(f'models/{filename}_critic.pht'))

        self.critic.encoder_net.load_state_dict(torch.load(f'models/{filename}_encoder.pht'))
        self.decoder.load_state_dict(torch.load(f'models/{filename}_decoder.pht'))

        logging.info("models has been loaded...")


