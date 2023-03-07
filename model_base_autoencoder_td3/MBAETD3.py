
import logging

import numpy as np
import torch
import torch.nn.functional as F

from Networks import WorldModel
from Networks import Decoder
from Networks import RewardModel

from Networks import Actor_AE as Actor
from Networks import Critic_AE as Critic


logging.basicConfig(level=logging.INFO)


class MB_AE_TD3:
    def __init__(self, device, latent_dim, action_dim, max_action_value):

        # ------------------- Hyperparameters ---------------------- #
        encoder_lr = 1e-3
        decoder_lr = 1e-3

        actor_lr  = 1e-4
        critic_lr = 1e-3

        world_model_lr  = 1e-3
        reward_model_lr = 1e-3

        self.max_action_value = max_action_value
        self.tau   = 0.005
        self.gamma = 0.99

        self.update_counter     = 0
        self.policy_freq_update = 2

        self.device     = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        # ----------------------------------------------------------#

        # main networks RL Agent
        self.actor   = Actor(self.latent_dim, self.action_dim, self.max_action_value).to(self.device)
        self.critic  = Critic(self.latent_dim, self.action_dim).to(self.device)
        self.decoder = Decoder(self.latent_dim).to(self.device)

        # main networks Models
        self.world_model  = WorldModel(self.latent_dim).to(self.device)
        self.reward_model = RewardModel(self.latent_dim).to(self.device)

        # target networks RL
        self.actor_target  = Actor(self.latent_dim, self.action_dim, self.max_action_value).to(self.device)
        self.critic_target = Critic(self.latent_dim, self.action_dim).to(self.device)

        # ------------------------- tie encoders --------------------------------------#
        # tie encoders between actor and critic
        self.actor.encoder_net.copy_conv_weights_from(self.critic.encoder_net)
        # tie encoders between world transition model and critic
        self.world_model.encoder_net.copy_conv_weights_from(self.critic.encoder_net)
        # tie encoders between reward model  and critic
        self.reward_model.encoder_net.copy_conv_weights_from(self.critic.encoder_net)
        # -----------------------------------------------------------------------------#


        # copy weights and bias from main to target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        # ------------------------------------ Optimizer ------------------------------------------------ #
        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder_net.parameters(), lr=encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=decoder_lr, weight_decay=1e-7)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.world_model_optimizer  = torch.optim.Adam(self.world_model.parameters(), lr=world_model_lr)
        self.reward_model_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=reward_model_lr)
        # ----------------------------------------------------------------------------------------------- #

        self.actor.train(True)
        self.critic.train(True)
        self.decoder.train(True)
        self.world_model.train(True)
        self.reward_model.train(True)
        self.critic_target.train(True)
        self.actor_target.train(True)

    def select_action_from_policy(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)  # torch.Size([1, 3, 84, 84])
            action = self.actor(state_tensor)
            action = action.cpu().data.numpy().flatten()
        return action


    def generate_dream_samples(self, experiences):
        states, _, _, _, _ = experiences
        batch = len(states)

        states = np.asarray(states)
        states_tensor = torch.FloatTensor(states).to(self.device)

        # batch of actions from currently policy
        with torch.no_grad():
            action_tensor = self.actor(states_tensor)
            actions       = action_tensor.cpu().data.numpy()

        # predict batch new "dream" samples
        with torch.no_grad():
            z_vector_next_tensor = self.world_model(states_tensor, action_tensor, detach_encoder=True)
            next_rec_obs_tensor  = self.decoder(z_vector_next_tensor)
            next_rec_prediction  = next_rec_obs_tensor.cpu().data.numpy()

        # predict batch new "dream" reward
        with torch.no_grad():
            reward_prediction_tensor = self.reward_model(states_tensor, action_tensor, detach_encoder=True)
            reward_prediction        = reward_prediction_tensor.cpu().data.numpy()

        # I will assume the done as terminator states here are always False the generated data
        dones = np.full((batch, 1), False)

        # keep in mind that those values are already numpy arrays
        return states, actions, reward_prediction, next_rec_prediction, dones


    def train_world_model(self, experiences):

        states, actions, _, next_states, _ = experiences

        # Convert into tensor
        # State and Next State are images here
        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)

        z_vector_next_true = self.world_model.encoder_net(next_states, detach=True)
        z_vector_next_prediction = self.world_model(states, actions, detach_encoder=True)

        model_loss = F.mse_loss(z_vector_next_true, z_vector_next_prediction)

        self.world_model_optimizer.zero_grad()
        model_loss.backward()
        self.world_model_optimizer.step()

        logging.info(f"Transition model loss: {model_loss.item()}")

    def train_reward_model(self, experiences):

        states, actions, rewards, _, _ = experiences
        batch_size = len(states)

        states  = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)

        reward_prediction = self.reward_model(states, actions, detach_encoder=True)

        reward_model_loss = F.mse_loss(rewards, reward_prediction)

        self.reward_model_optimizer.zero_grad()
        reward_model_loss.backward()
        self.reward_model_optimizer.step()

        logging.info(f"Reward model loss: {reward_model_loss.item()}")

    def train_policy(self, experiences):
        self.update_counter += 1

        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)

        # Convert into tensor
        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

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
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
        self.critic_optimizer.step()

        # Update the actor and soft updates of targets networks
        if self.update_counter % self.policy_freq_update == 0:
            actor_action = self.actor(states, detach_encoder=True)
            actor_q1, actor_q2           = self.critic(states, actor_action, detach_encoder=True)

            actor_q_min = torch.minimum(actor_q1, actor_q2)
            actor_loss  = - actor_q_min.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
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
        torch.save(self.actor.state_dict(), f'models/{filename}_actor_model.pht')
        torch.save(self.critic.state_dict(), f'models/{filename}_critic_mode.pht')

        torch.save(self.critic.encoder_net.state_dict(), f'models/{filename}_encoder_model.pht')
        torch.save(self.decoder.state_dict(), f'models/{filename}_decoder_model.pht')

        torch.save(self.world_model.state_dict(), f'models/{filename}_world_model.pht')
        torch.save(self.reward_model.state_dict(), f'models/{filename}_reward_model.pht')
