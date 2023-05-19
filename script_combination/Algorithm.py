
import os
import copy

import logging
logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from cares_reinforcement_learning.util import helpers as hlp
from skimage.metrics import structural_similarity as ssim
from PIL import ImageChops

from networks import Actor
from networks import Critic
from networks import Encoder
from networks import Decoder
from networks import EPPM


class Algorithm:

    def __init__(self, latent_size, action_num, device, k, color=True):

        self.latent_size = latent_size
        self.action_num  = action_num
        self.device      = device

        self.k = k*3 if color is True else k  # numer of stack frames, K*3  if I am using color images

        self.gamma = 0.99
        self.tau   = 0.005
        self.ensemble_size = 10

        self.learn_counter      = 0
        self.policy_update_freq = 2

        self.encoder = Encoder(latent_dim=self.latent_size, k=self.k).to(self.device)
        self.decoder = Decoder(latent_dim=self.latent_size, k=self.k).to(self.device)

        self.actor  = Actor(self.latent_size, self.action_num, self.encoder).to(self.device)
        self.critic = Critic(self.latent_size, self.action_num, self.encoder).to(self.device)

        self.actor_target  = Actor(self.latent_size, self.action_num, self.encoder).to(self.device)
        self.critic_target = Critic(self.latent_size, self.action_num, self.encoder).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.eppm = nn.ModuleList()
        networks = [EPPM(self.latent_size, self.action_num) for _ in range(self.ensemble_size)]
        self.eppm.extend(networks)
        self.eppm.to(self.device)

        lr_actor   = 1e-4
        lr_critic  = 1e-3
        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(),   lr=lr_actor)
        self.critic_optimizer  = torch.optim.Adam(self.critic.parameters(),  lr=lr_critic)

        lr_encoder = 1e-3
        lr_decoder = 1e-3
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr_encoder)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr_decoder, weight_decay=1e-7)

        lr_eppm      = 1e-4
        w_decay_epp  = 1e-3
        self.eppm_optimizers = [torch.optim.Adam(self.eppm[i].parameters(), lr=lr_eppm, weight_decay=w_decay_epp) for i in range(self.ensemble_size)]


    def get_action_from_policy(self, state, evaluation=False, noise_scale=0.1):
        #self.actor.eval()
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
        #self.actor.train()
        return action

    def get_intrinsic_values(self, state, action, next_state, plot_flag):

        with torch.no_grad():
            state_tensor  = torch.FloatTensor(state).to(self.device)
            state_tensor  = state_tensor.unsqueeze(0)
            action_tensor = torch.FloatTensor(action).to(self.device)
            action_tensor = action_tensor.unsqueeze(0)

            surprise_rate = self.get_surprise_rate(state_tensor, action_tensor, next_state)
            novelty_rate  = self.get_novelty_rate(state_tensor, plot_flag)

            return surprise_rate, novelty_rate


    def get_surprise_rate(self, state_tensor_img, action_tensor, next_state_array_img):

        with torch.no_grad():
            latent_state  = self.encoder(state_tensor_img, detach=True)

            predict_mean_set, predict_std_set = [], []
            for network in self.eppm:
                network.eval()
                predicted_distribution = network(latent_state, action_tensor)
                mean = predicted_distribution.mean
                std  = predicted_distribution.stddev
                predict_mean_set.append(mean.detach().cpu().numpy())
                predict_std_set.append(std.detach().cpu().numpy())

            ensemble_prediction_means = np.concatenate(predict_mean_set, axis=0)
            ensemble_prediction_stds  = np.concatenate(predict_std_set, axis=0)

            z_next_latent_prediction = np.mean(ensemble_prediction_means, axis=0)
            uncertainty_prediction   = np.mean(ensemble_prediction_stds, axis=0) # std of each element in the z _vector

            avr_uncertainty = np.mean(uncertainty_prediction) # avr uncertainty in the prediction

            z_next_latent_prediction_tensor = torch.FloatTensor(z_next_latent_prediction).to(self.device)

            next_state_rec_img       = self.decoder(z_next_latent_prediction_tensor)
            reconstr_stack_next_img  = next_state_rec_img.cpu().numpy()[0]  # --> (k , 84 ,84)

            target_images    = next_state_array_img / 255
            ssim_index_total = ssim(target_images, reconstr_stack_next_img, full=False, data_range=1, channel_axis=0)
            surprise_rate    = (1 - ssim_index_total) + avr_uncertainty

            return surprise_rate

    def get_novelty_rate(self, state_tensor_img, flag):

        with torch.no_grad():
            z_vector = self.encoder(state_tensor_img)
            rec_img  = self.decoder(z_vector) # Note: rec_img is a stack of k images --> (1, k , 84 ,84),

            original_stack_imgs  = state_tensor_img.cpu().numpy()[0]  # --> (k , 84 ,84)
            reconstruction_stack = rec_img.cpu().numpy()[0]           # --> (k , 84 ,84)

            target_images     = original_stack_imgs / 255
            ssim_index_total  = ssim(target_images, reconstruction_stack, full=False, data_range=original_stack_imgs.max() - original_stack_imgs.min(), channel_axis=0)
            novelty_rate      = 1 - ssim_index_total

            if flag:
                self.plot_img_reconstruction(original_stack_imgs, reconstruction_stack)

            return novelty_rate

    def plot_img_reconstruction(self, original_img, reconstruction_img):

        original_img       = np.moveaxis(original_img, 0, -1)
        reconstruction_img = np.moveaxis(reconstruction_img, 0, -1)

        original_img       = np.array_split(original_img, 3, axis=2)
        reconstruction_img = np.array_split(reconstruction_img, 3, axis=2)

        plt.subplot(2, 3, 1)
        plt.title("Image Input one")
        #plt.imshow(original_img[0], vmin=0, vmax=1)
        plt.imshow(original_img[0])

        plt.subplot(2, 3, 2)
        plt.title("Image Input two")
        #plt.imshow(original_img[1], vmin=0, vmax=1)
        plt.imshow(original_img[1])

        plt.subplot(2, 3, 3)
        plt.title("Image Input three")
        #plt.imshow(original_img[2], vmin=0, vmax=1)
        plt.imshow(original_img[2])

        plt.subplot(2, 3, 4)
        plt.title("Image Reconstruction one")
        #plt.imshow(reconstruction_img[0], vmin=0, vmax=1)
        plt.imshow(reconstruction_img[0])

        plt.subplot(2, 3, 5)
        plt.title("Image Reconstruction two")
        #plt.imshow(reconstruction_img[1], vmin=0, vmax=1)
        plt.imshow(reconstruction_img[1])

        plt.subplot(2, 3, 6)
        plt.title("Image Reconstruction three")
        #plt.imshow(reconstruction_img[2], vmin=0, vmax=1)
        plt.imshow(reconstruction_img[2])

        # difference = abs(original_img[0] - reconstruction_img[0])
        # plt.subplot(1, 3, 3)
        # plt.title("Difference")
        # plt.imshow(difference, vmin=0, vmax=1)

        #plt.savefig(f"plot_results/AE-TD3_{env_name}_image_reconstruction.png")
        #plt.show()
        plt.pause(0.01)

    def train_policy(self, experiences):
        self.encoder.train()
        self.decoder.train()
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

        # Update Autoencoder
        z_vector = self.encoder(states)
        rec_obs  = self.decoder(z_vector)

        target_images = states / 255 # this because the images 0- 255 and the prediction is [0-1], I did not normalized before to save experiences as Unit8
        rec_loss      = F.mse_loss(target_images, rec_obs)

        latent_loss = (0.5 * z_vector.pow(2).sum(1)).mean()  # add L2 penalty on latent representation
        ae_loss     = rec_loss + 1e-6 * latent_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        ae_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # Update Actor
        if self.learn_counter % self.policy_update_freq == 0:
            actor_q_one, actor_q_two = self.critic(states, self.actor(states, detach_encoder=True),  detach_encoder=True)
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

            # the encoders in target networks are the same of main networks, so I will not update them

    def train_predictive_model(self, experiences):

        states, actions, _, next_states, _ = experiences

        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)

        with torch.no_grad():
            latent_state      = self.encoder(states, detach=True)
            latent_next_state = self.encoder(next_states, detach=True)

        for predictive_network, optimizer in zip(self.eppm, self.eppm_optimizers):
            predictive_network.train()

            #Get the Prediction of each model
            prediction_distribution = predictive_network(latent_state, actions)
            loss_neg_log_likelihood = - prediction_distribution.log_prob(latent_next_state)
            loss_neg_log_likelihood = torch.mean(loss_neg_log_likelihood)

            # Update weights and bias
            optimizer.zero_grad()
            loss_neg_log_likelihood.backward()
            optimizer.step()

    def save_models(self, filename):
        dir_exists = os.path.exists("models")

        if not dir_exists:
            os.makedirs("models")

        torch.save(self.actor.state_dict(),  f'models/{filename}_actor.pht')
        torch.save(self.critic.state_dict(), f'models/{filename}_critic.pht')

        torch.save(self.encoder.state_dict(), f'models/{filename}_encoder.pht')
        torch.save(self.decoder.state_dict(), f'models/{filename}_decoder.pht')

        torch.save(self.eppm.state_dict(), f'models/{filename}_ensemble.pht')  # no sure if this is the correct way to solve ensemble

        print("models has been saved...")
