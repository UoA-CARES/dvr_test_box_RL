

import logging

import numpy as np
import torch
import torch.nn.functional as F

from Networks import WorldModel
from Networks import Decoder
from Networks import Actor_AE as Actor
from Networks import Critic_AE as Critic

logging.basicConfig(level=logging.INFO)


class MB_AE_TD3:
    def __init__(self, device, latent_dim, action_dim):


        world_model_lr = 1e-3

        self.device     = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.actor   = Actor(self.latent_dim, self.action_dim).to(self.device)
        self.critic  = Critic(self.latent_dim, self.action_dim).to(self.device)
        self.decoder = Decoder(self.latent_dim).to(self.device)

        self.world_model = WorldModel(self.latent_dim).to(self.device)

        # tie encoders between actor and critic
        self.actor.encoder_net.copy_conv_weights_from(self.critic.encoder_net)


        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=world_model_lr)




    def generate_dream_samples(self, experiences):
        #states, _, _, _, _ = experiences
        #batch_size = len(states)
        pass


    def train_world_model(self, experiences):
        # esto lo unico que aprende es a predecir el siguiente z'
        states, actions, _, next_states, _ = experiences

        # Convert into tensor
        # State and Next State are images here
        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)

        z_vector_next_true = self.critic.encoder_net(next_states, detach=True)
        z_vector_in        = self.critic.encoder_net(states, detach=True)

        # here everything are vectors
        z_vector_next_prediction = self.world_model(z_vector_in, actions)

        model_loss = F.mse_loss(z_vector_next_true, z_vector_next_prediction)

        self.world_model_optimizer.zero_grad()
        model_loss.backward()
        self.world_model_optimizer.step()

        logging.info(f"Transition model loss: {model_loss.item()}")















