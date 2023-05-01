

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from networks import Encoder
from networks import Decoder

class Deep_Novelty:
    def __init__(self, latent_dim, device, k=1):

        self.latent_dim = latent_dim
        self.k          = k
        self.device     = device

        self.encoder_net = Encoder(latent_dim=self.latent_dim, k=self.k).to(self.device)
        self.decoder_net = Decoder(latent_dim=self.latent_dim, k=self.k).to(self.device)

        lr_encoder = 1e-3
        lr_decoder = 1e-3

        self.encoder_optimizer = torch.optim.Adam(self.encoder_net.parameters(), lr=lr_encoder)

        #self.decoder_optimizer = torch.optim.Adam(self.decoder_net.parameters(), lr=lr_decoder)
        self.decoder_optimizer = torch.optim.Adam(self.decoder_net.parameters(), lr=lr_decoder, weight_decay=1e-7)


    def train_autoencoder_model(self, experiences):
        states_img, _, _, next_states_img, _ = experiences

        states_img      = torch.FloatTensor(np.asarray(states_img)).to(self.device)
        next_states_img = torch.FloatTensor(np.asarray(next_states_img)).to(self.device)

        z_vector = self.encoder_net(states_img)
        rec_img  = self.decoder_net(z_vector)
        rec_loss = F.mse_loss(states_img, rec_img)
        latent_loss = (0.5 * z_vector.pow(2).sum(1)).mean()  # add L2 penalty on latent representation

        ae_loss = rec_loss + 1e-6 * latent_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        ae_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


    def get_reconstruction_from_model(self, state):

        state_tensor_img = torch.FloatTensor(state).to(self.device)
        state_tensor_img = state_tensor_img.unsqueeze(0)

        self.encoder_net.eval()
        self.decoder_net.eval()

        with torch.no_grad():
            z_vector = self.encoder_net(state_tensor_img)
            rec_img  = self.decoder_net(z_vector)

        return rec_img



    def save_model(self):
        dir_exists = os.path.exists("models")
        filename = "autoencoder"

        if not dir_exists:
            os.makedirs("models")

        torch.save(self.encoder_net.state_dict(), f'models/{filename}_encoder_model.pht')
        torch.save(self.decoder_net.state_dict(), f'models/{filename}_decoder_model.pht')
        print("models has been saved...")


    def load_model(self):
        filename = "autoencoder"
        self.encoder_net.load_state_dict(torch.load(f'models/{filename}_encoder_model.pht'))
        self.decoder_net.load_state_dict(torch.load(f'models/{filename}_decoder_model.pht'))
        print("models has been loaded...")
