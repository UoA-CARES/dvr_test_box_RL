"""
Ensemble of Probabilistic Predictive Model (EPPM)
Predict output distribution rather than point estimate e.g. discrete value
This is a regression problem so the outputs are mean and variance
"""

import torch
import torch.nn as nn

from networks.weight_initialization import weight_init

class EPPM(nn.Module):
    def __init__(self, latent_size, num_actions, encoder):
        super(EPPM, self).__init__()

        #self.encoder_net = Encoder(latent_size)
        self.encoder_net = encoder


        self.input_dim   = latent_size + num_actions
        self.output_dim  = latent_size
        self.hidden_size = [512, 512]

        self.mean_layer = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size[1], out_features=self.output_dim),
        )

        self.std_layer = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size[1], out_features=self.output_dim),
            nn.Softplus()
        )

        self.apply(weight_init)

    def forward_encoder(self, state, detach_encoder):
        latent_vector = self.encoder_net(state, detach=detach_encoder)
        return latent_vector

    def forward(self, state, action, detach_encoder=False):
        z_vector = self.forward_encoder(state, detach_encoder)
        x   = torch.cat([z_vector, action], dim=1)
        u   = self.mean_layer(x)
        std = self.std_layer(x) + 1e-6
        return torch.distributions.Normal(u, std)
