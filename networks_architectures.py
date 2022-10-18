"""
Description: Networks Architectures

Author: David Valencia

"""

import torch
import torch.nn as nn


class VanillaVAE(nn.Module):
    def __init__(self, latent_dim):
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim

        # Build Encoder Part:
        self.encoder_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        self.fc_1   = nn.Linear(8192, 1024)
        self.fc_mu  = nn.Linear(1024, self.latent_dim)
        self.fc_var = nn.Linear(1024, self.latent_dim)

        # Build Decoder Part
        self.fc_2 = nn.Linear(self.latent_dim, 1024)
        self.fc_3 = nn.Linear(1024, 8192)

        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,  output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
            #nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.BatchNorm2d(32),
            #nn.LeakyReLU(),
            #nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1),
            #nn.Sigmoid()
        )

    def encode(self, x_input):
        encode  = self.encoder_net(x_input)  # torch.Size([batch_size, 512, 4, 4])
        encode  = torch.flatten(encode, start_dim=1)  # torch.Size([batch_size, 8192])
        fc      = self.fc_1(encode)
        mu      = self.fc_mu(fc)
        log_var = self.fc_var(fc)
        return [mu, log_var]

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        z   = mu + std * eps
        return z

    def decode(self, z_vector):
        x_rec = self.fc_2(z_vector)
        x_rec = self.fc_3(x_rec)  # torch.Size([16, 8192])
        x_rec = x_rec.view(-1, 512, 4, 4)
        x_rec = self.decoder_net(x_rec)
        x_rec = self.final_layer(x_rec)
        return x_rec

    def forward(self, x_input):
        mu, log_var = self.encode(x_input)
        z     = self.reparameterize(mu, log_var)
        x_rec = self.decode(z)
        return x_rec, mu, log_var, z




# -------------------Network for Forward Predictive Model -----------------------------#
class ForwardModelPrediction(nn.Module):
    def __init__(self):
        super(ForwardModelPrediction, self).__init__()

        self.number_mixture_gaussians = 3
        self.input_size_latent  = 32  # size z vector + size of action vector
        self.input_size_actions = 4
        self.output_size        = 32  # size of the next z prediction vector

        self.input_size = self.input_size_latent + self.input_size_actions  # size z vector + size of action vector

        self.hidden_size = [32, 32, 32]

        self.initial_shared_layer = nn.Sequential(
            nn.BatchNorm1d(self.input_size),
            nn.Linear(self.input_size, self.hidden_size[0], bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.Linear(self.hidden_size[0], self.hidden_size[1], bias=True),
            nn.ReLU(),
        )

        self.phi_layer = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size[1]),
            nn.Linear(self.hidden_size[1], self.hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size[2], self.output_size * self.number_mixture_gaussians),
            nn.Softmax()
        )

        self.mean_layer = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size[1]),
            nn.Linear(self.hidden_size[1], self.hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size[2], self.output_size * self.number_mixture_gaussians)
        )

        self.std_layer = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size[1]),
            nn.Linear(self.hidden_size[1], self.hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size[2], self.output_size * self.number_mixture_gaussians),
            nn.Softplus()
        )

    def forward(self, z_vector, action_vector):

        x = torch.cat([z_vector, action_vector], dim=1)  # Concatenates the seq tensors in the given dimension
        x = self.initial_shared_layer(x)

        u = self.mean_layer(x)
        std = torch.clamp(self.std_layer(x), min=0.001)
        phi = self.phi_layer(x)

        u   = torch.reshape(u, (-1, self.output_size, self.number_mixture_gaussians))
        std = torch.reshape(std, (-1, self.output_size, self.number_mixture_gaussians))
        phi = torch.reshape(phi, (-1, self.output_size, self.number_mixture_gaussians))

        mix = torch.distributions.Categorical(phi)
        norm_distr = torch.distributions.Normal(u, std)

        gmm = torch.distributions.MixtureSameFamily(mix, norm_distr)

        return gmm



