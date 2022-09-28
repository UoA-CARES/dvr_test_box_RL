
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input_v):
        return input_v.reshape(input_v.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input_v):
        return input_v.reshape(input_v.size(0), 256, 32, 32)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.latent_size = 32

        self.encoder_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=2, padding=2),
            nn.ReLU(),
            Flatten()
        )
        self.fc1 = nn.Linear(in_features=262144, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=self.latent_size)
        self.fc3 = nn.Linear(in_features=256, out_features=self.latent_size)



    def reparametrization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        z = mu + std * eps
        return z

    def encode_img(self, x):
        h = self.encoder_net(x)
        print(h.shape)
        h_dense = self.fc1(h)
        mu, log_var = self.fc2(h_dense), self.fc3(h_dense)
        z = self.reparametrization(mu, log_var)
        return z, mu, log_var

    def forward(self, x):
        z, mu, log_var = self.encode_img(x)
        return z, mu, log_var



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.latent_size = 32

        self.fc4 = nn.Linear(in_features=self.latent_size, out_features=1024)
        self.fc5 = nn.Linear(in_features=1024, out_features=262144)

        self.decoder_net = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,  out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,  out_channels=3,  kernel_size=(6, 6), stride=2, padding=1),
            nn.Sigmoid()
        )


    def decode_img(self, z):
        h_dense_1 = self.fc4(z)
        h_dense_2 = self.fc5(h_dense_1)
        xr = self.decoder_net(h_dense_2)
        print(xr.shape)
        return xr

    def forward(self, z):
        xr = self.decode_img(z)
        return xr
