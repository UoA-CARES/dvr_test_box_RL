import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input_v):
        return input_v.reshape(input_v.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input_v):
        return input_v.reshape(input_v.size(0), 256, 28, 28)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.latent_size = 14

        self.encoder_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=2, padding=0),
            nn.ReLU(),
            Flatten()
        )
        self.fc1 = nn.Linear(in_features=200704, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=self.latent_size)
        self.fc3 = nn.Linear(in_features=256, out_features=self.latent_size)

    def reparametrization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        z = mu + std * eps
        return z

    def encode_img(self, x):
        h = self.encoder_net(x)
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
        self.latent_size = 14

        self.fc4 = nn.Linear(in_features=self.latent_size, out_features=1024)
        self.fc5 = nn.Linear(in_features=1024, out_features=200704)

        self.decoder_net = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(5, 5), stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(8, 8), stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(8, 8), stride=2, padding=1),
            nn.Sigmoid()
        )

    def decode_img(self, z):
        h_dense_1 = self.fc4(z)
        h_dense_2 = self.fc5(h_dense_1)
        xr = self.decoder_net(h_dense_2)
        return xr

    def forward(self, z):
        xr = self.decode_img(z)
        return xr


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
        z = mu + std * eps
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
        z = self.reparameterize(mu, log_var)
        x_rec = self.decode(z)
        return x_rec, mu, log_var, z
