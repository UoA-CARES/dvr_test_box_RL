
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from networks.Encoder import Encoder
from networks.weight_initialization import weight_init


class Actor(nn.Module):
    def __init__(self, latent_size, num_actions, learning_rate):

        super(Actor, self).__init__()
        self.encoder_net = Encoder(latent_size)

        self.hidden_size = [1024, 1024]

        self.h_linear_1 = nn.Linear(in_features=latent_size,         out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=num_actions)

        self.apply(weight_init)
        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, state, detach_encoder=False):
        z_vector = self.encoder_net(state, detach=detach_encoder)
        output   = F.relu(self.h_linear_1(z_vector))
        output   = F.relu(self.h_linear_2(output))
        output   = torch.tanh(self.h_linear_3(output))
        return output
