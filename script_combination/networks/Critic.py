
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.Encoder import Encoder
from networks.weight_initialization import weight_init


class Critic(nn.Module):
    def __init__(self, latent_size, num_actions, encoder):
        super(Critic, self).__init__()

        #self.encoder_net  = Encoder(latent_size)
        self.encoder_net = encoder

        self.hidden_size  = [1024, 1024]

        # Q1 architecture
        self.h_linear_1 = nn.Linear(latent_size + num_actions, self.hidden_size[0])
        self.h_linear_2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.h_linear_3 = nn.Linear(self.hidden_size[1], 1)

        # Q2 architecture
        self.h_linear_12 = nn.Linear(latent_size + num_actions, self.hidden_size[0])
        self.h_linear_22 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.h_linear_32 = nn.Linear(self.hidden_size[1], 1)

        self.apply(weight_init)


    def forward(self, state, action, detach_encoder=False):
        z_vector = self.encoder_net(state, detach=detach_encoder)

        obs_action = torch.cat([z_vector, action], dim=1)

        q1 = F.relu(self.h_linear_1(obs_action))
        q1 = F.relu(self.h_linear_2(q1))
        q1 = self.h_linear_3(q1)

        q2 = F.relu(self.h_linear_12(obs_action))
        q2 = F.relu(self.h_linear_22(q2))
        q2 = self.h_linear_32(q2)

        return q1, q2
