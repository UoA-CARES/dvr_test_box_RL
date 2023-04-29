"""
Predict output distribution rather than point estimate e.g. discrete value
This is a regression problem so the outputs are mean and variance
"""
import torch
import torch.nn as nn


class Transition_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Transition_Network, self).__init__()

        self.input_dim   = input_dim
        self.output_dim  = output_dim
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

    def forward(self, state, action):
        x   = torch.cat([state, action], dim=1)
        u   = self.mean_layer(x)
        std = self.std_layer(x) + 1e-6
        return torch.distributions.Normal(u, std)
