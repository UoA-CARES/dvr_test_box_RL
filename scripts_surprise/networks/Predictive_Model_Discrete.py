"""
Predict outputs  a point estimate e.g. discrete value
"""
import torch
import torch.nn as nn


class Transition_Network_Discrete(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Transition_Network_Discrete, self).__init__()

        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.hidden_size = [512, 512]

        self.prediction_net = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size[1], out_features=self.output_dim),
        )

    def forward(self, state, action):
        x   = torch.cat([state, action], dim=1)
        out = self.prediction_net(x)
        return out
