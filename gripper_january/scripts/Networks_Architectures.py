

import torch
import torch.nn as nn


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Networks for normal with a vector state space TD3
class Actor_Normal(nn.Module):
    def __init__(self, obs_dim, actions_dim, max_action):
        super(Actor_Normal, self).__init__()

        self.input_size  = obs_dim
        self.actions_dim = actions_dim
        self.max_action  = max_action

        self.hidden_size = [1024, 1024]

        self.actor_net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], self.actions_dim),
            nn.Tanh()
        )

    def forward(self, state):
        x = self.actor_net(state)
        return x * self.max_action




