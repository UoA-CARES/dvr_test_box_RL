import torch
import torch.nn as nn

class Special_Critic(nn.Module):
    def __init__(self, observation_size, num_actions):
        super(Special_Critic, self).__init__()

        self.hidden_size = [1024, 1024]

        self.Q1 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1)
        )

    def forward(self, state, action):
        obs_action = torch.cat([state, action], dim=1)
        q1 = self.Q1(obs_action)
        return q1
