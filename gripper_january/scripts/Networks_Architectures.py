
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


#----------------------------------------------------------------
class QFunction_Normal(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], 1)
        )

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)

class Critic_Normal(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Critic_Normal, self).__init__()

        self.input_size = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = [1024, 1024]

        self.Q1 = QFunction_Normal(self.input_size, self.action_dim, self.hidden_dim)
        self.Q2 = QFunction_Normal(self.input_size, self.action_dim, self.hidden_dim)

    def forward(self, state, action):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2
