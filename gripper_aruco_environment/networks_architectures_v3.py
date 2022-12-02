import torch
import torch.nn as nn

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(

            nn.Linear(obs_dim + action_dim, hidden_dim[0]),
            nn.ReLU(),

            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),

            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),

            nn.Linear(hidden_dim[2], 1)
        )

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, action_dim):
        super(Critic, self).__init__()

        self.Q1 = QFunction(input_size, action_dim, hidden_size)
        self.Q2 = QFunction(input_size, action_dim, hidden_size)

    def forward(self, state, action):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, action_dim):
        super(Actor, self).__init__()
        self.act_net = nn.Sequential(

            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),

            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            #nn.BatchNorm1d(hidden_size[1], affine=True),

            nn.Linear(hidden_size[1], hidden_size[2]),
            nn.ReLU(),
            #nn.BatchNorm1d(hidden_size[2], affine=True),

            nn.Linear(hidden_size[2], action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        x = self.act_net(state)
        return x
