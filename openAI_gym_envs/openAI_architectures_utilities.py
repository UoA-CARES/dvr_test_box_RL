import torch
import torch.nn as nn

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim, max_value):
        super(Actor, self).__init__()
        self.max_value   = max_value
        self.encoder_net = Encoder(latent_dim)

        self.hidden_size = [1024, 1024]
        self.act_net = nn.Sequential(

            nn.Linear(latent_dim, self.hidden_size[0]),
            nn.ReLU(),

            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),

            nn.Linear(self.hidden_size[1], action_dim),
            nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, state, detach_encoder=False):
        z_vector = self.encoder_net(state, detach=detach_encoder)
        output   = self.act_net(z_vector)
        return output * self.max_value
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class QFunction(nn.Module):
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
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Critic(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(Critic, self).__init__()

        self.encoder_net = Encoder(latent_dim)
        self.hidden_dim  = [1024, 1024]

        self.Q1 = QFunction(latent_dim, action_dim, self.hidden_dim)
        self.Q2 = QFunction(latent_dim, action_dim, self.hidden_dim)

        self.apply(weight_init)

    def forward(self, state, action, detach_encoder=False):
        z_vector = self.encoder_net(state, detach=detach_encoder)
        q1 = self.Q1(z_vector, action)
        q2 = self.Q2(z_vector, action)
        return q1, q2

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.num_filters = 32
        self.latent_dim  = latent_dim

        self.fc_1 = nn.Linear(self.latent_dim, 39200)

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=3, kernel_size=3, stride=2, output_padding=1),
            nn.Sigmoid(),  # original paper no use activation function here. I added it and helps
        )

        self.apply(weight_init)

    def forward(self, x):
        x = torch.relu(self.fc_1(x))
        x = x.view(-1, 32, 35, 35)
        x = self.deconvs(x)
        return x
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.num_layers  = 4
        self.num_filters = 32
        self.latent_dim = latent_dim

        self.cov_net = nn.ModuleList([nn.Conv2d(3, self.num_filters, 3, stride=2)])
        for i in range(self.num_layers - 1):
            self.cov_net.append(nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1))

        self.fc = nn.Linear(39200, self.latent_dim)
        self.ln = nn.LayerNorm(self.latent_dim)

    def forward_conv(self, x):
        conv = torch.relu(self.cov_net[0](x))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.cov_net[i](conv))
        h = torch.flatten(conv, start_dim=1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        out = torch.tanh(h_norm)
        return out

    def copy_conv_weights_from(self, model_source):
        for i in range(self.num_layers):
            tie_weights(src=model_source.cov_net[i], trg=self.cov_net[i])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Networks for normal with a vector state space  TD3
class Actor_Normal(nn.Module):
    def __init__(self, obs_dim, actions_dim, max_value):
        super(Actor_Normal, self).__init__()

        self.input_size  = obs_dim
        self.actions_dim = actions_dim
        self.hidden_size = [1024, 1024]
        self.max_value   = max_value

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
        return x * self.max_value

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

        self.input_size  = obs_dim
        self.actions_dim = action_dim
        self.hidden_dim  = [1024, 1024]

        self.Q1 = QFunction_Normal(self.input_size, action_dim, self.hidden_dim)
        self.Q2 = QFunction_Normal(self.input_size, action_dim, self.hidden_dim)

    def forward(self, state, action):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2


