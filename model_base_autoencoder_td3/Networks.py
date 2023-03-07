import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias   = src.bias


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
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=50, k=3):
        super(Encoder, self).__init__()

        self.k           = k
        self.num_layers  = 4
        self.num_filters = 32
        self.latent_dim  = latent_dim

        self.cov_net = nn.ModuleList([nn.Conv2d(self.k, self.num_filters, 3, stride=2)])

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
            # todo maybe it is necessary detach the fc and norm too
        h_fc   = self.fc(h)
        h_norm = self.ln(h_fc)
        out    = torch.tanh(h_norm)
        return out

    def copy_conv_weights_from(self, model_source):
        for i in range(self.num_layers):
            tie_weights(src=model_source.cov_net[i], trg=self.cov_net[i])

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, latent_dim=50):

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
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = torch.relu(self.fc_1(z))
        x = x.view(-1, 32, 35, 35)
        x = self.deconvs(x)
        return x

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

class WorldModel(nn.Module):
    def __init__(self, latent_dim=50, action_dim=1):
        super(WorldModel, self).__init__()

        self.encoder_net = Encoder(latent_dim)

        hidden_size = [1024, 1024]

        self.model_net = nn.Sequential(
            nn.Linear(latent_dim+action_dim, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], latent_dim),
        )

    def forward(self, state, action, detach_encoder=False):
        z_vector      = self.encoder_net(state, detach=detach_encoder)
        z_n_action    = torch.cat([z_vector, action], dim=1)
        z_vector_next = self.model_net(z_n_action)
        return z_vector_next



class RewardModel(nn.Module):
    def __init__(self, latent_dim=50, action_dim=1):
        super(RewardModel, self).__init__()

        self.encoder_net = Encoder(latent_dim)

        hidden_size = [1024, 1024]

        self.reward_net = nn.Sequential(
            nn.Linear(latent_dim+action_dim, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1)
        )

    def forward(self, state, action, detach_encoder=False):
        z_vector   = self.encoder_net(state, detach=detach_encoder)
        z_n_action = torch.cat([z_vector, action], dim=1)
        reward     = self.reward_net(z_n_action)
        return reward
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

class Actor_AE(nn.Module):
    def __init__(self, latent_dim, action_dim, max_value, hidden_size=None):
        super(Actor_AE, self).__init__()

        self.encoder_net = Encoder(latent_dim)
        self.max_value = max_value

        if hidden_size is None:
            hidden_size = [1024, 1024]
        hidden_size = hidden_size

        self.l1 = nn.Linear(latent_dim, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], action_dim)

        self.ln = nn.LayerNorm(action_dim)

        self.apply(weight_init)

    def forward(self, state, detach_encoder=False):

        z_vector = self.encoder_net(state, detach=detach_encoder)

        a = F.relu(self.l1(z_vector))
        a = F.relu(self.l2(a))
        pre_activation = self.l3(a)
        #pre_activation = self.ln(pre_activation) # linear normalization layer

        output = torch.tanh(pre_activation)
        return output  * self.max_value


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
class Critic_AE(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_size=None):
        super(Critic_AE, self).__init__()

        self.encoder_net = Encoder(latent_dim)

        if hidden_size is None:
            hidden_size = [1024, 1024]
        hidden_size = hidden_size

        #Q1 architecture
        self.l1 = nn.Linear(latent_dim+action_dim, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], 1)

        #Q2 architecture
        self.l12 = nn.Linear(latent_dim+action_dim, hidden_size[0])
        self.l22 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l32 = nn.Linear(hidden_size[1], 1)

        self.apply(weight_init)

    def forward(self, state, action, detach_encoder=False):

        z_vector   = self.encoder_net(state, detach=detach_encoder)
        obs_action = torch.cat([z_vector, action], dim=1)

        q1 = F.relu(self.l1(obs_action))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l12(obs_action))
        q2 = F.relu(self.l22(q2))
        q2 = self.l32(q2)

        return q1, q2
