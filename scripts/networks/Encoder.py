import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class Encoder(nn.Module):
    def __init__(self, latent_dim, k=3):
        super(Encoder, self).__init__()
        self.num_layers  = 4
        self.num_filters = 32
        self.latent_dim = latent_dim

        self.cov_net = nn.ModuleList([nn.Conv2d(k, self.num_filters, 3, stride=2)])
        for i in range(self.num_layers - 1):
            self.cov_net.append(nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1))

        self.fc = nn.Linear(39200, self.latent_dim)
        self.ln = nn.LayerNorm(self.latent_dim)

        #self.encoder_optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) # todo no estoy seguro de esto aqui


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

    def copy_all_weights_from(self, model_source):
        for i in range(self.num_layers):
            tie_weights(src=model_source.cov_net[i], trg=self.cov_net[i])
        tie_weights(src=model_source.fc, trg=self.fc)
        #tie_weights(src=model_source.ln, trg=self.ln)


