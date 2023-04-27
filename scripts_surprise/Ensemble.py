
import torch
import torch.nn as nn

from networks import Transition_Network


# input_dim  = 32
# output_dim = 30
# ensemble_size = 5
#
# ensemble_network = nn.ModuleList()  # ModuleList have not a forward method
# networks = [Transition_Network(input_dim, output_dim) for _ in range(ensemble_size)]
# ensemble_network.extend(networks)


class Deep_Ensemble:
    def __init__(self, input_dim, output_dim, ensemble_size=3):
        self.input_dim     = input_dim
        self.output_dim    = output_dim
        self.ensemble_size = ensemble_size

        self.ensemble_network = nn.ModuleList()  # ModuleList have not a forward method

        networks = [Transition_Network(self.input_dim, self.output_dim) for _ in range(self.ensemble_size)]
        self.ensemble_network.extend(networks)

        learning_rate = 1e-3
        optimizers    = [torch.optim.Adam(self.ensemble_network[i].parameters(), lr=learning_rate, weight_decay=weight_decay) for i in range(self.ensemble_size)]


Deep_Ensemble(input_dim=23, output_dim=25)


