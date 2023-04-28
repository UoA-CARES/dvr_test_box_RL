
import os
import torch
import torch.nn as nn

import numpy as np
from networks import Transition_Network


class Deep_Ensemble:
    def __init__(self, input_dim, output_dim, device, ensemble_size=3):
        self.input_dim     = input_dim
        self.output_dim    = output_dim
        self.ensemble_size = ensemble_size
        self.device = device

        self.ensemble_network = nn.ModuleList()  # ModuleList have not a forward method

        networks = [Transition_Network(self.input_dim, self.output_dim) for _ in range(self.ensemble_size)]
        self.ensemble_network.extend(networks)
        self.ensemble_network.to(self.device)

        learning_rate = 1e-3
        weight_decay  = 1e-3

        #self.optimizers = [torch.optim.Adam(self.ensemble_network[i].parameters(), lr=learning_rate, weight_decay=weight_decay) for i in range(self.ensemble_size)]
        self.optimizers = [torch.optim.Adam(self.ensemble_network[i].parameters(), lr=learning_rate) for i in range(self.ensemble_size)]

    def train_transition_model(self, experiences):
        states, actions, _, next_states, _ = experiences

        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)

        for network, optimizer in zip(self.ensemble_network, self.optimizers):
            network.train()

            # Get the Prediction
            prediction = network(states, actions)  # these values are mean and variance

            # Calculate Loss
            prediction_loss_function = nn.GaussianNLLLoss(full=False, reduction='mean', eps=1e-6)
            loss = prediction_loss_function(prediction[0], next_states, prediction[1]) # input, target, variance

            # Update weights and bias
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def get_prediction_from_model(self, state, action):
        pass


    def evaluate_transition_model(self, experiences):
        states, actions, _, _, _ = experiences # need to pass a single value here, maybe

        predict_mean_set, predict_var_set = [], []
        for network in self.ensemble_network:
            network.eval()
            predict_mean, predict_variance = network(states, actions)

            predict_mean_set.append(predict_mean.detach().cpu().numpy())
            predict_var_set.append(predict_variance.detach().cpu().numpy())

        ensemble_means = np.concatenate(predict_mean_set, axis=0)
        ensemble_vars  = np.concatenate(predict_var_set, axis=0)

        ensemble_means = np.array(ensemble_means)
        ensemble_vars  = np.array(ensemble_vars)  # do i need this?

        avr_mean = np.mean(ensemble_means, axis=0)
        avr_var  = np.mean(ensemble_vars, axis=0)


    def save_model(self):
        dir_exists = os.path.exists("models")
        filename = "transition"

        if not dir_exists:
            os.makedirs("models")

        torch.save(self.ensemble_network.state_dict(), f'models/{filename}_ensemble_model.pht')
        print("models has been saved...")

    def load_model(self):
        filename = "transition"
        self.ensemble_network.load_state_dict(torch.load(f'models/{filename}_ensemble_model.pht'))
        print("models has been loaded...")
