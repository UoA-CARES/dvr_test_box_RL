
import os
import torch
import torch.nn as nn

import numpy as np
from networks import Transition_Network
from networks import Transition_Network_Discrete

import torch.nn.functional as F


class Deep_Ensemble:
    def __init__(self, input_dim, output_dim, device, ensemble_size=3):
        self.input_dim     = input_dim
        self.output_dim    = output_dim
        self.ensemble_size = ensemble_size
        self.device = device

        self.ensemble_network = nn.ModuleList()  # ModuleList have not a forward method

        networks = [Transition_Network(self.input_dim, self.output_dim) for _ in range(self.ensemble_size)]
        #networks = [Transition_Network_Discrete(self.input_dim, self.output_dim) for _ in range(self.ensemble_size)]

        self.ensemble_network.extend(networks)
        self.ensemble_network.to(self.device)

        learning_rate = 1e-6
        weight_decay  = 1e-3

        self.optimizers = [torch.optim.Adam(self.ensemble_network[i].parameters(), lr=learning_rate, weight_decay=weight_decay) for i in range(self.ensemble_size)]


    def train_transition_model_discrete(self, experiences):
        states, actions, _, next_states, _ = experiences

        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)

        for network, optimizer in zip(self.ensemble_network, self.optimizers):
            network.train()

            prediction_vector = network(states, actions)  # these values are mean and variance

            # Calculate Loss
            loss = F.mse_loss(prediction_vector, next_states)

            # Update weights and bias
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def train_transition_model(self, experiences):
        states, actions, _, next_states, _ = experiences

        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)

        for network, optimizer in zip(self.ensemble_network, self.optimizers):
            network.train()

            # Get the Prediction
            prediction_distribution = network(states, actions)  # these values are mean and variance
            loss_neg_log_likelihood = - prediction_distribution.log_prob(next_states)
            loss_neg_log_likelihood = torch.mean(loss_neg_log_likelihood)

            # Update weights and bias
            optimizer.zero_grad()
            loss_neg_log_likelihood.backward()
            optimizer.step()


    def get_prediction_from_model_discrete(self, state, action):

        state_tensor = torch.FloatTensor(state).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.FloatTensor(action).to(self.device)
        action_tensor = action_tensor.unsqueeze(0)

        predict_vector_set =  []
        for network in self.ensemble_network:
            network.eval()
            predicted_vector = network(state_tensor, action_tensor)

            predict_vector_set.append(predicted_vector.detach().cpu().numpy())

        ensemble_vector = np.concatenate(predict_vector_set, axis=0)
        avr_vector      = np.mean(ensemble_vector, axis=0)

        print("Prediction", avr_vector)


    def get_prediction_from_model(self, state, action):

        state_tensor = torch.FloatTensor(state).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.FloatTensor(action).to(self.device)
        action_tensor = action_tensor.unsqueeze(0)

        predict_mean_set, predict_std_set = [], []
        for network in self.ensemble_network:
            network.eval()
            predicted_distribution = network(state_tensor, action_tensor)

            mean = predicted_distribution.mean
            std  = predicted_distribution.stddev

            predict_mean_set.append(mean.detach().cpu().numpy())
            predict_std_set.append(std.detach().cpu().numpy())

        ensemble_means = np.concatenate(predict_mean_set, axis=0)
        ensemble_stds  = np.concatenate(predict_std_set, axis=0)

        avr_mean = np.mean(ensemble_means, axis=0)
        avr_std  = np.mean(ensemble_stds, axis=0)

        print(avr_mean, avr_std)

        return avr_mean, avr_std


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
