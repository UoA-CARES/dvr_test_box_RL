import torch
import random
import numpy as np
from collections import deque



class MemoryClass:

    def __init__(self, replay_max_size, device):

        self.replay_max_size = replay_max_size
        self.replay_buffer   = deque(maxlen=replay_max_size)  
        self.device          = device


    def replay_buffer_add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.append(experience)


    def sample_experience(self, batch_size):
        state_batch      = []
        action_batch     = []
        reward_batch     = []
        next_state_batch = []
        done_batch       = []

        batch = random.sample(self.replay_buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        state_batch      = np.array(state_batch)
        action_batch     = np.array(action_batch)
        reward_batch     = np.array(reward_batch).reshape(-1, 1)
        done_batch       = np.array(done_batch).reshape(-1, 1)
        next_state_batch = np.array(next_state_batch)


        state_batch_tensor      = torch.FloatTensor(state_batch).to(self.device)
        action_batch_tensor     = torch.FloatTensor(action_batch).to(self.device)
        reward_batch_tensor     = torch.FloatTensor(reward_batch).to(self.device)
        done_batch_tensor       = torch.FloatTensor(done_batch).to(self.device)
        next_batch_state_tensor = torch.FloatTensor(next_state_batch).to(self.device)


        return state_batch_tensor, action_batch_tensor, reward_batch_tensor, next_batch_state_tensor, done_batch_tensor

    
    def __len__(self):
        return len(self.replay_buffer)