
from collections import deque
import random

class MemoryBuffer:
    def __init__(self, max_capacity=int(1e6)):

        self.buffer_env    = deque([], maxlen=max_capacity)
        self.buffer_model  = deque([], maxlen=max_capacity)
        self.buffer_policy = deque([], maxlen=max_capacity)

    def add_env(self,  *experience):
        self.buffer_env.append(experience)

    def add_model(self,  *experience):
        # need the for loop because each experiences is a batch e.g action is a [batch_size, 4]
        # I take each element of the batch and store in the buffer
        for single_state, single_action, single_reward, single_next_state, single_done in zip(*experience):
            self.buffer_model.append((single_state, single_action, single_reward, single_next_state, single_done))

    def sample_env(self, sample_size):
        experience_batch = random.sample(self.buffer_env, sample_size)
        states, actions, rewards, next_states, dones = zip(*experience_batch)
        return states, actions, rewards, next_states, dones

    def sample_model(self, sample_size):
        experience_batch = random.sample(self.buffer_model, sample_size)
        states, actions, rewards, next_states, dones = zip(*experience_batch)
        return states, actions, rewards, next_states, dones

