
from collections import deque
import random

class MemoryBuffer:
    def __init__(self, max_capacity=int(1e6)):

        self.buffer_env    = deque([], maxlen=max_capacity)
        self.buffer_model  = deque([], maxlen=max_capacity)
        self.buffer_policy = deque([], maxlen=max_capacity)

    def add_env(self,  *experience):
        self.buffer_env.append(experience)

    def sample_env(self, sample_size):
        experience_batch = random.sample(self.buffer_env, sample_size)
        states, actions, rewards, next_states, dones = zip(*experience_batch)
        return states, actions, rewards, next_states, dones

