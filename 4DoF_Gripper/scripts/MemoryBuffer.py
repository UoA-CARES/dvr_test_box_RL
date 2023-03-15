
import random
from collections import deque


class MemoryBuffer:
    def __init__(self, max_capacity):
        self.buffer = deque([], maxlen=max_capacity)

    def add(self, *experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        experience_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experience_batch)
        return states, actions, rewards, next_states, dones
