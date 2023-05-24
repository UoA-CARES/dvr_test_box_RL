
import numpy as np

class CustomMemoryBuffer:
    def __init__(self, action_size, max_capacity=int(1e6), ):
        self.max_capacity = max_capacity

        obs_shape    = (9, 84, 84)
        action_shape = action_size

        self.states      = np.empty((max_capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.empty((max_capacity, *obs_shape), dtype=np.uint8)
        self.actions     = np.empty((max_capacity, action_shape), dtype=np.float32)
        self.rewards     = np.empty((max_capacity, 1), dtype=np.float32)
        self.dones       = np.empty((max_capacity, 1), dtype=np.float32)

        self.idx  = 0
        self.full = False

    def add(self, **experience):

        state      = experience["state"]
        action     = experience["action"]
        reward     = experience["reward"]
        next_state = experience["next_state"]
        done       = experience["done"]

        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_state)
        np.copyto(self.dones[self.idx], done)

        self.idx  = (self.idx + 1) % self.max_capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):

        idxs = np.random.randint(0, self.max_capacity if self.full else self.idx, size=batch_size)

        states      = self.states[idxs]
        rewards     = self.rewards[idxs]
        actions     = self.actions[idxs]
        next_states = self.next_states[idxs]
        dones       = self.dones[idxs]

        return states, actions, rewards, next_states, dones
