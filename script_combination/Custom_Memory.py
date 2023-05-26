
from scipy.spatial import distance
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class CustomMemoryBuffer:
    def __init__(self, action_size, max_capacity=int(1e6)):
        self.max_capacity = max_capacity

        obs_shape    = (9, 84, 84)
        action_shape = action_size
        latent_size  = 50

        self.states      = np.empty((max_capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.empty((max_capacity, *obs_shape), dtype=np.uint8)
        self.actions     = np.empty((max_capacity, action_shape), dtype=np.float32)
        self.rewards     = np.empty((max_capacity, 1), dtype=np.float32)
        self.dones       = np.empty((max_capacity, 1), dtype=np.float32)
        self.z_vectors   = np.empty((max_capacity, latent_size), dtype=np.float32)

        self.idx  = 0
        self.full = False

    def add(self, **experience):

        state      = experience["state"]
        action     = experience["action"]
        reward     = experience["reward"]
        next_state = experience["next_state"]
        done       = experience["done"]
        latent_z   = experience["latent_z"]

        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_state)
        np.copyto(self.dones[self.idx], done)
        np.copyto(self.z_vectors[self.idx], latent_z)

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



    def search_state(self, z_arrive):
        # search if the new z_arrive vector  already exist in memory (identically)
        # logging.info("----------")
        # new_idendical = z_arrive in self.z_vectors
        # logging.info(f" {new_idendical}, for identical searching")

        # search if the new z_arrive vector or a "very similar" one already exist in memory
        threshold_novelty = 0.05
        range_to_search   = (range(0, self.max_capacity) if self.full else range(0, self.idx))
        new = True
        for previous_z_idx in range_to_search:
            dist  = np.linalg.norm(z_arrive - self.z_vectors[previous_z_idx])
            if dist <= threshold_novelty:
                logging.info(f" State Representation found in memory, it is not new")
                new = False
                break
        if new:
            logging.info(f" State Representation No found in memory, it is new")
        logging.info("********************")

        return new
