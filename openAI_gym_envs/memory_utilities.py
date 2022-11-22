
import torch
import random
from collections import deque


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class MemoryClass:
    def __init__(self, replay_max_size):
        print("Memory size", replay_max_size)

        self.replay_max_size           = replay_max_size
        self.memory_buffer_experiences = deque(maxlen=replay_max_size)

    def save_frame_vector_experience_buffer(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory_buffer_experiences.append(experience)

    def sample_frame_vector_experiences(self, sample_size):
        state_batch      = []
        action_batch     = []
        reward_batch     = []
        next_state_batch = []
        done_batch       = []

        batch = random.sample(self.memory_buffer_experiences, sample_size)

        for experience in batch:
            state, action, reward, next_state, done = experience

            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
