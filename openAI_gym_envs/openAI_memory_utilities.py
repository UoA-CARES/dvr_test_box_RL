import cv2
import torch
import random
import numpy as np
from collections import deque

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Memory:
    def __init__(self, replay_max_size, device):
        self.device          = device
        self.replay_max_size = replay_max_size
        self.memory_buffer   = deque(maxlen=replay_max_size)

    def save_experience_to_buffer(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory_buffer.append(experience)

    def sample_experiences_from_buffer(self, sample_size):
        state_batch       = []
        action_batch      = []
        reward_batch      = []
        next_state_batch  = []
        done_batch        = []

        batch = random.sample(self.memory_buffer, sample_size)
        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        state_batch  = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch).reshape(-1, 1)
        done_batch = np.array(done_batch).reshape(-1, 1)
        next_state_batch = np.array(next_state_batch)

        state_batch_tensor  = torch.FloatTensor(state_batch).to(self.device)
        action_batch_tensor = torch.FloatTensor(action_batch).to(self.device)
        reward_batch_tensor = torch.FloatTensor(reward_batch).to(self.device)
        done_batch_tensor   = torch.FloatTensor(done_batch).to(self.device)
        next_batch_state_tensor = torch.FloatTensor(next_state_batch).to(self.device)

        return state_batch_tensor, action_batch_tensor, reward_batch_tensor, next_batch_state_tensor, done_batch_tensor
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class FrameStack:
    def __init__(self, k, env):
        self.env = env
        self.k   = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

    def reset(self):
        self.env.reset()
        obs = self.env.render(mode='rgb_array')
        obs = self.preprocessing_image(obs)
        for _ in range(self.k):
            self.frames_stacked.append(obs)
        # stacked_vector = np.concatenate(list(self.frames_stacked), axis=0)
        stacked_vector = np.array(list(self.frames_stacked))
        return stacked_vector

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self.env.render(mode='rgb_array')
        obs = self.preprocessing_image(obs)
        self.frames_stacked.append(obs)
        stacked_vector = np.array(list(self.frames_stacked))
        # stacked_vector = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_vector, reward, done, info

    def preprocessing_image(self, image_array):
        resized    = cv2.resize(image_array, (84, 84), interpolation=cv2.INTER_AREA)
        gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        norm_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_image