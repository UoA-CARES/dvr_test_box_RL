
import cv2
import gym
import numpy as np
from collections import deque


class CreateEnvironment:
    def __init__(self, env_name, k=3):

        self.env = gym.make(env_name)
        self.k   = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

        self.act_dim    = self.env.action_space.shape[0]
        self.max_action = self.env.action_space.high.max()

    def reset(self):
        self.env.reset()
        obs = self.env.render(mode='rgb_array')
        obs = self.preprocessing_image(obs)
        for _ in range(self.k):
            self.frames_stacked.append(obs)
        stacked_vector = np.array(list(self.frames_stacked))
        return stacked_vector

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self.env.render(mode='rgb_array')
        obs = self.preprocessing_image(obs)
        self.frames_stacked.append(obs)
        stacked_vector = np.array(list(self.frames_stacked))
        return stacked_vector, reward, done, info

    def preprocessing_image(self, image_array):
        #img_cropped = image_array[100:400, 100:400]
        resized = cv2.resize(image_array, (84, 84), interpolation=cv2.INTER_AREA)
        gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        norm_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_image

    def seed(self, seed):
        self.env.seed(seed)
        self.env.action_space.seed(seed)

    def action_sample(self):
        action = self.env.action_space.sample()
        return action
