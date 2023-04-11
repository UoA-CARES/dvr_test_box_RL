import cv2
import numpy as np
from collections import deque


class FrameStack:
    def __init__(self, env, k=3):
        self.env = env
        self.k   = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

    def reset(self):
        self.env.reset()
        obs = self.env.render()
        obs = self.preprocessing_image(obs)
        for _ in range(self.k):
            self.frames_stacked.append(obs)
        # stacked_vector = np.concatenate(list(self.frames_stacked), axis=0)
        stacked_vector = np.array(list(self.frames_stacked))
        return stacked_vector

    def step(self, action):
        _, reward, done, truncated, info = self.env.step(action)
        obs = self.env.render()
        obs = self.preprocessing_image(obs)
        self.frames_stacked.append(obs)
        stacked_vector = np.array(list(self.frames_stacked))
        # stacked_vector = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_vector, reward, done, truncated, info

    def preprocessing_image(self, image_array):
        resized    = cv2.resize(image_array, (84, 84), interpolation=cv2.INTER_AREA)
        gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        norm_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_image
