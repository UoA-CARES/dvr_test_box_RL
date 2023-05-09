import cv2
import numpy as np
from collections import deque


class FrameStack:
    def __init__(self, env, k=3, seed=123):
        self.env  = env
        self.seed = seed
        self.k    = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

    def reset(self):
        _ = self.env.reset()
        frame = self.env.physics.render(480, 600, camera_id=0)
        obs   = self.preprocessing_image(frame)
        for _ in range(self.k):
            self.frames_stacked.append(obs)
        stacked_frames = np.array(self.frames_stacked)
        return stacked_frames

    def step(self, action):
        time_step    = self.env.step(action)
        reward, done = time_step.reward, time_step.last()
        frame = self.env.physics.render(480, 600, camera_id=0)
        obs = self.preprocessing_image(frame)
        self.frames_stacked.append(obs)
        stacked_frames = np.array(self.frames_stacked)
        return stacked_frames, reward, done

    def preprocessing_image(self, image_array):
        resized    = cv2.resize(image_array, (84, 84), interpolation=cv2.INTER_AREA)
        gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        norm_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_image
