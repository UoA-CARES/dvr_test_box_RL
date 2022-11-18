import torch
import numpy as np
from collections import deque


class FrameStack:
    def __init__(self, k, env):
        self.env = env
        self.k   = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

    def reset(self):
        self.env.reset()
        obs = self.env.vision_config.get_camera_image()
        obs = self.env.vision_config.pre_pro_image(obs)
        for _ in range(self.k):
            self.frames_stacked.append(obs)
        stacked_vector = np.array(list(self.frames_stacked))
        return stacked_vector

    def step(self, action, goal_angle):
        self.env.step_action(action)
        original_img = self.env.vision_config.get_camera_image()
        obs          = self.env.vision_config.pre_pro_image(original_img)
        valve_angle  = self.env.get_valve_angle()
        ext_reward, done, distance = self.env.calculate_extrinsic_reward(goal_angle, valve_angle)
        self.frames_stacked.append(obs)
        stacked_images = np.array(list(self.frames_stacked))
        return stacked_images, ext_reward, done, distance, original_img, valve_angle





