"""


"""
import torch
import random
import numpy as np
from collections import deque


class MemoryClass:
    def __init__(self, replay_max_size, device):

        self.replay_max_size = replay_max_size
        self.memory_buffer   = deque(maxlen=replay_max_size)
        self.device          = device

    def save_experience_to_buffer(self, state, action, reward, next_state, done, goal):
        experience = (state, action, reward, next_state, done, goal)
        self.memory_buffer.append(experience)

    def sample_experiences_from_buffer(self, sample_size):
        state_batch      = []
        action_batch     = []
        reward_batch     = []
        next_state_batch = []
        done_batch       = []
        goal_batch       = []

        batch = random.sample(self.memory_buffer, sample_size)
        for experience in batch:
            state, action, reward, next_state, done, target = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            goal_batch.append(target)

        state_batch      = np.array(state_batch)
        action_batch     = np.array(action_batch)
        reward_batch     = np.array(reward_batch).reshape(-1, 1)
        done_batch       = np.array(done_batch).reshape(-1, 1)
        next_state_batch = np.array(next_state_batch)
        goal_batch       = np.array(goal_batch).reshape(-1, 1)

        state_batch_tensor      = torch.FloatTensor(state_batch).to(self.device)
        action_batch_tensor     = torch.FloatTensor(action_batch).to(self.device)
        reward_batch_tensor     = torch.FloatTensor(reward_batch).to(self.device)
        done_batch_tensor       = torch.FloatTensor(done_batch).to(self.device)
        next_batch_state_tensor = torch.FloatTensor(next_state_batch).to(self.device)
        goal_batch_tensor     = torch.FloatTensor(goal_batch).to(self.device)

        return state_batch_tensor, action_batch_tensor, reward_batch_tensor, next_batch_state_tensor, done_batch_tensor, goal_batch_tensor



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
        valve_angle_prev  = self.env.get_valve_angle() # get the value previous take the action
        #print("valve angle previous action:", valve_angle_prev)
        self.env.step_action(action)
        original_img = self.env.vision_config.get_camera_image()
        obs          = self.env.vision_config.pre_pro_image(original_img)

        valve_angle_aft  = self.env.get_valve_angle()
        #print("valve angle after action:", valve_angle_aft)

        ext_reward, done, distance = self.env.calculate_extrinsic_reward(goal_angle, valve_angle_prev, valve_angle_aft)

        self.frames_stacked.append(obs)
        stacked_images = np.array(list(self.frames_stacked))
        return stacked_images, ext_reward, done, distance, original_img, valve_angle_aft
