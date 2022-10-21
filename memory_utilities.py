
from collections import deque
import random


class MemoryClass:

    def __init__(self, replay_max_size):

        self.replay_max_size = replay_max_size

        self.memory_buffer               = deque(maxlen=replay_max_size)
        self.memory_buffer_frames        = deque(maxlen=replay_max_size)
        self.memory_buffer_experiences   = deque(maxlen=replay_max_size)

    def save_frame_experience_buffer(self, frame_state):
        experience = frame_state
        self.memory_buffer_frames.append(experience)

    def save_vector_experience_buffer(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory_buffer_experiences.append(experience)

    def save_full_experience_buffer(self, state, action, reward, next_state, done, valve_angle, next_valve_angle, target_angle):
        # here the state and new state are images (3 channel),
        # Reward is normal
        # Angle in degrees
        experience = (state, action, reward, next_state, done, valve_angle, next_valve_angle, target_angle)
        self.memory_buffer.append(experience)

    def sample_frames_experiences(self, sample_size):
        batch_imgs = random.sample(self.memory_buffer_frames, sample_size)
        return batch_imgs

    def sample_vector_experiences(self, sample_size):
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

    def sample_full_experiences(self, sample_size):
        state_batch       = []
        action_batch      = []
        reward_batch      = []
        next_state_batch  = []
        done_batch        = []
        valve_angle_batch        = []
        valve_next_angle_batch   = []
        valve_target_angle_batch = []

        batch = random.sample(self.memory_buffer, sample_size)
        for experience in batch:
            state, action, reward, next_state, done, valve, next_valve, target = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            valve_angle_batch.append(valve)
            valve_next_angle_batch.append(next_valve)
            valve_target_angle_batch.append(target)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, valve_angle_batch, valve_next_angle_batch, valve_target_angle_batch
