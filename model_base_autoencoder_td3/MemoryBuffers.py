
from collections import deque
import random

class MemoryBuffer:
    def __init__(self, max_capacity=int(1e6)):

        self.buffer_env    = deque([], maxlen=max_capacity)
        self.buffer_model  = deque([], maxlen=50_000)

        self.buffer_episodes = deque([], maxlen=max_capacity)

    def add_env(self,  *experience):
        self.buffer_env.append(experience)

    def extend_env(self, experience):
        self.buffer_env.extend(experience)

    def add_model(self,  *experience):
        # need the for loop because each experiences is a batch e.g action is a [batch_size, 4]
        # I take each element of the batch and store in the buffer
        for single_state, single_action, single_reward, single_next_state, single_done in zip(*experience):
            self.buffer_model.append((single_state, single_action, single_reward, single_next_state, single_done))

    def add_episode_buffer(self, episode):
        self.buffer_episodes.append(episode)



    def sample_env(self, sample_size):
        experience_batch = random.sample(self.buffer_env, sample_size)
        states, actions, rewards, next_states, dones = zip(*experience_batch)
        return states, actions, rewards, next_states, dones

    def sample_model(self, sample_size):
        experience_batch = random.sample(self.buffer_model, sample_size)
        states, actions, rewards, next_states, dones = zip(*experience_batch)
        return states, actions, rewards, next_states, dones

    def sample_episode(self, sample_size):
        # this is a batch of episodes where each episode include several experiences of (s,a,s,r, d)
        episode_batch = random.sample(self.buffer_episodes, sample_size)

        state_batch      = []
        action_batch     = []
        reward_batch     = []
        next_state_batch = []
        done_batch       = []

        # this will extract the experiences of each episode
        for episode in episode_batch:
            states, actions, rewards, next_states, dones = zip(*episode)

            state_batch.append(states)
            action_batch.append(actions)
            reward_batch.append(rewards)
            next_state_batch.append(next_states)
            done_batch.append(dones)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

