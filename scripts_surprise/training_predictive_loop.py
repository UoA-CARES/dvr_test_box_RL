"""
Just
"""

import gym
import torch
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from Ensemble import Deep_Ensemble
from cares_reinforcement_learning.util import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp



def train(env, prediction_model):

    max_steps_training    = 10_000
    max_steps_exploration = 1_000
    batch_size = 4
    seed = 452

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.action_space.seed(seed)

    memory = MemoryBuffer()

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state, _ = env.reset(seed=seed)

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action_env = env.action_space.sample()
            action     = hlp.normalize(action_env, max_action_value, min_action_value)  # algorithm range [-1, 1]

        else:
            action_env = env.action_space.sample() # REMOVE THIS line to the actual agent/policy
            action = hlp.normalize(action_env, max_action_value, min_action_value)  # REMOVE THIS line to the actual agent/policy

        next_state, reward, done, truncated, info = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        state = next_state

        if total_step_counter >= max_steps_exploration:
            experiences = memory.sample(batch_size)
            prediction_model.train_transition_model(experiences)

        if done or truncated:
            logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

    prediction_model.save_model()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_gym_name = "Pendulum-v1"  # BipedalWalker-v3, Pendulum-v1, HalfCheetah-v4"

    env = gym.make(env_gym_name, render_mode=None)  # for AE needs to be rgb_array in render_mode

    obs_size    = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    prediction_model = Deep_Ensemble(
        input_dim=obs_size+action_size,
        output_dim=obs_size,
        device=device,
        ensemble_size=5
    )

    train(env, prediction_model)


if __name__ == '__main__':
    main()