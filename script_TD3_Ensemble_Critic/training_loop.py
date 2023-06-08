import gym
import torch
import time
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from special_TD3 import Special_Agent
from cares_reinforcement_learning.util import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp


def train(env, special_agent):

    max_steps_training    = 100_000
    max_steps_exploration = 1_000

    batch_size = 128
    seed       = 1  # 571 seed gives no that great results
    G          = 1

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]
    action_size      = env.action_space.shape[0]

    # -----------------------------------#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.action_space.seed(seed)
    # -----------------------------------#

    memory = MemoryBuffer()

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0
    historical_reward = {"step": [], "episode_reward": []}

    start_time = time.time()
    state, _   = env.reset()

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1
        if total_step_counter < max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action_env = env.action_space.sample()  # action range the env uses [e.g. -2 , 2 for pendulum]
            action     = hlp.normalize(action_env, max_action_value, min_action_value)  # algorithm range [-1, 1]
        else:
            action     = special_agent.get_action_from_policy(state)
            action_env = hlp.denormalize(action, max_action_value, min_action_value)

        next_state, reward_extrinsic, done, truncated, info = env.step(action_env)

        memory.add(state=state, action=action, reward=reward_extrinsic, next_state=next_state, done=done)

        state = next_state
        episode_reward += reward_extrinsic

        if total_step_counter >= max_steps_exploration:
            num_updates = max_steps_exploration if total_step_counter == max_steps_exploration else G
            for _ in range(num_updates):
                experience = memory.sample(batch_size)
                special_agent.train_policy(experience)

        if done or truncated:
            episode_duration = time.time() - start_time
            start_time = time.time()

            logging.info(
                f"Total T:{total_step_counter + 1} | Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f} | Duration= {episode_duration:.2f} Seg")
            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state, _       = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1


def main():
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_gym_name = "Pendulum-v1"  # BipedalWalker-v3, Pendulum-v1, HalfCheetah-v4"

    env = gym.make(env_gym_name, render_mode=None)

    obs_size    = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    special_agent = Special_Agent(
        input_dim=obs_size,
        action_num=action_size,
        device=device,
    )


    train(env, special_agent)


if __name__ == '__main__':
    main()
