
import cv2
import gym
import torch
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from cares_reinforcement_learning.util import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp

from Algorithm import Algorithm
from FrameStack import FrameStack



def preprocessing_image(image_array):
    resized    = cv2.resize(image_array, (84, 84), interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image_out  = np.expand_dims(norm_image, axis=0) # to make it (1, channel, w, h) because I am using a single img here
    return image_out


def train(env, model_policy):

    max_steps_training    = 50_000
    max_steps_exploration = 1_000

    batch_size = 32
    seed       = 571
    G          = 10
    k          = 3

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.action_space.seed(seed)

    memory       = MemoryBuffer()
    frames_stack = FrameStack(env, k, seed)

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state = frames_stack.reset()  # for 3 images
    # _, _  = env.reset(seed=seed)
    # state = env.render()
    # state = preprocessing_image(state)  # for single image

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action_env = env.action_space.sample()  # action range the env uses [e.g. -2 , 2 for pendulum]
            action     = hlp.normalize(action_env, max_action_value, min_action_value)  # algorithm range [-1, 1]
        else:
            action     = model_policy.get_action_from_policy(state)
            action_env = hlp.denormalize(action, max_action_value, min_action_value)

        # _, reward_extrinsic, done, truncated, info = env.step(action_env)
        # next_state = env.render()
        # next_state = preprocessing_image(next_state)

        next_state, reward_extrinsic, done, truncated, info = frames_stack.step(action_env)

        # intrinsic rewards
        # surprise_rate = model_policy.get_surprise_rate(state, action)
        # novelty_rate  = model_policy.get_novelty_rate(state)

        rew_surprise = 0
        rew_novelty  = 0
        # dopamine   = None  # to include later if reach the goal

        # Total Reward
        reward = reward_extrinsic + rew_surprise + rew_novelty

        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        state = next_state

        episode_reward += reward

        if total_step_counter >= max_steps_exploration:
            for _ in range(G):
                experiences = memory.sample(batch_size)
                model_policy.train_policy(experiences)
            model_policy.train_predictive_model(experiences)

        if done or truncated:
            logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            # Reset environment
            # _, _ = env.reset(seed=seed)
            # state = env.render()
            # state = preprocessing_image(state)
            state = frames_stack.reset()

            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env_gym_name = "Pendulum-v1" # BipedalWalker-v3, Pendulum-v1, HalfCheetah-v4"
    env          = gym.make(env_gym_name, render_mode="rgb_array")

    action_size = env.action_space.shape[0]
    latent_size = 50

    model_policy = Algorithm(
        latent_size=latent_size,
        action_num=action_size,
        device=device)

    train(env, model_policy)


if __name__ == '__main__':
    main()
