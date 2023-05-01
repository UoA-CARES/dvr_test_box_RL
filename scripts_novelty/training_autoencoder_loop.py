import cv2
import gym
import torch
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from Novelty import Deep_Novelty
from cares_reinforcement_learning.util import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp


def preprocessing_image(image_array):
    resized    = cv2.resize(image_array, (84, 84), interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image_out  = np.expand_dims(norm_image, axis=0) # to make it (1, channel, w, h) because i am using a single img here
    return image_out

def train(env, autoencoder_model):

    max_steps_training    = 10_000
    max_steps_exploration = 2_000
    batch_size = 32
    seed = 123

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.action_space.seed(seed)

    memory = MemoryBuffer()

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    _, _  = env.reset(seed=seed)
    state = env.render()
    state = preprocessing_image(state)

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action_env = env.action_space.sample()
            action     = hlp.normalize(action_env, max_action_value, min_action_value)  # algorithm range [-1, 1]
        else:
            action_env = env.action_space.sample() # REMOVE THIS line to the actual agent/policy
            action = hlp.normalize(action_env, max_action_value, min_action_value)  # REMOVE THIS line to the actual agent/policy

        _, reward, done, truncated, info = env.step(action_env)
        next_state = env.render()
        next_state = preprocessing_image(next_state)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        state = next_state

        if total_step_counter >= max_steps_exploration:
            experiences = memory.sample(batch_size)
            autoencoder_model.train_autoencoder_model(experiences)

        if done or truncated:
            logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            # Reset environment
            _, _  = env.reset()
            state = env.render()
            state = preprocessing_image(state)
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

    autoencoder_model.save_model()


def main():
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_gym_name = "Pendulum-v1"  # BipedalWalker-v3, Pendulum-v1, HalfCheetah-v4"

    env = gym.make(env_gym_name, render_mode="rgb_array")  # for AE needs to be rgb_array in render_mode

    latent_dim = 50

    autoencoder_model = Deep_Novelty(
        latent_dim=latent_dim,
        device=device,
        k=1
    )

    train(env, autoencoder_model)


if __name__ == '__main__':
    main()
