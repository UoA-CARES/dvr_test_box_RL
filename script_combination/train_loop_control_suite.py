
import cv2
import logging
logging.basicConfig(level=logging.INFO)
from cares_reinforcement_learning.util import helpers as hlp

import torch
import numpy as np
from dm_control import suite

from Algorithm import Algorithm
from FrameStack_DMCS import FrameStack

from cares_reinforcement_learning.util import MemoryBuffer


def train(env, model_policy):
    max_steps_training    = 100_000
    max_steps_exploration = 1_000

    batch_size = 64
    seed = 571
    G = 10
    k = 3

    action_spec      = env.action_spec()
    action_size      = action_spec.shape[0]
    max_action_value = action_spec.maximum[0]
    min_action_value = action_spec.minimum[0]

    torch.manual_seed(seed)
    np.random.seed(seed)
    # env.action_space.seed(seed)

    memory       = MemoryBuffer()
    frames_stack = FrameStack(env, k, seed)

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state = frames_stack.reset()  # for 3 images

    historical_reward = {"step": [], "episode_reward": []}

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action = np.random.uniform(min_action_value, max_action_value, size=action_size)
        else:
            action = model_policy.get_action_from_policy(state)  # no normalization need for action already [-1, 1]

        next_state, reward_extrinsic, done = frames_stack.step(action)

        if total_step_counter % 100 == 0:
            render_flag = True
        else:
            render_flag = False

        surprise_rate, novelty_rate = model_policy.get_intrinsic_values(state, action, render_flag)

        # intrinsic rewards
        # # dopamine   = None  # to include later if reach the goal
        reward_surprise = (surprise_rate)
        reward_novelty  = (1 - novelty_rate)
        logging.info(f"Surprise Rate = {reward_surprise},  Novelty Rate = {reward_novelty}, Normal Reward = {reward_extrinsic}, {total_step_counter}")

        # Total Reward
        total_reward = reward_extrinsic + reward_surprise + reward_novelty

        memory.add(state=state, action=action, reward=total_reward, next_state=next_state, done=done)
        state = next_state

        episode_reward += reward_extrinsic  # just for plotting purposes use this reward as it is

        if total_step_counter >= max_steps_exploration:
            for _ in range(G):
                experiences = memory.sample(batch_size)
                model_policy.train_policy(experiences)

            model_policy.train_predictive_model(experiences)

        if done:
            logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            state = frames_stack.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    hlp.plot_reward_curve(historical_reward)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    domain_name = "cartpole"
    task_name   = "balance"
    seed        = 571
    env         = suite.load(domain_name, task_name, task_kwargs={'random': seed})

    action_spec = env.action_spec()
    action_size = action_spec.shape[0]

    latent_size = 50

    model_policy = Algorithm(
        latent_size=latent_size,
        action_num=action_size,
        device=device,
        k=3)


    train(env, model_policy)



if __name__ == '__main__':
    main()
