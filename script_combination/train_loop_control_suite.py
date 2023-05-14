
import logging
logging.basicConfig(level=logging.INFO)

import torch
import numpy as np
from dm_control import suite

from cares_reinforcement_learning.util import MemoryBuffer

from Algorithm import Algorithm
from FrameStack_DMCS import FrameStack

import pandas as pd
import matplotlib.pyplot as plt

def plot_reward_curve(data_reward, filename):
    data = pd.DataFrame.from_dict(data_reward)
    data.plot(x='step', y='episode_reward', title="Reward Curve")
    plt.title(filename)
    plt.show()


def train(env, model_policy, file_name, intrinsic_on):

    max_steps_training    = 100_000
    max_steps_exploration = 1_000

    batch_size = 64
    seed       = 571
    G          = 5
    k          = 3

    action_spec      = env.action_spec()
    action_size      = action_spec.shape[0]
    max_action_value = action_spec.maximum[0]
    min_action_value = action_spec.minimum[0]

    torch.manual_seed(seed)
    np.random.seed(seed)

    memory       = MemoryBuffer()
    frames_stack = FrameStack(env, k, seed)

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state = frames_stack.reset()  # for 3 images with color

    historical_reward = {"step": [], "episode_reward": []}

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action = np.random.uniform(min_action_value, max_action_value, size=action_size)
        else:
            action = model_policy.get_action_from_policy(state)  # no normalization need for action, already [-1, 1]

        next_state, reward_extrinsic, done = frames_stack.step(action)

        if total_step_counter % 200 == 0:
            render_flag = True
        else:
            render_flag = False

        # intrinsic rewards
        a = 5
        b = 10

        surprise_rate, novelty_rate = model_policy.get_intrinsic_values(state, action, next_state, render_flag)
        reward_surprise = surprise_rate * a
        reward_novelty  = novelty_rate  * b



        # Total Reward
        if intrinsic_on:
            if total_step_counter >= max_steps_exploration:
                logging.info(f"Surprise Rate = {reward_surprise},  Novelty Rate = {reward_novelty}, Normal Reward = {reward_extrinsic}, {total_step_counter}")
                total_reward = reward_extrinsic +  reward_surprise +  reward_novelty
            else:
                total_reward = reward_extrinsic
        else:
            total_reward = reward_extrinsic


        memory.add(state=state, action=action, reward=total_reward, next_state=next_state, done=done)
        state = next_state

        episode_reward += reward_extrinsic  # just for plotting purposes use this reward as it is

        if total_step_counter >= max_steps_exploration:
            for _ in range(G):
                experiences = memory.sample(batch_size)
                model_policy.train_policy(experiences)

            if intrinsic_on:
                model_policy.train_predictive_model(experiences)

        if done:
            logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            state = frames_stack.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    model_policy.save_models(filename=file_name)
    plot_reward_curve(historical_reward, filename=file_name)


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Domain = cartpole, cheetah, reacher
    # task   = balance , run,     easy

    domain_name = "reacher"
    task_name   = "easy"

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

    intrinsic_on = True

    file_name = domain_name + "_" + task_name + "_" + "TD3_AE_Detach_True" + "_Intrinsic_" + str(intrinsic_on)
    train(env, model_policy, file_name, intrinsic_on)


if __name__ == '__main__':
    main()
