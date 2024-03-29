
import os
import cv2
import gym
import time
import torch
import random
from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO)

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp

from Algorithm import Algorithm
from FrameStack_3CH import FrameStack

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def save_evaluation_values(data_eval_reward, filename):
    data = pd.DataFrame.from_dict(data_eval_reward)
    data.to_csv(f"data_plots/{filename}_evaluation", index=False)


def plot_reward_curve(data_reward, filename):
    data = pd.DataFrame.from_dict(data_reward)
    data.to_csv(f"data_plots/{filename}", index=False)
    data.plot(x='step', y='episode_reward', title="Reward Curve")
    plt.title(filename)
    plt.savefig(f"plots/{filename}.png")
    plt.close()


def plot_reconstruction_img(original, reconstruction):
    input_img      = original[0]/255
    reconstruction = reconstruction[0]
    difference     = abs(input_img - reconstruction)

    plt.subplot(1, 3, 1)
    plt.title("Image Input")
    plt.imshow(input_img, vmin=0, vmax=1)
    plt.subplot(1, 3, 2)
    plt.title("Image Reconstruction")
    plt.imshow(reconstruction, vmin=0, vmax=1)
    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.imshow(difference, vmin=0, vmax=1)
    plt.pause(0.01)


def train(env, agent,  file_name, intrinsic_on, number_stack_frames):

    max_steps_training    = 500_000
    max_steps_exploration = 1_000

    batch_size = 128
    G          = 1
    k          = number_stack_frames

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    memory       = MemoryBuffer()
    frames_stack = FrameStack(env, k)

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0
    historical_reward = {"step": [], "episode_reward": []}
    historical_reward_evaluation = {"step": [], "avg_episode_reward": []}

    historical_reward_evaluation["step"].append(0)
    historical_reward_evaluation["avg_episode_reward"].append(0)

    start_time = time.time()
    state      = frames_stack.reset()  # for k images

    for total_step_counter in range(1, int(max_steps_training)+1):
        episode_timesteps += 1

        if total_step_counter <= max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action_env = env.action_space.sample()  # action range the env uses [e.g. -2 , 2 for pendulum]
            action     = hlp.normalize(action_env, max_action_value, min_action_value)  # algorithm range [-1, 1]
        else:
            action     = agent.select_action_from_policy(state)
            action_env = hlp.denormalize(action, max_action_value, min_action_value)

        next_state, reward_extrinsic, done, truncated, info = frames_stack.step(action_env)

        if intrinsic_on and total_step_counter > max_steps_exploration:
            a = 0.5
            b = 0.5
            surprise_rate, novelty_rate = agent.get_intrinsic_values(state, action, next_state)
            reward_surprise = surprise_rate * a
            reward_novelty  = novelty_rate  * b
            #logging.info(f"Surprise Rate = {reward_surprise},  Novelty Rate = {reward_novelty}, Normal Reward = {reward_extrinsic}, {total_step_counter}")
            total_reward = reward_extrinsic + reward_surprise + reward_novelty

        else:
            total_reward = reward_extrinsic

        memory.add(state=state, action=action, reward=total_reward, next_state=next_state, done=done)
        state = next_state

        episode_reward += reward_extrinsic  # just for plotting and evaluation purposes use the  reward as it is

        if total_step_counter > max_steps_exploration:
            #num_updates = max_steps_exploration if total_step_counter == max_steps_exploration+1 else G
            for i in range(G):
                experience = memory.sample(batch_size)
                agent.train_policy((
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    experience['next_state'],
                    experience['done'],
                ))
                if intrinsic_on:
                    agent.train_predictive_model((
                        experience['state'],
                        experience['action'],
                        experience['next_state'],
                    ))

        if done or truncated:
            episode_duration = time.time() - start_time
            start_time       = time.time()

            logging.info(f"Total T:{total_step_counter} | Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f} | Duration= {episode_duration:.2f} Sec")
            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state = frames_stack.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num       += 1

            if episode_num % 10 == 0:
                print("--------------------------------------------")
                plot_reward_curve(historical_reward, filename=file_name)
                evaluation_loop(env, agent, frames_stack, total_step_counter, max_action_value, min_action_value, file_name, historical_reward_evaluation)
                print("--------------------------------------------")

    agent.save_models(filename=file_name)
    plot_reward_curve(historical_reward, file_name)


def evaluation_loop(env, agent, frames_stack, total_counter, max_action_value, min_action_value, file_name, historical_reward_evaluation):
    max_steps_evaluation = 1_000
    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state = frames_stack.reset()
    frame = grab_frame(env)

    historical_episode_reward_evaluation = []


    fps = 30
    video_name = f'videos/{file_name}_{total_counter}.mp4'
    height, width, channels = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for total_step_counter in range(max_steps_evaluation):
        episode_timesteps += 1
        action     = agent.select_action_from_policy(state, evaluation=True)
        action_env = hlp.denormalize(action, max_action_value, min_action_value)
        state, reward_extrinsic, done, truncated, info = frames_stack.step(action_env)
        episode_reward += reward_extrinsic

        if episode_num == 0:
            video.write(grab_frame(env))

        if done or truncated:
            original_img, reconstruction = agent.get_reconstruction_for_evaluation(state)
            plot_reconstruction_img(original_img, reconstruction)

            logging.info(f" EVALUATION | Eval Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f}")
            historical_episode_reward_evaluation.append(episode_reward)

            state = frames_stack.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num       += 1

    mean_reward_evaluation = np.round(np.mean(historical_episode_reward_evaluation),2)
    historical_reward_evaluation["avg_episode_reward"].append(mean_reward_evaluation)
    historical_reward_evaluation["step"].append(total_counter)

    save_evaluation_values(historical_reward_evaluation, file_name)
    video.release()


def grab_frame(env):
    frame = env.render()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to BGR for use with OpenCV
    return frame


def main():

    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed         = 50

    env_gym_name = "Pendulum-v1" # BipedalWalker-v3, Pendulum-v1, HalfCheetah-v4"
    env          = gym.make(env_gym_name, render_mode="rgb_array")

    # env_gym_name = "SuperMarioBros-v0"
    # env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")


    env.reset(seed=seed)

    action_size  = env.action_space.shape[0]
    print(action_size)

    latent_size  = 50
    number_stack_frames = 3

    # Create Directories
    # ---------------------------------------
    dir_exists = os.path.exists("videos")
    if not dir_exists:
        os.makedirs("videos")

    dir_exists = os.path.exists("plots")
    if not dir_exists:
        os.makedirs("plots")

    dir_exists = os.path.exists("data_plots")
    if not dir_exists:
        os.makedirs("data_plots")

    # set seeds
    # ---------------------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.action_space.seed(seed)
    # ---------------------------------------

    # agent = Algorithm(
    #     latent_size=latent_size,
    #     action_num=action_size,
    #     device=device,
    #     k=number_stack_frames)
    #
    # intrinsic_on  = True
    # date_time_str = datetime.now().strftime("%m_%d_%H_%M")
    # file_name     = env_gym_name  + "_" + str(date_time_str) + "_" + "NASA_TD3" + "_Intrinsic_" + str(intrinsic_on)
    #
    # train(env, agent, file_name, intrinsic_on, number_stack_frames)


if __name__ == '__main__':
    main()
