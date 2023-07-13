

import os
import cv2
import time
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import logging
logging.basicConfig(level=logging.INFO)

from TD3_Pixels import TD3_Pixel
from dm_control import suite
from FrameStack import FrameStack
from cares_reinforcement_learning.memory import MemoryBuffer

def plot_reward_curve(data_reward, filename):
    data = pd.DataFrame.from_dict(data_reward)
    data.to_csv(f"plot_results/{filename}", index=False)
    data.plot(x='step', y='episode_reward', title="Reward Curve")
    plt.title(filename)
    plt.savefig(f"plot_results/{filename}.png")
    plt.close()


def train(env, agent, file_name, number_stack_frames):

    # Training-parameters
    # ------------------------------------#
    max_steps_training    = 1_000_000
    max_steps_exploration = 1_000

    batch_size = 32
    G = 1 # todo put this 5
    k = number_stack_frames

    # Action size and format
    # ------------------------------------#
    action_spec      = env.action_spec()
    action_size      = action_spec.shape[0]    # For example, 6 for cheetah
    max_action_value = action_spec.maximum[0]  # --> +1
    min_action_value = action_spec.minimum[0]  # --> -1

    # Needed classes
    # ------------------------------------#
    memory       = MemoryBuffer()
    frames_stack = FrameStack(env, k)

    # Training Loop
    # ------------------------------------#
    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    historical_reward = {"step": [], "episode_reward": []}

    start_time = time.time()
    state      = frames_stack.reset()  # unit8 , (9, 84 , 84)

    for total_step_counter in range(1, int(max_steps_training) + 1):
        episode_timesteps += 1

        if total_step_counter <= max_steps_exploration:
            logging.info(f"Running Pre-Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action = np.random.uniform(min_action_value, max_action_value, size=action_size)
        else:
            action = agent.select_action_from_policy(state)

        next_state, reward_extrinsic, done = frames_stack.step(action)

        memory.add(state=state, action=action, reward=reward_extrinsic, next_state=next_state, done=done)
        state = next_state
        episode_reward += reward_extrinsic

        if total_step_counter > max_steps_exploration:
            for _ in range(G):
                experience = memory.sample(batch_size)

                agent.train_policy((
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    experience['next_state'],
                    experience['done'],
                ))

        if done:
            episode_duration = time.time() - start_time
            start_time = time.time()
            logging.info(f"Total T:{total_step_counter} | Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f} | Duration= {episode_duration:.2f} Sec")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            if episode_num % 10 == 0:
                print("*************--Evaluation--*************")
                plot_reward_curve(historical_reward, filename=file_name)
                evaluation_loop(env, agent, frames_stack, total_step_counter, file_name)
                print("--------------------------------------------")

            state = frames_stack.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    agent.save_models(filename=file_name)
    plot_reward_curve(historical_reward, filename=file_name)
    logging.info("All GOOD AND DONE :)")


def evaluation_loop(env, agent, frames_stack, total_counter, file_name):
    max_steps_evaluation = 1_000
    episode_timesteps    = 0
    episode_reward       = 0
    episode_num          = 0

    state = frames_stack.reset()
    frame = grab_frame(env)

    fps = 30
    video_name = f'videos_evaluation/{file_name}_{total_counter}.mp4'
    height, width, channels = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1
        action = agent.select_action_from_policy(state, evaluation=True)
        state, reward_extrinsic, done = frames_stack.step(action)
        episode_reward += reward_extrinsic
        video.write(grab_frame(env))
        if done:
            # original_img, reconstruction = agent.get_reconstruction_for_evaluation(state)
            # plot_reconstruction_img(original_img, reconstruction)
            logging.info(f" EVALUATION | Eval Episode was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f}")
            state = frames_stack.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
    video.release()

def grab_frame(env):
    frame = env.physics.render(camera_id=0, height=480, width=600)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to BGR for use with OpenCV
    return frame

def define_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--env',  type=str, default="ball_in_cup")
    parser.add_argument('--task', type=str, default="catch")
    args   = parser.parse_args()
    return args


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f" Working with = {device}")

    args = define_parse_args()
    domain_name = args.env
    task_name   = args.task
    seed        = args.seed
    logging.info(f" Environment and Task Selected: {domain_name}_{task_name}")

    env         = suite.load(domain_name, task_name, task_kwargs={'random': seed})
    action_spec = env.action_spec()
    action_size = action_spec.shape[0]
    latent_size = 50
    number_stack_frames = 3

    # Create Directories
    logging.info(f" Creating Folders")
    # ------------------------------------------------#
    dir_exists = os.path.exists("videos_evaluation")
    if not dir_exists:
        os.makedirs("videos_evaluation")

    dir_exists = os.path.exists("plot_results")
    if not dir_exists:
        os.makedirs("plot_results")

    dir_exists = os.path.exists("models")
    if not dir_exists:
        os.makedirs("models")
    # ------------------------------------------------#

    # set seeds
    # ------------------------------------------------#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # ------------------------------------------------#

    logging.info(f" Initializing Algorithm.....")
    agent = TD3_Pixel(
        latent_size=latent_size,
        action_num=action_size,
        device=device,
        k=number_stack_frames)


    logging.info(f"Working with Encoder-Pixel-TD3")

    date_time_str = datetime.now().strftime("%m_%d_%H_%M")
    file_name     = domain_name + "_" + str(date_time_str) + "_" + task_name + "_" + "Pixel_TD3"
    logging.info(f" File name for this training loop: {file_name}")

    logging.info("Initializing Training Loop....")
    train(env, agent, file_name, number_stack_frames)


if __name__ == '__main__':
    main()
