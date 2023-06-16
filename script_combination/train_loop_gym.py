
import cv2
import gym
import time
import torch
import random

from datetime import datetime

import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

#from cares_reinforcement_learning.util import MemoryBuffer
from Custom_Memory import CustomMemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp

from Algorithm import Algorithm
from FrameStack_3CH import FrameStack

import pandas as pd
import matplotlib.pyplot as plt


from networks import Actor
from networks import Critic
from networks import Encoder
from networks import Decoder



def plot_reward_curve(data_reward, filename):
    data = pd.DataFrame.from_dict(data_reward)
    data.plot(x='step', y='episode_reward', title="Reward Curve")
    plt.title(filename)
    # plt.show()
    plt.savefig(f"plots/{filename}.png")
    plt.close()

def train(env, model_policy,  file_name, intrinsic_on, k):

    max_steps_training    = 100_000
    max_steps_exploration = 1_000

    batch_size = 128
    seed       = 1 # 571 seed gives no that great results
    G          = 1
    k          = k

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]
    action_size      = env.action_space.shape[0]

    #-----------------------------------#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.action_space.seed(seed)
    # -----------------------------------#

    # memory = MemoryBuffer()
    memory = CustomMemoryBuffer(action_size)

    frames_stack = FrameStack(env, k, seed)

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0
    historical_reward = {"step": [], "episode_reward": []}

    start_time = time.time()
    state      = frames_stack.reset()  # for k images

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1
        if total_step_counter < max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action_env = env.action_space.sample()  # action range the env uses [e.g. -2 , 2 for pendulum]
            action     = hlp.normalize(action_env, max_action_value, min_action_value)  # algorithm range [-1, 1]
        else:
            action     = model_policy.get_action_from_policy(state)
            action_env = hlp.denormalize(action, max_action_value, min_action_value)

        next_state, reward_extrinsic, done, truncated, info = frames_stack.step(action_env)

        if intrinsic_on:
            if total_step_counter > max_steps_exploration:
                a = 1
                b = 1
                surprise_rate, novelty_rate = model_policy.get_intrinsic_values(state, action, next_state)

                reward_surprise = surprise_rate * a
                reward_novelty  = novelty_rate  * b
                #logging.info(f"Surprise Rate = {reward_surprise},  Novelty Rate = {reward_novelty}, Normal Reward = {reward_extrinsic}, {total_step_counter}")

                total_reward = reward_extrinsic + reward_surprise + reward_novelty

            else:
                total_reward = reward_extrinsic
        else:
            total_reward = reward_extrinsic

        memory.add(state=state, action=action, reward=total_reward, next_state=next_state, done=done)
        state = next_state

        episode_reward += reward_extrinsic  # just for plotting and evaluation purposes use the  reward as it is

        if total_step_counter >= max_steps_exploration:
            num_updates = max_steps_exploration if total_step_counter == max_steps_exploration else G
            for _ in range(num_updates):
                experience = memory.sample(batch_size)
                model_policy.train_policy(experience)
                if intrinsic_on:
                    # experiences = memory.sample(batch_size)
                    model_policy.train_predictive_model(experience)

        if done or truncated:
            episode_duration = time.time() - start_time
            start_time       = time.time()

            logging.info(f"Total T:{total_step_counter + 1} | Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f} | Duration= {episode_duration:.2f} Seg")
            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state = frames_stack.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num       += 1

            if episode_num % 10 == 0:
                plot_reward_curve(historical_reward, filename=file_name)
                print("--------------------------------------------")
                evaluation_loop(env, model_policy, frames_stack, total_step_counter, max_action_value, min_action_value)
                print("--------------------------------------------")

    model_policy.save_models(filename=file_name)
    hlp.plot_reward_curve(historical_reward)


def evaluation_loop(env, model_policy, frames_stack, total_counter, max_action_value, min_action_value):
    max_steps_evaluation = 1_000

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state = frames_stack.reset()
    frame = grab_frame(env)

    fps = 60
    video_name = f'videos/Result_Gym_{total_counter+1}.mp4'
    height, width, channels = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1

        action     = model_policy.get_action_from_policy(state, evaluation=True)
        action_env = hlp.denormalize(action, max_action_value, min_action_value)

        state, reward_extrinsic, done, truncated, info = frames_stack.step(action_env)

        episode_reward += reward_extrinsic
        video.write(grab_frame(env))

        if done or truncated:
            logging.info(f" EVALUATION | Eval Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f}")
            state = frames_stack.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num       += 1

    video.release()


def grab_frame(env):
    frame = env.render()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to BGR for use with OpenCV
    return frame


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env_gym_name = "Pendulum-v1" # BipedalWalker-v3, Pendulum-v1, HalfCheetah-v4"
    env          = gym.make(env_gym_name, render_mode="rgb_array")
    action_size  = env.action_space.shape[0]
    latent_size  = 50
    number_stack_frames = 3
    k = number_stack_frames * 3

    # encoder = Encoder(latent_dim=latent_size, k=k)
    # decoder = Decoder(latent_dim=latent_size, k=k)
    #
    # actor  = Actor(latent_size, action_size, encoder)
    # critic = Critic(latent_size, action_size, encoder)
    #
    # model_policy = Algorithm(
    #     encoder,
    #     decoder,
    #     actor,
    #     critic,
    #     action_size,
    #     latent_size,
    #     device
    # )

    model_policy = Algorithm(
        latent_size=latent_size,
        action_num=action_size,
        device=device,
        k=number_stack_frames)


    intrinsic_on  = False
    date_time_str = datetime.now().strftime("%m_%d_%H_%M")
    file_name     = env_gym_name  + "_" + str(date_time_str) + "_" + "TD3_AE_Surprise_Novelty" + "_Intrinsic_" + str(intrinsic_on)

    train(env, model_policy, file_name, intrinsic_on, number_stack_frames)


if __name__ == '__main__':
    main()
