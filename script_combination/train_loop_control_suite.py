
import cv2
from dm_control import suite
import time
import torch
import random

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

from FrameStack_DMCS import FrameStack
from cares_reinforcement_learning.util   import helpers as hlp

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.algorithm.policy.NaSATD3 import NASA_TD3
from cares_reinforcement_learning.networks.NaSATD3 import Actor
from cares_reinforcement_learning.networks.NaSATD3 import Critic
from cares_reinforcement_learning.networks.NaSATD3 import Encoder
from cares_reinforcement_learning.networks.NaSATD3 import Decoder


def plot_reward_curve(data_reward, filename):
    data = pd.DataFrame.from_dict(data_reward)
    data.plot(x='step', y='episode_reward', title="Reward Curve")
    plt.title(filename)
    # plt.show()
    plt.savefig(f"plots/{filename}.png")
    plt.close()

def train(env, agent, file_name, intrinsic_on, number_stack_frames, seed):
    # Hyperparameters
    # ------------------------------------#
    max_steps_training    = 1_000_000
    max_steps_exploration = 1_000

    batch_size = 128
    seed       = seed
    G          = 1
    k          = number_stack_frames
    # ------------------------------------#

    # Action size and format
    # ------------------------------------#
    action_spec      = env.action_spec()
    action_size      = action_spec.shape[0]    # For example, 6 for cheetah
    max_action_value = action_spec.maximum[0]  # --> +1
    min_action_value = action_spec.minimum[0]  # --> -1
    # ------------------------------------#

    # set seeds
    # ------------------------------------#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # ------------------------------------#

    # Needed classes
    # ------------------------------------#
    memory       = MemoryBuffer()
    frames_stack = FrameStack(env, k)
    # ------------------------------------#

    # Training Loop
    # ------------------------------------#
    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0
    historical_reward = {"step": [], "episode_reward": []}

    start_time = time.time()
    state      = frames_stack.reset()  # for 3 images with color, unit8 , (9, 84 , 84)

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1
        if total_step_counter < max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action = np.random.uniform(min_action_value, max_action_value, size=action_size)
        else:
            action = agent.select_action_from_policy(state)  # no normalization needed for action, already between [-1, 1]

        next_state, reward_extrinsic, done = frames_stack.step(action)

        if intrinsic_on and total_step_counter >= max_steps_exploration:
            a = 0.5
            b = 0.5
            surprise_rate, novelty_rate = agent.get_intrinsic_values(state, action, next_state)
            reward_surprise = surprise_rate * a
            reward_novelty  = novelty_rate  * b
            logging.info(f"Surprise Rate = {reward_surprise},  Novelty Rate = {reward_novelty}, Normal Reward = {reward_extrinsic}, {total_step_counter}")
            total_reward = reward_extrinsic + reward_surprise + reward_novelty
        else:
            total_reward = reward_extrinsic

        memory.add(state=state, action=action, reward=total_reward, next_state=next_state, done=done)
        state = next_state

        episode_reward += reward_extrinsic  # just for plotting purposes use this reward as it is

        if total_step_counter >= max_steps_exploration:
            num_updates = max_steps_exploration if total_step_counter == max_steps_exploration else G

            for _ in range(num_updates):
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
                        experience['next_state']
                    ))

        if done:
            episode_duration = time.time() - start_time
            start_time       = time.time()

            logging.info(f"Total T:{total_step_counter + 1} | Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f} | Duration= {episode_duration:.2f} Seg")
            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            state = frames_stack.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num      += 1

            if episode_num % 10 == 0:
                plot_reward_curve(historical_reward, filename=file_name)
                print("--------------------------------------------")
                evaluation_loop(env, agent, frames_stack, total_step_counter, file_name)
                print("--------------------------------------------")

    agent.save_models(filename=file_name)
    plot_reward_curve(historical_reward, filename=file_name)



def evaluation_loop(env, agent, frames_stack, total_counter, file_name):
    max_steps_evaluation = 1_000

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state = frames_stack.reset()
    frame = grab_frame(env)

    fps = 30
    video_name = f'videos/{file_name}_{total_counter+1}.mp4'
    height, width, channels = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1
        action = agent.get_action_from_policy(state, evaluation=True)
        state, reward_extrinsic, done = frames_stack.step(action)
        episode_reward += reward_extrinsic

        video.write(grab_frame(env))

        if done:
            logging.info(f" EVALUATION | Eval Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f}")
            state = frames_stack.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num       += 1

    video.release()


def grab_frame(env):
    frame = env.physics.render(camera_id=0, height=480, width=600)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to BGR for use with OpenCV
    return frame


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Domain = cartpole, cheetah, reacher, ball_in_cup
    # task   = balance , run,     easy,    catch

    domain_name = "ball_in_cup"
    task_name   = "catch"
    seed        = 1
    env         = suite.load(domain_name, task_name, task_kwargs={'random': seed})
    action_spec = env.action_spec()
    action_size = action_spec.shape[0]
    latent_size = 50
    number_stack_frames = 3
    k = number_stack_frames * 3

    encoder = Encoder(latent_dim=latent_size, k=k)
    decoder = Decoder(latent_dim=latent_size, k=k)
    actor   = Actor(latent_size, action_size, encoder)
    critic = Critic(latent_size, action_size, encoder)

    gamma = 0.99
    tau   = 0.005

    agent = NASA_TD3(
        encoder,
        decoder,
        actor,
        critic,
        gamma,
        tau,
        action_size,
        latent_size,
        device
    )

    intrinsic_on  = False
    date_time_str = datetime.now().strftime("%m_%d_%H_%M")
    file_name     = domain_name + "_" + str(date_time_str) + "_" + task_name + "_" + "NASA_TD3" + "_Intrinsic_" + str(intrinsic_on)

    train(env, agent, file_name, intrinsic_on, number_stack_frames, seed)


if __name__ == '__main__':
    main()
