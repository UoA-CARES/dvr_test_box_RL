
"""
Author: David Valencia
Date: 17/11/2022
Modification: 23/11/2022
Description:
            AE-Using test bed camera
            original image size      = 1024 * 960
            image input size for NN  = 84 * 84
            latent vector size       = 50
"""
import random
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from gripper_agent import Td3Agent
from gripper_environment import ENV
from gripper_memory_utilities import MemoryClass, FrameStack



def define_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--k',                     type=int,  default=3)
    parser.add_argument('--include_goal_angle_on', type=bool, default=True)
    parser.add_argument('--camera_index',          type=int,  default=2)
    parser.add_argument('--usb_index',             type=int,  default=0)
    parser.add_argument('--robot_index',           type=str,  default='robot-1')
    parser.add_argument('--replay_max_size',       type=int,  default=200_000)

    parser.add_argument('--seed',                     type=int, default=100)
    parser.add_argument('--batch_size',               type=int,  default=256)
    parser.add_argument('--num_exploration_episodes', type=int,  default=1_000)
    parser.add_argument('--num_training_episodes',    type=int,  default=10_000)
    parser.add_argument('--episode_horizont',         type=int,  default=20)

    args   = parser.parse_args()
    return args

def main_run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args   = define_parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = ENV(
        camera_index=args.camera_index,
        device_index=args.usb_index,
    )

    memory_buffer = MemoryClass(
        replay_max_size=args.replay_max_size,
        device=device,
    )

    agent = Td3Agent(
        env=env,
        robot_index=args.robot_index,
        device=device,
        memory_buffer=memory_buffer,
        include_goal_angle_on=args.include_goal_angle_on,
        batch_size=args.batch_size
    )

    frame_stack = FrameStack(
        k=args.k,
        env=env
    )


    initial_exploration(env, frame_stack, memory_buffer, args.num_exploration_episodes, args.episode_horizont)
    train_function(env, agent, frame_stack, memory_buffer, args.num_training_episodes, args.episode_horizont)

def initial_exploration(env, frames_stack, memory, num_exploration_episodes, episode_horizont):
    print("exploration start")
    for episode in tqdm(range(1, num_exploration_episodes + 1)):
        state_images  = frames_stack.reset()
        goal_angle    = env.define_goal_angle()
        for step in range(1, episode_horizont + 1):
            action = env.generate_sample_action()
            new_state_images, reward, done, distance, original_img, valve_angle = frames_stack.step(action, goal_angle)
            memory.save_experience_to_buffer(state_images, action, reward, new_state_images, done, goal_angle)
            state_images = new_state_images
            env.render(original_img, step, episode, valve_angle, goal_angle, done)
            if done:
                break
    print("exploration end")

def train_function(env, agent, frames_stack, memory, num_training_episodes, episode_horizont):
    episodes_total_reward     = []
    episodes_distance_to_goal = []
    for episode in range(1, num_training_episodes + 1):
        state_images   = frames_stack.reset()
        goal_angle     = env.define_goal_angle()
        episode_reward   = 0
        distance_to_goal = 0
        for step in range(1, episode_horizont + 1):
            action = agent.select_action_from_policy(state_images, goal_angle)
            noise  = np.random.normal(0, scale=0.15, size=4)
            action = action + noise
            action = np.clip(action, -1, 1)
            new_state_images, reward, done, distance_to_goal, original_img, valve_angle = frames_stack.step(action, goal_angle)
            memory.save_experience_to_buffer(state_images, action, reward, new_state_images, done, goal_angle)
            state_images = new_state_images
            episode_reward += reward
            env.render(original_img, step, episode, valve_angle, goal_angle, done)

            agent.update_function()  # --> update function

            if done:
                print("done ---> TRUE, breaking loop, end of this episode")
                break

        episodes_total_reward.append(episode_reward)
        episodes_distance_to_goal.append(distance_to_goal)

        print(f"Episode {episode} End, Total reward: {episode_reward}, Final Distance to Goal: {distance_to_goal} \n")
        if episode % 100 == 0:
            agent.plot_results(episodes_total_reward, episodes_distance_to_goal, check_point=True)

    agent.save_models()
    agent.plot_results(episodes_total_reward, episodes_distance_to_goal, check_point=False)



if __name__ == '__main__':
    main_run()
