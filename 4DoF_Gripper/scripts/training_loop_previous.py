

import logging

import torch
import random
import numpy as np
from argparse import ArgumentParser

from TD3 import TD3
from MemoryBuffer import MemoryBuffer
from Four_DoF_Environment import GripperEnvironment
from Plot import Plot


logging.basicConfig(level=logging.INFO)


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def normalise_state(state):
    state_norm = [(element - np.mean(state)) / np.std(state) for element in state]
    return state_norm

def train(args, agent, memory, env, act_dim, file_name, plt):
    total_step_counter = 0
    historical_reward  = {}
    historical_reward["episode"] = []
    historical_reward["reward"]  = []

    noise_scale         = 0.2
    minimal_noise_scale = 0.01
    noise_decay_range   = 0.9995

    for episode in range(0, args.train_episode_num):
        state = env.reset()
        #state = normalise_state(state)
        episode_reward = 0
        step           = 0
        done           = False

        noise_scale =  noise_scale * noise_decay_range
        noise_scale = max(minimal_noise_scale, noise_scale)

        for step in range(0, args.episode_horizont):
            print("\n")
            logging.info(f"Taking step {step+1}/{args.episode_horizont}")
            total_step_counter += 1

            if total_step_counter < args.max_exploration_steps:
                logging.info(f"Exploration steps {total_step_counter}/{args.max_exploration_steps}")
                action = agent.action_sample()
            else:
                action = agent.select_action(state)
                noise  = np.random.normal(0, scale=noise_scale, size=act_dim)
                print("Pure action:", action)
                action = action + noise
                action = np.clip(action, -1, 1)
                action = action.tolist()

            print("Action:", action)

            next_state, reward, done, truncated = env.step(action)
            memory.add(state, action, reward, next_state, done)
            state = next_state
            #state = normalise_state(state)

            episode_reward += reward

            if len(memory.buffer) >= args.max_exploration_steps:
            #if len(memory.buffer) >= args.batch_size:
                logging.info("Training Network")
                for _ in range(0, args.G):
                    experiences = memory.sample(args.batch_size)
                    agent.learn(experiences)
            if done:
                break

        plt.post(episode_reward)
        historical_reward["episode"].append(episode)
        historical_reward["reward"].append(episode_reward)
        logging.info(f"Episode {episode} was completed with {step + 1} actions taken and a Reward= {episode_reward:.3f}\n")

    agent.save_models(file_name)
    plt.save_plot(file_name)
    plt.save_csv(file_name)
    plt.plot()


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--G",               type=int,  default=5)
    parser.add_argument("--seed",            type=int,  default=0)
    parser.add_argument("--batch_size",      type=int,  default=32) # 100
    parser.add_argument("--buffer_capacity", type=int,  default=100_000)

    parser.add_argument("--train_episode_num",      type=int, default=5_000)  # 1000 Episodes
    parser.add_argument("--episode_horizont",       type=int, default=20)     # 50
    parser.add_argument("--max_exploration_steps",  type=int, default=2000)   # 3k - 5k Steps

    parser.add_argument('--train_mode', type=str, default='aruco')  # aruco, servos, aruco_servos
    parser.add_argument('--usb_port',   type=str, default='/dev/ttyUSB1') # '/dev/ttyUSB1', '/dev/ttyUSB0'
    parser.add_argument('--robot_id',   type=str, default='RR')  # RR, RL
    parser.add_argument('--camera_id',  type=int, default=0)  # 0, 2
    parser.add_argument('--num_motors', type=int, default=4)
    parser.add_argument('--plot_freq',  type=int, default=50)

    return parser.parse_args()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args   = parse_args()
    set_seeds(args.seed)

    file_name = f"TD3_mode_{args.train_mode}_seed_{args.seed}_{args.robot_id}"

    print("---------------------------------------")
    print(f"Train mode: {args.train_mode}, robot: {args.robot_id}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.train_mode == 'aruco':
        logging.info("Training Using Aruco markers coord, target and Object's Yaw angle")
        obs_dim = 16
    elif args.train_mode == 'servos':
        logging.info("Training Using Servo positions, target and Object's Yaw angle")
        obs_dim = 6
    elif args.train_mode == 'aruco_servos':
        logging.info("Training Using Aruco markers coord, Servo positions, target and Object's Yaw angle")
        obs_dim = 20
    else:
        logging.info("Please select a correct train mode")
        exit()

    act_dim = args.num_motors

    env    = GripperEnvironment(args.num_motors, args.camera_id, args.usb_port, args.train_mode)
    memory = MemoryBuffer(args.buffer_capacity)
    td3    = TD3(device, obs_dim, act_dim)
    plt    = Plot(title=file_name, plot_freq=args.plot_freq)
    train(args, td3, memory, env, act_dim, file_name, plt)



if __name__ == '__main__':
    main()