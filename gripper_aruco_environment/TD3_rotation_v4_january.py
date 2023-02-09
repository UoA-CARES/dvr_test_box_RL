
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser

from environment_rotation_v3   import RL_ENV
from memory_utilities_v3       import MemoryClass
from TD3_Agent_New             import TD3


def define_set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def define_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--camera_index', type=int, default=0)
    parser.add_argument('--usb_index',    type=int, default=1)
    parser.add_argument('--robot_index',  type=str, default='robot-2')

    parser.add_argument('--replay_memory_size',    type=int, default=1_000_000)
    parser.add_argument('--max_exploration_steps', type=int, default=5_000)

    parser.add_argument('--num_training_episodes', type=int, default=10_000)
    parser.add_argument('--episode_horizont',      type=int, default=50)

    parser.add_argument('--G',          type=int, default=1)
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)

    args   = parser.parse_args()
    return args


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args   = define_parse_args()

    env           = RL_ENV(camera_index=args.camera_index, device_index=args.usb_index)  # todo confirm this with the new version
    memory_buffer = MemoryClass(args.memory_size, device)  # todo also migrate this to the new version
    define_set_seed(args.seed)

    agent = TD3(device)

    # Training Process
    total_rewards      = []
    episode_num        = 0
    episode_reward     = 0
    episode_time_steps = 0

    done = False
    env.reset_env()

    for t in range(args.max_training_steps):

        episode_time_steps += 1
        action = env.generate_sample_act()















if __name__ == '__main__':
    main()