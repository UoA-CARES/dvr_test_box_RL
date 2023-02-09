import logging
import signal
import time

import torch

import random
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

from MemoryBuffer import MemoryBuffer
from two_finger_gripper_env import GripperEnvironment
from TD3 import TD3

logging.basicConfig(level=logging.INFO)

def ctrlc_handler(signum, frame):
    res = input("ctrl-c pressed. press anything to quit")
    exit()

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(args, agent, memory, env, obs_dim, act_dim):
    total_step_counter = 0
    for episode in range(0, args.train_episode_num):
        state = env.reset()
        #normalised_state = normalise_state(state)  # TODO normalize this state

        episode_reward = 0
        done           = False

        for step in range(0, args.episode_horizont):
            logging.info(f"Taking step {step}/{args.episode_horizont}")
            total_step_counter += 1

            if total_step_counter < args.max_exploration_steps:
                logging.info(f"Exploration steps {total_step_counter}/{args.max_exploration_steps}")
                action = agent.action_sample()
            else:
                action = agent.select_action(state)
                noise  = np.random.normal(0, scale=0.1, size=act_dim)
                action = action + noise
                action = np.clip(action, -1, 1)
                action = action.astype(int)

            next_state, reward, done, truncated = env.step(action)
            memory.add(state, action, reward, next_state, done)

            if len(memory.buffer) >= args.max_exploration_steps: # todo add also batch_size check
                logging.info("Training Network")
                experiences = memory.sample(args.batch_size)
                agent.learn(experiences)
                #for _ in range(0, args.G):
                    #agent.learn(experiences)







def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--G",               type=int,  default=1)
    parser.add_argument("--seed",            type=int,  default=6969)
    parser.add_argument("--batch_size",      type=int,  default=4)
    parser.add_argument("--buffer_capacity", type=int,  default=1_000_000)

    parser.add_argument("--train_episode_num",      type=int, default=1000)
    parser.add_argument("--episode_horizont",       type=int, default=10)
    parser.add_argument("--max_exploration_steps",  type=int, default=5)

    return parser.parse_args()

def main():
    now = datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M-%S")
    #log_path = f"logs/{now}.log"
    #logging.basicConfig(filename=log_path, level=logging.DEBUG)
    #signal.signal(signal.SIGINT, ctrlc_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args   = parse_args()

    set_seeds(args.seed)

    # todo custom this for different modes
    obs_dim = 15
    act_dim = 4

    env    = GripperEnvironment()
    memory = MemoryBuffer(args.buffer_capacity)
    td3    = TD3(device, obs_dim, act_dim)

    train(args, td3, memory, env, obs_dim, act_dim)



if __name__ == '__main__':
    main()