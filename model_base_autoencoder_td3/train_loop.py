"""
This will be used only with gym/control suite


"""

import os
import gym
import torch
import random
import logging
import numpy as np
from argparse import ArgumentParser

import MemoryBuffers
import MBAETD3
import Gym_Environment

logging.basicConfig(level=logging.INFO)

def set_seeds(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    env.seed(seed)

def create_directories():
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

def train(args, agent, memory, env, file_name):
    f_counter = 0
    total_step_counter = 0
    historical_reward  = {"episode": [], "reward": []}

    for episode in range(0, args.train_episode_num):
        step = 0
        episode_reward = 0

        state = env.reset()
        done  = False

        while not done:

            total_step_counter += 1
            step += 1
            logging.info(f"Taking step {step} of Episode {episode} \n")

            if total_step_counter < args.max_exploration_steps:
                logging.info(f"Running Exploration Steps {total_step_counter}/{args.max_exploration_steps}")

                if args.agent == "TD3":
                    action = env.action_space.sample()
                else:
                    action = env.action_sample()

            else:
                pass

            new_state, reward, done, _ = env.step(action)
            memory.add_env(state, action, reward, new_state, done)
            state = new_state
            episode_reward += reward

            if len(memory.buffer_env) >= args.max_exploration_steps:
                # maybe i can put all this as update function inside de agent

                if f_counter <= args.F:
                    experiences = memory.sample_env(args.batch_size)
                    agent.train_world_model(experiences)
                    f_counter += 1
                    logging.info(" Training world model")

                for _ in range(args.M):
                    # experiences = memory.sample_env(args.batch_size)
                    # prediction here
                    # store the prediction experiences here
                    pass

                for _ in range(args.G):
                    # choose the buffer where we sample after some experiences
                    # = memory.sample_env(args.batch_size) # sample from selected buffer
                    # agent learn
                    pass

        logging.info(f"Episode {episode} was completed with {step} actions taken and a Reward= {episode_reward:.3f}\n")
        historical_reward["episode"].append(episode)
        historical_reward["reward"].append(episode_reward)

    #todo save model
    #todo plot results

def parse_args():

    parser = ArgumentParser()
    parser.add_argument("--seed",       type=int,  default=123)
    parser.add_argument("--batch_size", type=int,  default=32)
    parser.add_argument('--env_name',   type=str, default='Pendulum-v1')

    parser.add_argument("--agent",      type=str, default='MB_AE_TD3')  # MB_AE_TD3 , TD3
    parser.add_argument('--latent_dim', type=int, default=50)

    parser.add_argument("--max_exploration_steps",  type=int, default=200)   # 3k - 5k Steps
    parser.add_argument("--train_episode_num",      type=int, default=1000)

    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--G", type=int, default=1)
    parser.add_argument("--F", type=int, default=10)

    return parser.parse_args()



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args   = parse_args()

    create_directories()
    file_name = f"{args.agent}_{args.env_name}_{args.seed}"


    if args.agent == "MB_AE_TD3":
        env     = Gym_Environment.CreateEnvironment(args.env_name)
        act_dim = env.act_dim
        obs_dim = args.latent_dim  # latent dimension
        agent   = MBAETD3.MB_AE_TD3(device, obs_dim, act_dim)

    elif args.agent == "AE_TD3":
        #env   = Gym_Environment.CreateEnvironment(args.env.name)
        #agent = AETD3()
        pass

    elif args.agent == "TD3":
        agent = None
        env = gym.make(args.env_name)
        act_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        pass


    set_seeds(args.seed, env)
    replay_buffers = MemoryBuffers.MemoryBuffer()

    train(args, agent, replay_buffers, env, file_name)


if __name__ == '__main__':
    main()
