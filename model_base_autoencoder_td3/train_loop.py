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

    if not os.path.exists("./data_plots"):
        os.makedirs("./data_plots")

def train(args, agent, memory, env, act_dim, max_value, file_name):
    f_counter = 0
    total_step_counter = 0
    historical_reward  = {"episode": [], "reward": []}


    for episode in range(1, args.train_episode_num):
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
                action = agent.select_action_from_policy(state)
                noise = np.random.normal(0, scale=0.10*max_value, size=act_dim)
                action = action + noise
                action = np.clip(action, -max_value, max_value)

            new_state, reward, done, _ = env.step(action)
            memory.add_env(state, action, reward, new_state, done)
            state = new_state
            episode_reward += reward

            if len(memory.buffer_env) >= args.max_exploration_steps:
                if f_counter <= args.F:
                    logging.info(" Training World and Reward model")
                    experiences = memory.sample_env(args.batch_size)
                    agent.train_world_model(experiences)
                    agent.train_reward_model(experiences)
                    f_counter += 1

                for _ in range(args.G):
                    if len(memory.buffer_model) and len(memory.buffer_env) >= args.batch_size:
                        experiences = memory.sample_env(args.batch_size)
                        #experiences = memory.sample_model(args.batch_size)
                        #experiences = memory.sample_policy(args.batch_size) # still need to code this

                        agent.train_policy(experiences)
                        logging.info(" Training actor critic agent models")

                for _ in range(args.M):
                    experiences = memory.sample_env(args.batch_size)
                    d_state, d_action, d_reward, d_next_state, d_done = agent.generate_dream_samples(experiences)
                    memory.add_model(d_state, d_action, d_reward, d_next_state, d_done)


        logging.info(f"Episode {episode} was completed with {step} actions taken and a Reward= {episode_reward:.3f}\n")
        historical_reward["episode"].append(episode)
        historical_reward["reward"].append(episode_reward)

    agent.save_models(file_name)
    #todo plot results and save values for re_ploting


def parse_args():

    parser = ArgumentParser()
    parser.add_argument("--seed",       type=int,  default=571)
    parser.add_argument("--batch_size", type=int,  default=32)
    parser.add_argument('--env_name',   type=str, default='Pendulum-v1')

    parser.add_argument("--agent",      type=str, default='MB_AE_TD3')  # MB_AE_TD3 , TD3
    parser.add_argument('--latent_dim', type=int, default=50)

    parser.add_argument("--max_exploration_steps",  type=int, default=400)   # 3k - 5k Steps
    parser.add_argument("--train_episode_num",      type=int, default=7)

    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--G", type=int, default=10)
    parser.add_argument("--F", type=int, default=10)

    return parser.parse_args()



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args   = parse_args()

    create_directories()
    file_name = f"{args.agent}_{args.env_name}_Seed_{args.seed}"


    if args.agent == "MB_AE_TD3":
        env     = Gym_Environment.CreateEnvironment(args.env_name)
        act_dim = env.act_dim
        obs_dim = args.latent_dim  # latent dimension
        max_action_value = env.max_action
        agent   = MBAETD3.MB_AE_TD3(device, obs_dim, act_dim, max_action_value)

    elif args.agent == "AE_TD3":
        #env   = Gym_Environment.CreateEnvironment(args.env.name)
        #agent = AETD3()
        pass

    elif args.agent == "TD3":
        #agent = None
        #env = gym.make(args.env_name)
        #max_action_value = env.action_space.high.max()
        #act_dim = env.action_space.shape[0]
        #obs_dim = env.observation_space.shape[0]
        pass


    set_seeds(args.seed, env)
    replay_buffers = MemoryBuffers.MemoryBuffer()
    train(args, agent, replay_buffers, env, act_dim, max_action_value, file_name)


if __name__ == '__main__':
    main()
