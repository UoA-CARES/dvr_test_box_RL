"""
This will be used only with gym/control suite

"""
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import os
import gym
import torch
import random
import logging
import numpy as np
from argparse import ArgumentParser

import MemoryBuffers
import MBAETD3
import AETD3
import TD3
import Gym_Environment

logging.basicConfig(level=logging.INFO)


def set_seeds(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

def create_directories():
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./data_plots"):
        os.makedirs("./data_plots")

    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")

def plot_reward_curve(data_reward, filename, check_point=False):
    if check_point:
        data = pd.DataFrame.from_dict(data_reward)
        data.to_csv(f"checkpoints/{filename}_checkpoint", index=False)
        data.plot(x='episode', y='reward', title=filename)
        plt.savefig(f"checkpoints/{filename}_checkpoint")
        plt.close()

    else:
        data = pd.DataFrame.from_dict(data_reward)
        data.to_csv(f"data_plots/{filename}", index=False)
        data.plot(x='episode', y='reward', title=filename)
        plt.savefig(f"results/{filename}")
        plt.show()



def backward_episode_reward(episode_experience, episode_reward):

    experience_length = len(episode_experience)
    discount_reward = []
    discount        = 0.98

    for i in range(experience_length):
        discount_reward.append(episode_reward * discount ** i)

    discount_reward = reversed(discount_reward)

    discounted_experience = []
    for experience, dis_reward in zip(episode_experience, discount_reward):
        state, action, reward, next_state, done = experience
        new_reward = reward + dis_reward
        discounted_experience.append((state, action, new_reward, next_state, done))
    return discounted_experience


def train(args, agent, memory, env, act_dim, max_value, file_name, reward_type):
    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state = env.reset()
    done  = False

    episode_experiences = []

    historical_reward   = {"episode": [], "reward": []}
    #historical_reward   = {"step": [], "reward": []}

    for total_step_counter in range(int(args.max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < args.max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{args.max_steps_exploration}")

            if args.agent == "TD3":
                action = env.action_space.sample()
            else:
                action = env.action_sample()

        else:
            logging.info(f"\n Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter} \n")
            action = agent.select_action_from_policy(state)
            noise  = np.random.normal(0, scale=0.10 * max_value, size=act_dim)
            action = action + noise
            action = np.clip(action, -max_value, max_value)

        new_state, reward, done, _ = env.step(action)
        if reward_type == "backward_reward":
            episode_experiences.append((state, action, reward, new_state, done))
        else:
            memory.add_env(state, action, reward, new_state, done)
        state = new_state
        episode_reward += reward

        if total_step_counter >= args.max_steps_exploration:
            if args.agent == "MB_AE_TD3":
                logging.info("Training World and Reward Model")
                experiences = memory.sample_env(args.batch_size)
                agent.train_world_model(experiences)
                agent.train_reward_model(experiences)
                p = np.random.random()
                for _ in range(args.G):
                    if len(memory.buffer_model) >= args.batch_size:
                        if p < 0.6:
                            logging.info(" Training Agent Model with Model Data")
                            experiences = memory.sample_model(args.batch_size)
                            agent.train_policy(experiences)
                        else:
                            logging.info(" Training Agent Model with Env Data")
                            experiences = memory.sample_env(args.batch_size)
                            agent.train_policy(experiences)
                for _ in range(args.M):
                    experiences = memory.sample_env(args.batch_size)
                    d_state, d_action, d_reward, d_next_state, d_done = agent.generate_dream_samples(experiences)
                    memory.add_model(d_state, d_action, d_reward, d_next_state, d_done)

            else:
                for _ in range(args.G):
                    logging.info(" Training Agent Model")
                    experiences = memory.sample_env(args.batch_size)
                    agent.train_policy(experiences)

        if done:
            logging.info(f"Total T:{total_step_counter} Episode {episode_num} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}\n")
            historical_reward["episode"].append(episode_num)
            historical_reward["reward"].append(episode_reward)

            if reward_type == "backward_reward":
                discounted_episode_experience = backward_episode_reward(episode_experiences, episode_reward)
                memory.extend_env(discounted_episode_experience)
                episode_experiences = []

            # Reset environment
            state = env.reset()
            done  = False
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

            if episode_num % args.plot_freq == 0:
                check_point = True
                plot_reward_curve(historical_reward, file_name, check_point)
                check_point = False

    agent.save_models(file_name)
    plot_reward_curve(historical_reward, file_name)


def encoder_models_evaluation(args, agent, env, device, file_name):

    if args.agent == "TD3":
        logging.info("No encoder-decoder models to evaluate in this configuration")
        return

    evaluation_seed = 53
    agent.load_models(file_name)
    set_seeds(evaluation_seed, env)

    state  = env.reset()
    action = env.action_sample()
    new_state, reward, done, _ = env.step(action)

    state_image_tensor = torch.FloatTensor(state)
    state_image_tensor = state_image_tensor.unsqueeze(0).to(device)

    new_state_tensor = torch.FloatTensor(new_state)
    new_state_tensor = new_state_tensor.unsqueeze(0).to(device)

    action_tensor = torch.FloatTensor(action)
    action_tensor = action_tensor.unsqueeze(0).to(device)

    if args.agent == "MB_AE_TD3":

        with torch.no_grad():
            reward_prediction_tensor = agent.reward_model(state_image_tensor, action_tensor, detach_encoder=True)
            reward_prediction        = reward_prediction_tensor.cpu().data.numpy()
        print("Predicted Reward:", reward_prediction, "Actual Reward", reward)

        with torch.no_grad():
            z_next_prediction_tensor = agent.world_model(state_image_tensor, action_tensor, detach_encoder=True)

        with torch.no_grad():
            rec_prediction = agent.decoder(z_next_prediction_tensor)
            rec_prediction = rec_prediction.cpu().data.numpy()

        current_state_true  = state[2]
        next_state_true     = new_state[2]
        reconstructed_image = rec_prediction[0][2]

        diff = cv2.subtract(next_state_true, reconstructed_image)

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.title("Current State")
        plt.grid(visible=None)
        plt.imshow(current_state_true, cmap='gray')

        plt.subplot(2, 2, 2)
        plt.title("Next State True")
        plt.grid(visible=None)
        plt.imshow(next_state_true, cmap='gray')

        plt.subplot(2, 2, 3)
        plt.title("Difference")
        plt.grid(visible=None)
        plt.imshow(diff, cmap='gray')

        plt.subplot(2, 2, 4)
        plt.title("Next State Predicted")
        plt.grid(visible=None)
        plt.imshow(reconstructed_image, cmap='gray')

        plt.show()


    elif args.agent == "AE_TD3":
        with torch.no_grad():
            z_vector = agent.critic.encoder_net(state_image_tensor)
            rec_prediction = agent.decoder(z_vector)
            rec_prediction = rec_prediction.cpu().data.numpy()

        current_state_true  = state[2]
        reconstructed_image = rec_prediction[0][2]

        diff = cv2.subtract(current_state_true, reconstructed_image)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title("Current State")
        plt.grid(visible=None)
        plt.imshow(current_state_true, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title("Reconstruction State")
        plt.grid(visible=None)
        plt.imshow(reconstructed_image, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Difference")
        plt.grid(visible=None)
        plt.imshow(diff, cmap='gray')
        plt.show()

    else:
        logging.info("No encoder- decoder models to evaluate in this configuration")

def agent_models_evaluation(args, agent, env, device, file_name):
    agent.load_models(file_name)

    evaluation_steps_max = 10_000
    episode_timesteps    = 0
    episode_reward       = 0
    episode_num          = 0

    state = env.reset()
    done  = False
    historical_reward = {"episode": [], "reward": []}

    for total_step_counter in range(evaluation_steps_max):
        env.render()
        episode_timesteps += 1
        action = agent.select_action_from_policy(state)
        new_state, reward, done, _ = env.step(action)
        state = new_state
        episode_reward += reward

        if done:
            logging.info(f" Evaluation Episode {episode_num} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}\n")
            historical_reward["episode"].append(episode_num)
            historical_reward["reward"].append(episode_reward)

            # Reset environment
            state = env.reset()
            done  = False
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

    file_name = file_name+"_EVALUATION"
    plot_reward_curve(historical_reward, file_name)



def parse_args():

    parser = ArgumentParser()

    parser.add_argument("--seed",       type=int,  default=571)
    parser.add_argument("--batch_size", type=int,  default=32)

    parser.add_argument('--env_name',    type=str, default='Pendulum-v1') # BipedalWalker-v3, Pendulum-v1
    parser.add_argument("--agent",       type=str, default='TD3')  # MB_AE_TD3 , AE_TD3, TD3
    parser.add_argument("--reward_type", type=str, default="normal_reward")  # normal_reward, backward_reward

    parser.add_argument('--latent_dim',             type=int, default=50)
    parser.add_argument("--max_steps_exploration",  type=int, default=5000) # 3000
    parser.add_argument("--max_steps_training",     type=int, default=50_000)

    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--G", type=int, default=10)
    parser.add_argument("--F", type=int, default=10)

    parser.add_argument("--plot_freq", type=int, default=10)

    return parser.parse_args()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args   = parse_args()

    create_directories()
    file_name = f"{args.agent}_{args.reward_type}_{args.env_name}_seed_{args.seed}"

    if args.agent == "MB_AE_TD3":
        logging.info("Training with Model Based Autoencoder TD3")
        env     = Gym_Environment.CreateEnvironment(args.env_name)
        act_dim = env.act_dim
        obs_dim = args.latent_dim  # latent dimension
        max_action_value = env.max_action
        agent   = MBAETD3.MB_AE_TD3(device, obs_dim, act_dim, max_action_value)

    elif args.agent == "AE_TD3":
        logging.info("Training with Autoencoder TD3")
        env     = Gym_Environment.CreateEnvironment(args.env_name)
        act_dim = env.act_dim
        obs_dim = args.latent_dim  # latent dimension
        max_action_value = env.max_action
        agent = AETD3.AE_TD3(device, obs_dim, act_dim, max_action_value)

    elif args.agent == "TD3":
        logging.info("Training with TD3")
        env        = gym.make(args.env_name)
        act_dim    = env.action_space.shape[0]
        obs_dim    = env.observation_space.shape[0]
        max_action_value = env.action_space.high.max()
        agent = TD3.TD3(device, obs_dim, act_dim, max_action_value)
        env.action_space.seed(args.seed)

    else:
        logging.info("Please select a correct learning method")
        exit()

    set_seeds(args.seed, env)
    replay_buffers = MemoryBuffers.MemoryBuffer()

    train(args, agent, replay_buffers, env, act_dim, max_action_value, file_name, args.reward_type)
    encoder_models_evaluation(args, agent, env, device, file_name)
    agent_models_evaluation(args, agent, env, device, file_name)



if __name__ == '__main__':
    main()
