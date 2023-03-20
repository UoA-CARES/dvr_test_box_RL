

import pandas as pd
import matplotlib.pyplot as plt

import cv2
import os
import torch
import random
import logging
import numpy as np
from argparse import ArgumentParser


import TD3
import TD3_AE
import MemoryBuffer
from Four_DoF_Environment import GripperEnvironment

logging.basicConfig(level=logging.INFO)

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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
        data.to_csv(f"checkpoints/{filename}_check_point", index=False)
        data.plot(x='episode', y='reward', title=filename)
        plt.savefig(f"checkpoints/{filename}_check_point")
        plt.close()

    else:
        data = pd.DataFrame.from_dict(data_reward)
        data.to_csv(f"data_plots/{filename}", index=False)
        data.plot(x='episode', y='reward', title=filename)
        plt.savefig(f"results/{filename}")
        plt.show()



def train(args, agent, memory, env, act_dim, file_name):
    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state = env.reset()
    done  = False

    episode_experiences = []
    historical_reward = {"episode": [], "reward": []}

    for total_step_counter in range(int(args.max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < args.max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{args.max_steps_exploration}")
            action = agent.action_sample()

        else:
            logging.info(f"Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter} \n")
            action = agent.select_action_from_policy(state)
            noise  = np.random.normal(0, scale=0.10, size=act_dim)
            action = action + noise
            action = np.clip(action, -1, 1)
            action = action.tolist()
            #logging.info(f"Action: {action}")

        next_state, reward, done, _ = env.step(action)
        if not args.discriminate_reward:
            memory.add(state, action, reward, next_state, done)
        else:
            episode_experiences.append((state, action, reward, next_state, done))

        state = next_state
        episode_reward += reward

        if total_step_counter >= args.max_steps_exploration:
            logging.info("Training Agent Model")
            for _ in range(0, args.G):
                experiences = memory.sample(args.batch_size)
                agent.train_policy(experiences)

        if (done == True) or (episode_timesteps >= args.episode_horizont):

            logging.info(f"Total T:{total_step_counter+1} Episode {episode_num+1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}\n")

            historical_reward["episode"].append(episode_num)
            historical_reward["reward"].append(episode_reward)

            if args.discriminate_reward:
                if not episode_reward == 0.0:
                    memory.extend(episode_experiences)
                    logging.info(f"Buffer_size: {len(memory.buffer)}")
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

    logging.info("Evaluating  Autoencoder")
    evaluation_seed = 147
    agent.load_models(file_name)
    set_seeds(evaluation_seed)

    # evaluate the encoder-decoder
    state  = env.reset()
    #action = agent.action_sample()
    #new_state, reward, done, _ = env.step(action)

    state_image_tensor = torch.FloatTensor(state)
    state_image_tensor = state_image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        z_vector = agent.critic.encoder_net(state_image_tensor)
        rec_obs  = agent.decoder(z_vector)
        rec_obs  = rec_obs.cpu().data.numpy()

    current_state_true  = state[2]
    reconstructed_image = rec_obs[0][2]

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
    plt.savefig(f"results/image_reconstruction_{file_name}.png")
    plt.show()

def agent_models_evaluation(args, agent, env, device, file_name):
    agent.load_models(file_name)

    episode_timesteps    = 0
    episode_reward       = 0
    episode_num          = 0

    state = env.reset()
    done  = False
    historical_reward = {"episode": [], "reward": []}

    for total_step_counter in range(args.max_evaluation_steps):
        episode_timesteps += 1
        logging.info(f" Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter} \n")

        action = agent.select_action_from_policy(state)
        action = action.tolist()
        new_state, reward, done, _ = env.step(action)
        state = new_state
        episode_reward += reward

        if (done == True) or (episode_timesteps >= args.episode_horizont):

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

    parser.add_argument("--seed",       type=int, default=571)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument('--agent', type=str, default='TD3')  # AE_TD3 , TD3

    parser.add_argument('--latent_dim',             type=int, default=50)
    parser.add_argument("--max_steps_exploration",  type=int, default=3_000)
    parser.add_argument("--max_steps_training",     type=int, default=50_000)
    parser.add_argument("--episode_horizont",       type=int, default=30)
    parser.add_argument("--max_evaluation_steps",   type=int, default=100)

    parser.add_argument("--discriminate_reward",    default=True)
    parser.add_argument("--motor_reset_on",  type=str,  default="On")

    parser.add_argument("--buffer_capacity", type=int, default=1_000_000)

    parser.add_argument("--G",         type=int, default=10)
    parser.add_argument('--plot_freq', type=int, default=25)

    parser.add_argument('--usb_port',   type=str, default='/dev/ttyUSB1')  # '/dev/ttyUSB1', '/dev/ttyUSB0'
    parser.add_argument('--robot_id',   type=str, default='RR')  # RR, RL
    parser.add_argument('--camera_id',  type=int, default=0)  # 0, 2
    parser.add_argument('--num_motors',  type=int, default=4)


    return parser.parse_args()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args   = parse_args()
    create_directories()
    set_seeds(args.seed)

    if args.agent == "AE_TD3":
        logging.info("Training with Autoencoder TD3")

        train_mode = 'autoencoder'
        obs_dim = args.latent_dim  # latent dimension
        act_dim = 4
        agent = TD3_AE.TD3(device, obs_dim, act_dim)

    elif args.agent == "TD3":
        logging.info("Training with Vector TD3")

        train_mode = 'vector'
        obs_dim = 15
        act_dim = 4
        agent   = TD3.TD3(device, obs_dim, act_dim)

    else:
        logging.info("Please select a correct learning method")
        exit()

    file_name      = f"{args.agent}_seed_{args.seed}_{args.robot_id}_motor_reset_{args.motor_reset_on}"
    replay_buffers = MemoryBuffer.MemoryBuffer(args.buffer_capacity)


    env = GripperEnvironment(num_motors=args.num_motors, motor_reset=args.motor_reset_on, camera_id=args.camera_id, device_name=args.usb_port, train_mode=train_mode)

    train(args, agent, replay_buffers, env, act_dim, file_name)
    encoder_models_evaluation(args, agent, env, device, file_name)
    agent_models_evaluation(args, agent, env, device, file_name)


if __name__ == '__main__':
    main()