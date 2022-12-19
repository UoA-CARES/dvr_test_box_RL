"""
    Task: Rotation Cylinder V3
    Algorithm: TD3- MFRL
    Version V3.0
    Date: 19/11/22
    Modification: 26/11/2022
"""
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
from networks_architectures_v3 import Actor, Critic


class TD3agent_rotation:
    def __init__(self, env, device, memory_size, batch_size, G):

        # -------- Hyper-parameters --------------- #
        self.env    = env
        self.device = device 

        self.actor_learning_rate  = 1e-4
        self.critic_learning_rate = 1e-3

        self.gamma = 0.99  # discount factor
        self.tau   = 0.005

        self.G                  = G
        self.update_counter     = 0
        self.policy_freq_update = 2

        self.batch_size  = batch_size
        self.num_states  = 14
        self.num_actions = 4

        self.max_memory_size = memory_size

        self.hidden_size_critic = [256, 256, 256]
        self.hidden_size_actor  = [256, 256, 256]

        # ------------- Initialization memory --------------------- #
        self.memory = MemoryClass(self.max_memory_size, self.device)

        # ---------- Initialization and build the networks ----------- #
        # Main networks
        self.actor  = Actor(self.num_states, self.hidden_size_actor, self.num_actions).to(self.device)
        self.critic = Critic(self.num_states, self.hidden_size_critic, self.num_actions).to(self.device)

        # Target networks
        self.actor_target  = Actor(self.num_states, self.hidden_size_actor, self.num_actions).to(self.device)
        self.critic_target = Critic(self.num_states, self.hidden_size_critic, self.num_actions).to(self.device)

        # copy weights and bias from main to target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

        self.actor.train(True)
        self.critic.train(True)
        self.actor_target.train(True)
        self.critic_target.train(True)


    def get_action_from_policy(self, state):
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # numpy to a tensor with shape [1,14]
            action       = self.actor(state_tensor)
            action       = action.cpu().data.numpy().flatten()
        self.actor.train(True)
        return action


    def step_training(self):
        # check, if enough samples are available in memory
        if self.memory.__len__() <= self.batch_size:
            return

        # update the networks every G times
        for it in range(1, self.G + 1):

            self.update_counter += 1

            states, actions, rewards, next_states, dones = self.memory.sample_experience(self.batch_size)

            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                target_noise = 0.2 * torch.randn_like(actions)
                target_noise = target_noise.clamp(-0.5, 0.5)
                next_actions = next_actions + target_noise
                next_actions = next_actions.clamp(-1, 1)

                next_q_values_q1, next_q_values_q2 = self.critic_target(next_states, next_actions)
                q_min     = torch.min(next_q_values_q1, next_q_values_q2)
                q_target  = rewards + (self.gamma * (1 - dones) * q_min)

            q1, q2 = self.critic(states, actions)

            critic_loss_1 = F.mse_loss(q1, q_target)
            critic_loss_2 = F.mse_loss(q2, q_target)
            critic_loss_total = critic_loss_1 + critic_loss_2

            # Critic step Update
            self.critic_optimizer.zero_grad()
            critic_loss_total.backward()
            self.critic_optimizer.step()

            if self.update_counter % self.policy_freq_update == 0:
                # ------- calculate the actor loss
                action_actor       = self.actor(states)
                actor_q1, actor_q2 = self.critic(states, action_actor)

                actor_q_min = torch.min(actor_q1, actor_q2)
                actor_loss  = - actor_q_min.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ------------------------------------- Update target networks --------------- #
                # update the target networks using tao "soft updates"
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_model(self):
        torch.save(self.actor.state_dict(), 'trained_models/TD3_actor.pth')
        print("models has been saved...")


    def plot_reward_curves(self, rewards, distance):
        plt.figure(figsize=(20, 10))

        plt.subplot(2, 1, 1)
        plt.title("Reward Raw Curve")
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards)

        plt.subplot(2, 1, 2)
        plt.title("Final Distance to Goal")
        plt.ylabel('Distance')
        plt.plot(distance)

        plt.savefig(f"plot_results/TD3_rewards_gripper.png")

        np.savetxt(f'plot_results/TD3_rewards_gripper.txt', rewards)
        np.savetxt(f'plot_results/TD3_distance_gripper.txt', distance)
        print("training curve has been saved...")
        # plt.show()


def run_exploration(env, episodes, horizont, agent):
    mode = "Exploration"
    for episode in tqdm(range(1, episodes+1)):
        env.reset_env()
        for step in range(1, horizont+1):
            state, _, valve_angel_before = env.state_space_function()
            action   = env.generate_sample_act()
            env.env_step(action)
            next_state, image_state, valve_angel_after = env.state_space_function()
            reward, done, distance_to_goal = env.calculate_reward(valve_angel_before, valve_angel_after)
            agent.memory.replay_buffer_add(state, action, reward, next_state, done)
            env.env_render(image=image_state, episode=episode, step=step, done=done, mode=mode, cylinder=next_state[-2:-1])
            if done:
                break
    print(f"******* -----{episodes} for exploration ended-----********* ")



def run_training(env, num_episodes_training, episode_horizont, agent):
    mode    = "Training TD3"
    rewards = []
    episodes_distance_to_goal = []
    for episode in range(1, num_episodes_training + 1):
        env.reset_env()
        episode_reward   = 0
        distance_to_goal = 0

        for step in range(1, episode_horizont + 1):
            state, _, valve_angel_before = env.state_space_function()
            action   = agent.get_action_from_policy(state)
            noise    = np.random.normal(0, scale=0.15, size=4)
            action   = action + noise
            action   = np.clip(action, -1, 1)
            env.env_step(action)
            next_state, image_state, valve_angel_after = env.state_space_function()
            reward, done, distance_to_goal  = env.calculate_reward(valve_angel_before, valve_angel_after)
            episode_reward += reward
            agent.memory.replay_buffer_add(state, action, reward, next_state, done)
            env.env_render(image=image_state, episode=episode, step=step, done=done, mode=mode, cylinder=next_state[-2:-1])

            agent.step_training()

            if done:
                break

        rewards.append(episode_reward)
        episodes_distance_to_goal.append(distance_to_goal)

        print(f"******* -----Episode {episode} Ended-----********* ")
        print(f"Episode Total reward: {episode_reward}, Final Distance to Goal: {distance_to_goal} \n")

        if episode % 100 == 0:
            agent.plot_reward_curves(rewards, episodes_distance_to_goal)
    agent.save_model()
    agent.plot_reward_curves(rewards, episodes_distance_to_goal)



def define_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--camera_index',     type=int, default=0)
    parser.add_argument('--usb_index',        type=int, default=1)
    parser.add_argument('--robot_index',      type=str, default='robot-2')
    parser.add_argument('--replay_max_size',  type=int, default=100_000)

    parser.add_argument('--seed',                     type=int, default=100)
    parser.add_argument('--batch_size',               type=int, default=256)
    parser.add_argument('--G',                        type=int, default=10)
    parser.add_argument('--num_exploration_episodes', type=int, default=1_000)
    parser.add_argument('--num_training_episodes',    type=int, default=10_000)
    parser.add_argument('--episode_horizont',         type=int, default=20)

    args   = parser.parse_args()
    return args


def main_run():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args   = define_parse_args()

    env   = RL_ENV(camera_index=args.camera_index, device_index=args.usb_index)
    agent = TD3agent_rotation(env=env, device=device,  memory_size=args.replay_max_size, batch_size=args.batch_size, G=args.G)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_exploration(env, args.num_exploration_episodes, args.episode_horizont, agent)
    run_training(env, args.num_training_episodes, args.episode_horizont, agent)


if __name__ == '__main__':
    main_run()
