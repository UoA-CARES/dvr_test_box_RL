
"""
Author: David Valencia
Date: 17/11/2022
Modification:
Description:
            AE-Using test bed camera
            input image size = 1024 * 960
            input NN size    = 84 * 84
            latent vector    = 50
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from gripper_environment import ENV
from gripper_architectures import Actor, Critic, Decoder
from gripper_function_utilities import FrameStack
from gripper_memory_utilities import MemoryClass



'''
    def plot_functions(self, rewards, final_distance, check_point):

        np.savetxt(f"plot_results/{self.robot_index}_rewards.txt", rewards)
        np.savetxt(f"plot_results/{self.robot_index}_final_distance.txt", final_distance)

        avg_plot_window = 100
        plt.figure(figsize=(20, 10))

        plt.subplot(2, 2, 1)  # row 1, col 2 index 1
        plt.title("Reward Function Curve")
        rewards_smoothed = pd.Series(rewards).rolling(avg_plot_window, min_periods=avg_plot_window).mean()
        plt.plot(rewards)
        plt.plot(rewards_smoothed)

        plt.subplot(2, 2, 2)  # index 2
        plt.title("Final Distance to Goal")
        distance_smoothed = pd.Series(final_distance).rolling(avg_plot_window, min_periods=avg_plot_window).mean()
        plt.plot(final_distance)
        plt.plot(distance_smoothed)

        plt.subplot(2, 2, 3)  # index 3
        plt.title("Prediction Loss Curve")
        plt.plot(self.forward_prediction_loss)

        plt.subplot(2, 2, 4)  # index 4
        plt.title("VAE Loss Curve")
        plt.plot(self.vae_loss)

        if check_point:
            #plt.subplots(figsize=(20, 10))
            plt.savefig(f"plot_results/{self.robot_index}_check_point_curve.png")
        else:
            plt.savefig(f"plot_results/{self.robot_index}_final_curve.png")
            #plt.show()
'''


class Td3Agent:
    def __init__(self, env, robot_index, device, memory_buffer, include_goal_angle_on):
        self.env         = env
        self.robot_index = robot_index
        self.device      = device
        self.memory      = memory_buffer

        # ---------------- Hyperparameters  -------------------------#
        self.encoder_lr = 1e-3
        self.decoder_lr = 1e-3
        self.critic_lr  = 1e-3
        self.actor_lr   = 1e-4

        self.gamma       = 0.99
        self.tau         = 0.005
        self.tau_encoder = 0.001

        self.G                  = 3  # 1
        self.update_counter     = 0
        self.policy_freq_update = 2

        self.batch_size = 64  # 32
        self.action_dim = 4

        self.include_goal_angle_on = include_goal_angle_on

        self.ae_loss_record = []

        # by include_goal_angle_on True, the  target angle is concatenated with the latent vector
        if self.include_goal_angle_on:
            self.latent_dim = 50
            self.input_dim  = 51  # 50 for latent size and 1 for target value
        else:
            self.latent_dim = 50
            self.input_dim  = 50

        # ----------------- Networks ------------------------------#
        # main networks
        self.actor  = Actor(self.latent_dim, self.input_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.latent_dim, self.input_dim, self.action_dim).to(self.device)

        # target networks
        self.actor_target = Actor(self.latent_dim, self.input_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.latent_dim, self.input_dim, self.action_dim).to(self.device)

        # tie encoders between actor and critic, any changes in the critic encoder will also be affecting the actor-encoder
        self.actor.encoder_net.copy_conv_weights_from(self.critic.encoder_net)

        # copy weights and bias from main to target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Decoder
        self.decoder = Decoder(self.latent_dim).to(device)

        # ----------------- Optimizer ------------------------------#
        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder_net.parameters(), lr=self.encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.decoder_lr, weight_decay=1e-7)
        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(),   lr=self.actor_lr, betas=(0.9, 0.999))
        self.critic_optimizer  = torch.optim.Adam(self.critic.parameters(),  lr=self.critic_lr, betas=(0.9, 0.999))

        self.actor.train(True)
        self.critic.train(True)
        self.critic_target.train(True)
        self.actor_target.train(True)
        self.decoder.train(True)

    def select_action_from_policy(self, state_image_pixel, goal_angle):
        with torch.no_grad():
            goal_angle         = np.array(goal_angle).reshape(-1, 1)
            goal_angle_tensor  = torch.FloatTensor(goal_angle).to(self.device)  # torch.Size([1, 1])
            state_image_tensor = torch.FloatTensor(state_image_pixel)
            state_image_tensor = state_image_tensor.unsqueeze(0).to(self.device)  # torch.Size([1, 3, 84, 84])
            action = self.actor(state_image_tensor, goal_angle_tensor, self.include_goal_angle_on)
        action = action.cpu().data.numpy().flatten()
        return action

    def update_function(self):
        if len(self.memory.memory_buffer) <= self.batch_size:
            return
        else:
            for _ in range(1, self.G+1):
                self.update_counter += 1
                state_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, goal_batch = self.memory.sample_experiences_from_buffer(self.batch_size)

                # %%%%%%%%%%%%%%%%%%% Update the critic part %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                with torch.no_grad():
                    next_actions = self.actor_target(next_states_batch, goal_batch, self.include_goal_angle_on)
                    target_noise = 0.2 * torch.randn_like(actions_batch)
                    target_noise = target_noise.clamp(-0.5, 0.5)
                    next_actions = next_actions + target_noise
                    next_actions = next_actions.clamp(-1, 1)

                    next_q_values_q1, next_q_values_q2 = self.critic_target(next_states_batch, next_actions, goal_batch, self.include_goal_angle_on)
                    q_min    = torch.min(next_q_values_q1, next_q_values_q2)
                    q_target = rewards_batch + (self.gamma * (1 - dones_batch) * q_min)

                q1, q2 = self.critic(state_batch, actions_batch, goal_batch, self.include_goal_angle_on)

                critic_loss_1 = F.mse_loss(q1, q_target)
                critic_loss_2 = F.mse_loss(q2, q_target)
                critic_loss_total = critic_loss_1 + critic_loss_2

                self.critic_optimizer.zero_grad()
                critic_loss_total.backward()
                self.critic_optimizer.step()

                # %%%%%%%%%%% Update the actor and soft updates of targets networks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if self.update_counter % self.policy_freq_update == 0:
                    action_actor       = self.actor(state_batch, goal_batch, self.include_goal_angle_on, detach_encoder=True)
                    actor_q1, actor_q2 = self.critic(state_batch, action_actor, goal_batch, self.include_goal_angle_on, detach_encoder=True)

                    actor_q_min = torch.min(actor_q1, actor_q2)
                    actor_loss  = - actor_q_min.mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # ------------------------------------- Update target networks --------------- #
                    for param, target_param in zip(self.critic.Q1.parameters(), self.critic_target.Q1.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.critic.Q2.parameters(), self.critic_target.Q2.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.critic.encoder_net.parameters(), self.critic_target.encoder_net.parameters()):
                        target_param.data.copy_(self.tau_encoder * param.data + (1 - self.tau_encoder) * target_param.data)

                    for param, target_param in zip(self.actor.encoder_net.parameters(), self.actor_target.encoder_net.parameters()):
                        target_param.data.copy_(self.tau_encoder * param.data + (1 - self.tau_encoder) * target_param.data)

                # %%%%%%%%%%%%%%%% Update the autoencoder part %%%%%%%%%%%%%%%%%%%%%%%%
                z_vector = self.critic.encoder_net(state_batch)
                rec_obs  = self.decoder(z_vector)

                rec_loss    = F.mse_loss(state_batch, rec_obs)
                latent_loss = (0.5 * z_vector.pow(2).sum(1)).mean()  # add L2 penalty on latent representation
                ae_loss     = rec_loss + 1e-6 * latent_loss
                self.ae_loss_record.append(ae_loss.item())

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                ae_loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

    def save_models(self):
        torch.save(self.actor.state_dict(), f'trained_models/AE-TD3_actor_gripper.pht')
        torch.save(self.critic.encoder_net.state_dict(), f'trained_models/AE-TD3_encoder_gripper.pht')
        torch.save(self.decoder.state_dict(), f'trained_models/AE-TD3_decoder_gripper.pht')
        print("models have been saved...")

    def plot_results(self, rewards, distance, check_point=True):
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 3, 1)
        plt.title("Reward Raw Curve")
        plt.plot(rewards)

        plt.subplot(1, 3, 2)
        plt.title("Final Distance to Goal")
        plt.plot(distance)

        plt.subplot(1, 3, 3)
        plt.title("AE Loss Curve")
        plt.plot(self.ae_loss_record)

        if check_point:
            plt.savefig(f"plot_results/AE-TD3_gripper_check_point_image_include_goal_{self.include_goal_angle_on}.png")
        else:
            plt.savefig(f"plot_results/AE-TD3_gripper_reward_curve_include_goal_{self.include_goal_angle_on}.png")
            np.savetxt(f"plot_results/AE-TD3_gripper_reward_curve_include_goal_{self.include_goal_angle_on}.txt", rewards)
            np.savetxt(f"plot_results/AE-TD3_gripper_distance_curve_include_goal_{self.include_goal_angle_on}.txt", distance)
            np.savetxt(f"plot_results/AE-TD3_gripper_ae_loss_curve_include_goal_{self.include_goal_angle_on}.txt", self.ae_loss_record)


def define_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--include_goal_angle_on', type=bool, default=True)
    parser.add_argument('--camera_index', type=int, default=2)
    parser.add_argument('--usb_index',    type=int, default=0)
    parser.add_argument('--robot_index',  type=str, default='robot-1')
    parser.add_argument('--replay_max_size',  type=int, default=20_000)

    parser.add_argument('--num_exploration_episodes', type=int, default=1000)
    parser.add_argument('--num_training_episodes',    type=int, default=4000)
    parser.add_argument('--episode_horizont',         type=int, default=20)

    args   = parser.parse_args()
    return args

def main_run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args   = define_parse_args()
    # todo define a seed value here https://github.com/denisyarats/pytorch_sac_ae/blob/master/train.py#:~:text=utils.set_seed_everywhere(args.seed)

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
    total_reward = []
    episode_distance_to_goal = []
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
            new_state_images, reward, done, distance, original_img, valve_angle = frames_stack.step(action, goal_angle)
            memory.save_experience_to_buffer(state_images, action, reward, new_state_images, done, goal_angle)
            state_images = new_state_images
            episode_reward += reward
            distance_to_goal = distance
            env.render(original_img, step, episode, valve_angle, goal_angle, done)
            if done:
                print("done ---> TRUE, breaking loop, end of this episode")
                break

            agent.update_function()
        total_reward.append(episode_reward)
        episode_distance_to_goal.append(distance_to_goal)

        print(f"Episode {episode} End, Total reward: {episode_reward}, Final Distance to Goal: {distance_to_goal} \n")
        if episode % 100 == 0:
            agent.plot_results(total_reward, episode_distance_to_goal, check_point=True)

    agent.save_models()
    agent.plot_results(total_reward, episode_distance_to_goal, check_point=False)



if __name__ == '__main__':
    main_run()
