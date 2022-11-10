
"""
Description:
            VAE-Using test bed camera
            input image size = 1024 * 960
            input NN size    = 128 * 128
            latent vector    = 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import cv2
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from gripper_function_utilities import Utilities
from gripper_environment_utilities import RL_ENV

from memory_utilities import MemoryClass
from networks_architectures import VanillaVAE, ForwardModelPrediction, Actor, Critic


class ReductionLearning:
    def __init__(self, params):

        self.camera_index = params['camera_index']
        self.usb_index    = params['usb_index']
        self.robot_index  = params['robot_index']

        # values for loops
        self.G = 10
        self.N = 10

        self.batch_size          = 32
        self.minimal_buffer_size = 256  # to start sampling value from buffer
        self.latent_vector_size  = 16

        self.z_in_size  = self.latent_vector_size + 2
        self.num_action = 4

        self.gamma = 0.99
        self.tau   = 0.005
        self.update_counter     = 0
        self.policy_freq_update = 2

        self.exploration_episodes = 10_000  # todo do i need this ?

        self.max_memory_size = 80_000

        self.device = Utilities().detect_device()
        self.memory = MemoryClass(self.max_memory_size)
        self.env = RL_ENV(self.camera_index, self.usb_index)

        # ---- Initialization and build VAE Model --- #
        self.learning_rate_vae = 0.0001
        self.vae = VanillaVAE(self.latent_vector_size).to(self.device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate_vae)
        self.vae_loss = []

        # ---- Initialization and build Forward Predictive Model --- #
        self.learning_rate_forward = 0.001
        self.forward_prediction_model = ForwardModelPrediction().to(self.device)
        self.forward_prediction_optimizer = optim.Adam(self.forward_prediction_model.parameters(), lr=self.learning_rate_forward)
        self.forward_prediction_loss = []

        # ---- Initialization and build Actor Critic Models --- #
        # Main networks
        self.actor     = Actor(self.z_in_size, self.num_action).to(self.device)
        self.critic_q1 = Critic(self.z_in_size, self.num_action).to(self.device)
        self.critic_q2 = Critic(self.z_in_size, self.num_action).to(self.device)

        # Target networks
        self.actor_target     = copy.deepcopy(self.actor)
        self.critic_target_q1 = copy.deepcopy(self.critic_q1)
        self.critic_target_q2 = copy.deepcopy(self.critic_q2)

        self.actor_learning_rate  = 1e-4
        self.critic_learning_rate = 1e-3

        self.actor_optimizer    = optim.Adam(self.actor.parameters(),     lr=self.actor_learning_rate)
        self.critic_optimizer_1 = optim.Adam(self.critic_q1.parameters(), lr=self.critic_learning_rate)
        self.critic_optimizer_2 = optim.Adam(self.critic_q2.parameters(), lr=self.critic_learning_rate)

        # empty vectors to store values
        self.novelty_values  = []
        self.surprise_values = []


    def learn_vae_model_function(self):
        # Full reconstruction VAE model
        # input random sampled batch of preprocessed images (batch, 3, 128, 128)
        if len(self.memory.memory_buffer) <= self.minimal_buffer_size:
            return
        else:
            self.vae.train()

            # sample from memory a batch but care about image-only
            img_states, _, _, _, _, _, _, _ = self.memory.sample_full_experiences(self.batch_size)

            img_states = np.array(img_states)
            img_states = torch.FloatTensor(img_states)  # change to tensor
            img_states = img_states.permute(0, 3, 1, 2)  # just put in the right order [b, 3, 128, 128]
            img_states = img_states.to(self.device)  # send batch to GPU

            x_rec, mu, log_var, z = self.vae.forward(img_states)

            # ---------------- Loss Function Reconstruction + KL --------------------------#
            # rec_loss = F.mse_loss(x_rec, img_states, reduction="sum")
            rec_loss = F.binary_cross_entropy(x_rec, img_states, reduction="sum")
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            total_loss = rec_loss + kld_loss
            # ------------------------------------------------------------------------------#
            self.vae_optimizer.zero_grad()
            total_loss.backward()
            self.vae_optimizer.step()

            print("VAE training Loss:", total_loss.item(), rec_loss.item(), kld_loss.item())
            self.vae_loss.append(total_loss.item())

            # --------------------------------------------------------------------------------

    def learn_predictive_model_function(self):
        if len(self.memory.memory_buffer) <= self.minimal_buffer_size:
            return
        else:
            self.forward_prediction_model.train()
            self.vae.eval()

            img_states, actions, _, img_next_states, done,  _, _, _ = self.memory.sample_full_experiences(self.batch_size)

            img_states = np.array(img_states)
            img_states = torch.FloatTensor(img_states)  # change to tensor
            img_states = img_states.permute(0, 3, 1, 2)  # just put in the right order [b, 3, 128, 128]
            img_states = img_states.to(self.device)  # send batch to GPU

            actions = np.array(actions)
            actions = torch.FloatTensor(actions)
            actions = actions.to(self.device)  # send batch to GPU

            img_next_states = np.array(img_next_states)
            img_next_states = torch.FloatTensor(img_next_states)
            img_next_states = img_next_states.permute(0, 3, 1, 2)
            img_next_states = img_next_states.to(self.device)  # send batch to GPU

        with torch.no_grad():
            _, _, _, z_input  = self.vae.forward(img_states)
            _, _, _, z_target = self.vae.forward(img_next_states)

        distribution_probability_model = self.forward_prediction_model.forward(z_input, actions)
        loss_neg_log_likelihood = - distribution_probability_model.log_prob(z_target)
        loss_neg_log_likelihood = torch.mean(loss_neg_log_likelihood)

        self.forward_prediction_optimizer.zero_grad()
        loss_neg_log_likelihood.backward()
        self.forward_prediction_optimizer.step()

        print("Forward Prediction Loss:", loss_neg_log_likelihood.item())
        self.forward_prediction_loss.append(loss_neg_log_likelihood.item())


    def policy_learning_function(self):
        if len(self.memory.memory_buffer) <= self.minimal_buffer_size:
            return
        else:
            self.vae.eval()
            self.actor.train()
            self.critic_q1.train()
            self.critic_q2.train()

            for it in range(1, self.G+1):
                self.update_counter += 1  # this is used for delay/

                img_states, actions, rewards, img_next_states, dones, \
                state_valve_angles, next_valve_angles, target_angles = self.memory.sample_full_experiences(self.batch_size)

                img_states = np.array(img_states)
                img_states = torch.FloatTensor(img_states)  # change to tensor
                img_states = img_states.permute(0, 3, 1, 2)  # just put in the right order [b, 3, 128, 128]
                img_states = img_states.to(self.device)  # send batch to GPU

                actions = np.array(actions)
                actions = torch.FloatTensor(actions)
                actions = actions.to(self.device)  # send batch to GPU

                img_next_states = np.array(img_next_states)
                img_next_states = torch.FloatTensor(img_next_states)
                img_next_states = img_next_states.permute(0, 3, 1, 2)
                img_next_states = img_next_states.to(self.device)  # send batch to GPU

                rewards = np.array(rewards).reshape(-1, 1)
                rewards = torch.FloatTensor(rewards)
                rewards = rewards.to(self.device)  # send batch to GPU

                dones = np.array(dones).reshape(-1, 1)
                dones = torch.FloatTensor(dones)
                dones = dones.to(self.device)  # send batch to GPU

                state_valve_angles = np.array(state_valve_angles).reshape(-1, 1)
                state_valve_angles = torch.FloatTensor(state_valve_angles)
                state_valve_angles = state_valve_angles.to(self.device)

                next_valve_angles = np.array(next_valve_angles).reshape(-1, 1)
                next_valve_angles = torch.FloatTensor(next_valve_angles)
                next_valve_angles = next_valve_angles.to(self.device)  # send batch to GPU

                target_angles = np.array(target_angles).reshape(-1, 1)
                target_angles = torch.FloatTensor(target_angles)
                target_angles = target_angles.to(self.device)  # send batch to GPU

                # Calculate the encode image representation
                with torch.no_grad():
                    _, _, _, z_state      = self.vae.forward(img_states)
                    _, _, _, z_next_state = self.vae.forward(img_next_states)

                # Create the observation state-space
                # Observation space (encode_image vector, valve_angle, target_angle, novelty, surprise)
                # todo add novelty, surprise

                state_space      = torch.cat([z_state, state_valve_angles, target_angles], dim=1)
                next_state_space = torch.cat([z_next_state, next_valve_angles, target_angles], dim=1)

                with torch.no_grad():
                    next_actions = self.actor_target(next_state_space)
                    target_noise = 0.2 * torch.randn_like(next_actions)
                    target_noise = target_noise.clamp_(-0.5, 0.5)
                    next_actions = next_actions + target_noise
                    next_actions = next_actions.clamp_(-1, 1)

                    next_q_values_q1 = self.critic_target_q1.forward(next_state_space, next_actions)
                    next_q_values_q2 = self.critic_target_q2.forward(next_state_space, next_actions)
                    q_min = torch.minimum(next_q_values_q1, next_q_values_q2)

                    Q_target = rewards + (self.gamma * (1 - dones) * q_min)

                Q_vals_q1 = self.critic_q1.forward(state_space, actions)
                Q_vals_q2 = self.critic_q2.forward(state_space, actions)

                critic_loss_1 = F.mse_loss(Q_vals_q1, Q_target)
                critic_loss_2 = F.mse_loss(Q_vals_q2, Q_target)

                # Critic step Update
                self.critic_optimizer_1.zero_grad()
                critic_loss_1.backward()
                self.critic_optimizer_1.step()

                self.critic_optimizer_2.zero_grad()
                critic_loss_2.backward()
                self.critic_optimizer_2.step()

                # Delayed policy updates
                # TD3 updates the policy (and target networks) less frequently than the Q-function

                if self.update_counter % self.policy_freq_update == 0:
                    # ------- calculate the actor loss
                    actor_loss = - self.critic_q1.forward(state_space, self.actor.forward(state_space)).mean()

                    # Actor step Update
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # ------------------------------------- Update target networks --------------- #
                    for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                    for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                    for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def get_action_from_policy(self, state_img_pixels, valve_angle, target):
        state_image_tensor  = torch.FloatTensor(state_img_pixels)
        valve_angle_tensor  = torch.from_numpy(np.array(valve_angle).astype(np.float32))
        target_angle_tensor = torch.from_numpy(np.array(target).astype(np.float32))

        state_image_tensor  = state_image_tensor.unsqueeze(0)
        state_image_tensor  = state_image_tensor.permute(0, 3, 1, 2).to(self.device)
        valve_angle_tensor  = valve_angle_tensor.unsqueeze(0).to(self.device)
        target_angle_tensor = target_angle_tensor.unsqueeze(0).to(self.device)

        self.actor.eval()
        self.vae.eval()
        with torch.no_grad():
            _, _, _, z_state  = self.vae.forward(state_image_tensor)
            state_space_input = torch.cat([z_state[0], valve_angle_tensor, target_angle_tensor])
            state_space_input = state_space_input.unsqueeze(0)
            action = self.actor.forward(state_space_input)
            action = action.cpu().data.numpy()
        return action[0]


    def calculate_novelty(self):
        pass

    def calculate_surprise(self):
        pass

    def learn_all_online(self):
        if len(self.memory.memory_buffer) <= self.minimal_buffer_size:
            return
        else:
            img_states, actions, rewards, img_next_states, dones, \
            state_valve_angles, next_valve_angles, target_angles = self.memory.sample_full_experiences(self.batch_size)

            img_states = np.array(img_states)
            img_states = torch.FloatTensor(img_states)  # change to tensor
            img_states = img_states.permute(0, 3, 1, 2)  # just put in the right order [b, 3, 128, 128]
            img_states = img_states.to(self.device)  # send batch to GPU

            actions = np.array(actions)
            actions = torch.FloatTensor(actions)
            actions = actions.to(self.device)  # send batch to GPU

            img_next_states = np.array(img_next_states)
            img_next_states = torch.FloatTensor(img_next_states)
            img_next_states = img_next_states.permute(0, 3, 1, 2)
            img_next_states = img_next_states.to(self.device)  # send batch to GPU

            rewards = np.array(rewards).reshape(-1, 1)
            rewards = torch.FloatTensor(rewards)
            rewards = rewards.to(self.device)  # send batch to GPU

            dones = np.array(dones).reshape(-1, 1)
            dones = torch.FloatTensor(dones)
            dones = dones.to(self.device)  # send batch to GPU

            state_valve_angles = np.array(state_valve_angles).reshape(-1, 1)
            state_valve_angles = torch.FloatTensor(state_valve_angles)
            state_valve_angles = state_valve_angles.to(self.device)

            next_valve_angles = np.array(next_valve_angles).reshape(-1, 1)
            next_valve_angles = torch.FloatTensor(next_valve_angles)
            next_valve_angles = next_valve_angles.to(self.device)  # send batch to GPU

            target_angles = np.array(target_angles).reshape(-1, 1)
            target_angles = torch.FloatTensor(target_angles)
            target_angles = target_angles.to(self.device)  # send batch to GPU

            #  =========================== train vae model ======================================
            self.vae.train()
            x_state_rec, mu_state, log_var_state, z_input = self.vae.forward(img_states)
            x_next_rect, mu_next, log_var_next, z_target  = self.vae.forward(img_next_states)

            # --Loss Function Reconstruction + KL -
            # rec_loss = F.mse_loss(x_rec, img_states, reduction="sum")
            rec_loss = F.binary_cross_entropy(x_state_rec, img_states, reduction="sum")
            kld_loss = -0.5 * torch.sum(1 + log_var_state - mu_state.pow(2) - log_var_state.exp())
            total_loss_vae = rec_loss + kld_loss

            self.vae_optimizer.zero_grad()
            total_loss_vae.backward()
            self.vae_optimizer.step()

            print("VAE training Loss:", total_loss_vae.item(), rec_loss.item(), kld_loss.item())
            self.vae_loss.append(total_loss_vae.item())
            # ------------------------------------------------------------------------------#

            '''
            #  =========================== train forward model ======================================
            self.forward_prediction_model.train()

            z_in   = z_input.detach()
            z_next = z_target.detach()  # all good here no grands

            distribution_probability_model = self.forward_prediction_model.forward(z_in, actions)
            loss_neg_log_likelihood = - distribution_probability_model.log_prob(z_next)
            loss_neg_log_likelihood = torch.mean(loss_neg_log_likelihood)

            self.forward_prediction_optimizer.zero_grad()
            loss_neg_log_likelihood.backward()
            self.forward_prediction_optimizer.step()

            print("Forward Prediction Loss:", loss_neg_log_likelihood.item())
            self.forward_prediction_loss.append(loss_neg_log_likelihood.item())
            # ------------------------------------------------------------------------------#
            '''

            #  =========================== train actor critic model ===================================
            z_in   = z_input.detach()
            z_next = z_target.detach()  # all good here no grands

            self.actor.train()
            self.critic_q1.train()
            self.critic_q2.train()

            state_space      = torch.cat([z_in, state_valve_angles, target_angles], dim=1)
            next_state_space = torch.cat([z_next, next_valve_angles, target_angles], dim=1)

            with torch.no_grad():
                next_actions = self.actor_target(next_state_space)
                target_noise = 0.2 * torch.randn_like(next_actions)
                target_noise = target_noise.clamp_(-0.5, 0.5)
                next_actions = next_actions + target_noise
                next_actions = next_actions.clamp_(-1, 1)

                next_q_values_q1 = self.critic_target_q1.forward(next_state_space, next_actions)
                next_q_values_q2 = self.critic_target_q2.forward(next_state_space, next_actions)
                q_min = torch.minimum(next_q_values_q1, next_q_values_q2)

                Q_target = rewards + (self.gamma * (1 - dones) * q_min)

            Q_vals_q1 = self.critic_q1.forward(state_space, actions)
            Q_vals_q2 = self.critic_q2.forward(state_space, actions)

            critic_loss_1 = F.mse_loss(Q_vals_q1, Q_target)
            critic_loss_2 = F.mse_loss(Q_vals_q2, Q_target)

            # Critic step Update
            self.critic_optimizer_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer_1.step()

            self.critic_optimizer_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optimizer_2.step()

            # Delayed policy updates
            # TD3 updates the policy (and target networks) less frequently than the Q-function

            if self.update_counter % self.policy_freq_update == 0:
                # ------- calculate the actor loss
                actor_loss = - self.critic_q1.forward(state_space, self.actor.forward(state_space)).mean()

                # Actor step Update
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ------------------------------------- Update target networks --------------- #
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


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


    def save_models(self):
        torch.save(self.actor.state_dict(), f'trained_models/{self.robot_index}_TD3_actor.pht')
        torch.save(self.vae.state_dict(), f'trained_models/{self.robot_index}_vae_model_gripper.pth')
        torch.save(self.forward_prediction_model.state_dict(), f'trained_models/{self.robot_index}_forward_model_gripper.pth')
        print("models have been saved...")

    def update_models(self):

        #self.learn_vae_model_function()
        #self.learn_predictive_model_function()
        #self.policy_learning_function()

        # all in a single function
        self.learn_all_online()


    def rl_idea_training(self, horizontal_steps=20, num_episodes=1000):
        rewards = []
        episode_distance_to_goal = []

        for episode in range(1, num_episodes+1):
            self.env.env_reset()
            episode_reward   = 0
            distance_to_goal = 0

            state_raw_image = self.env.vision_config.get_camera_image()
            state_image     = self.env.vision_config.pre_pro_image(state_raw_image)

            target_angle    = self.env.define_goal_angle()
            valve_angle     = self.env.get_valve_angle()

            for step in range(1, horizontal_steps+1):
                #action = self.get_action_from_policy(state_image, valve_angle, target_angle)
                action = self.env.generate_sample_act()
                noise  = np.random.normal(0, scale=0.15, size=4)
                action = action + noise
                action = np.clip(action, -1, 1)

                self.env.env_step(action)

                new_state_raw_image = self.env.vision_config.get_camera_image()
                new_state_image     = self.env.vision_config.pre_pro_image(new_state_raw_image)
                new_valve_angle     = self.env.get_valve_angle()

                ext_reward, done, distance = self.env.calculate_extrinsic_reward(target_angle, new_valve_angle)

                self.memory.save_full_experience_buffer(state_image, action, ext_reward, new_state_image, done, valve_angle, new_valve_angle, target_angle)

                episode_reward += ext_reward
                distance_to_goal = distance  # just need the final distance at the need of the episode

                state_image = new_state_image
                valve_angle = new_valve_angle

                self.env.render(new_state_raw_image, step, episode, new_valve_angle, target_angle, done)

                if done:
                    print("done TRUE, breaking loop, end of this episode")
                    break

                self.update_models()

            rewards.append(episode_reward)
            episode_distance_to_goal.append(distance_to_goal)
            print(f"Episode {episode} End, Total reward: {episode_reward}, Distance to Goal: {distance_to_goal}")

            if episode % 100 == 0:
                check_point = True
                self.plot_functions(rewards, episode_distance_to_goal, check_point)

        self.save_models()
        check_point = False
        self.plot_functions(rewards, episode_distance_to_goal, check_point)



    def vae_evaluation(self):
        Utilities().load_vae_model(self.vae)
        state_image = self.env.vision_config.get_camera_image()
        state_image = self.env.vision_config.pre_pro_image(state_image)

        self.vae.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_image)
            state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.permute(0, 3, 1, 2).to(self.device)
            x_rec, _, _, _ = self.vae.forward(state_tensor)
            x_rec = x_rec.permute(0, 2, 3, 1)
            x_rec = x_rec.cpu().numpy()

        plt.subplot(1, 2, 1)  # row 1, col 2 index 1
        plt.title("Input")
        #plt.imshow(state_image)
        plt.imshow((state_image * 255).astype(np.uint8))
        plt.subplot(1, 2, 2)  # index 2
        plt.title("Reconstruction")
        #plt.imshow(x_rec[0])
        plt.imshow((x_rec[0] * 255).astype(np.uint8))
        plt.show()


def main_run():
    parser = ArgumentParser()
    parser.add_argument('--camera_index', type=int, default=2)
    parser.add_argument('--usb_index', type=int, default=0)
    parser.add_argument('--robot_index', type=str, default='robot-1')
    args = parser.parse_args()
    params = vars(args)

    model = ReductionLearning(params)
    model.rl_idea_training()
    #model.vae_evaluation()


if __name__ == '__main__':
    main_run()
