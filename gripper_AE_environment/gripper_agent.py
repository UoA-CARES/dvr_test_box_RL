
import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from gripper_architectures import Actor, Critic, Decoder


class Td3Agent:
    def __init__(self, env, robot_index, device, memory_buffer, include_goal_angle_on, batch_size):

        # ---------------- parameters  -------------------------#
        self.env         = env
        self.robot_index = robot_index
        self.device      = device
        self.memory      = memory_buffer
        self.action_dim  = 4

        # ---------------- Hyperparameters  -------------------------#
        self.encoder_lr = 1e-3
        self.decoder_lr = 1e-3
        self.critic_lr  = 1e-3
        self.actor_lr   = 1e-4

        self.gamma       = 0.99
        self.tau         = 0.005
        self.tau_encoder = 0.001

        self.G                  = 10
        self.update_counter     = 0
        self.policy_freq_update = 2
        self.batch_size         = batch_size

        # ---------------- Extras  -------------------------#
        self.ae_loss_record = []
        self.include_goal_angle_on = include_goal_angle_on

        # by include_goal_angle_on True, the  target angle is concatenated with the latent vector
        if self.include_goal_angle_on:
            self.latent_dim = 50
            self.input_dim  = 51  # 50 for latent size + 1 for target value
        else:
            self.latent_dim = 50
            self.input_dim  = 50

        # ----------------- Networks ------------------------------#
        # main networks
        self.actor  = Actor(self.latent_dim, self.input_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.latent_dim, self.input_dim, self.action_dim).to(self.device)

        # target networks
        self.actor_target  = Actor(self.latent_dim, self.input_dim, self.action_dim).to(self.device)
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
        torch.save(self.actor.state_dict(), f'trained_models/AE-TD3_actor_gripper_{self.include_goal_angle_on}.pht')
        torch.save(self.critic.encoder_net.state_dict(), f'trained_models/AE-TD3_encoder_gripper_{self.include_goal_angle_on}.pht')
        torch.save(self.decoder.state_dict(), f'trained_models/AE-TD3_decoder_gripper_{self.include_goal_angle_on}.pht')
        print("models have been saved...")

    def plot_results(self, rewards, distance, check_point=True):
        plt.figure(figsize=(20, 10))

        plt.subplot(3, 1, 1)
        plt.title("Reward Raw Curve")
        plt.plot(rewards)

        plt.subplot(3, 1, 2)
        plt.title("Final Distance to Goal")
        plt.plot(distance)

        plt.subplot(3, 1, 3)
        plt.title("AE Loss Curve")
        plt.plot(self.ae_loss_record)

        if check_point:
            plt.savefig(f"plot_results/AE-TD3_gripper_check_point_image_include_goal_{self.include_goal_angle_on}.png")
            np.savetxt(f"plot_results/AE-TD3_gripper_check_point_reward_curve_include_goal_{self.include_goal_angle_on}.txt", rewards)
            np.savetxt(f"plot_results/AE-TD3_gripper_check_point_distance_curve_include_goal_{self.include_goal_angle_on}.txt", distance)
            np.savetxt(f"plot_results/AE-TD3_gripper_check_point_ae_loss_curve_include_goal_{self.include_goal_angle_on}.txt", self.ae_loss_record)
        else:
            plt.savefig(f"plot_results/AE-TD3_gripper_reward_curve_include_goal_{self.include_goal_angle_on}.png")
            np.savetxt(f"plot_results/AE-TD3_gripper_reward_curve_include_goal_{self.include_goal_angle_on}.txt", rewards)
            np.savetxt(f"plot_results/AE-TD3_gripper_distance_curve_include_goal_{self.include_goal_angle_on}.txt", distance)
            np.savetxt(f"plot_results/AE-TD3_gripper_ae_loss_curve_include_goal_{self.include_goal_angle_on}.txt", self.ae_loss_record)