
import cv2
import gym
import copy
import math

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

import torch
import torch.optim as optim
import torch.nn.functional as F

from memory_utilities import MemoryClass
from networks_architectures import VanillaVAE, SpecialCritic, Actor


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Working with GPU")
else:
    device = torch.device('cpu')
    print("Working with CPU")


class VAE_RL_AGENT:
    def __init__(self):

        self.device = device

        # -------- Hyper-parameters --------------- #
        self.batch_size          = 32
        self.minimal_buffer_size = 256

        self.gamma = 0.99
        self.tau   = 0.005
        self.update_counter     = 0
        self.policy_freq_update = 2
        self.latent_vector_size = 128

        self.z_in_size  = self.latent_vector_size
        self.num_action = 1

        # ---- Initialization replay memory --- #
        self.max_memory_size = 10_000
        self.memory = MemoryClass(self.max_memory_size)

        # ---- Initialization and build VAE Model --- #
        self.lr_actor   = 3e-4
        self.lr_critic  = 3e-3
        self.lr_vae     = 3e-3

        # ---- Initialization Models --- #
        self.vae      = VanillaVAE(self.latent_vector_size).to(self.device)
        self.actor    = Actor(self.z_in_size, self.num_action).to(self.device)
        self.critic   = SpecialCritic(self.z_in_size, self.num_action).to(self.device)

        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.vae_optimizer    = optim.Adam(self.vae.parameters(),    lr=self.lr_vae)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.N = 1   # internal loop for training VAE
        self.G = 1  # internal loop for Policy update

        self.encoder_loss = []

    def learn_encoder_function(self, img_states, actions, rewards, img_next_states, dones):
        self.vae.train()

        for _ in range(1, self.N+1):

            x_rec, mu, log_var, z = self.vae.forward(img_states)

            # ---------------- Loss Function Reconstruction + KL --------------------------#

            #rec_loss = F.mse_loss(x_rec, img_states, reduction="sum")
            rec_loss   = F.binary_cross_entropy(x_rec, img_states, reduction="sum")
            kld_loss   = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            total_loss = rec_loss + kld_loss
            # ------------------------------------------------------------------------------#
            self.vae_optimizer.zero_grad()
            total_loss.backward()
            self.vae_optimizer.step()

            #print("VAE training Loss:", total_loss.item(), rec_loss.item(), kld_loss.item())
            self.encoder_loss.append(total_loss.item())
            # --------------------------------------------------------------------------------

    def policy_learning_function(self, img_states, actions, rewards, img_next_states, dones):
        self.actor.train()
        self.critic.train()
        self.vae.train()

        for it in range(1, self.G+1):
            self.update_counter += 1  # this is used for delay

            x_rec, mu, log_var, z_current_state = self.vae.forward(img_states)

            with torch.no_grad():
                x_next_rec, _, _, z_next_state = self.vae.forward(img_next_states)
                next_actions = self.actor_target(z_next_state)
                target_noise = 0.2 * torch.randn_like(next_actions)
                target_noise = target_noise.clamp_(-0.5, 0.5)
                next_actions = next_actions + target_noise
                next_actions = next_actions.clamp_(-2, 2)

                next_q_values_q1, next_q_values_q2 = self.critic_target.forward(z_next_state, next_actions)
                q_min    = torch.minimum(next_q_values_q1, next_q_values_q2)
                q_target = rewards + (self.gamma * (1 - dones) * q_min)


            q_vals_q1, q_vals_q2  = self.critic.forward(z_current_state, actions)

            critic_loss_1     = F.mse_loss(q_vals_q1, q_target)
            critic_loss_2     = F.mse_loss(q_vals_q2, q_target)
            critic_loss_total = critic_loss_1 + critic_loss_2

            # Critic step Update
            self.critic_optimizer.zero_grad()
            critic_loss_total.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            # TD3 updates the policy (and target networks) less frequently than the Q-function
            if self.update_counter % self.policy_freq_update == 0:
                # ------- calculate the actor loss
                z_actor  = z_current_state.detach()  # detach for update the actor
                actor_q1, actor_q2  = self.critic.forward(z_actor, self.actor.forward(z_actor))
                actor_q_min = torch.minimum(actor_q1, actor_q2)
                actor_loss  = - actor_q_min.mean()

                # Actor step Update
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ------------------------------------- Update target networks --------------- #
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def prepare_tensor(self, img_states, actions, rewards, img_next_states, dones):
        img_states = np.array(img_states)
        img_states = torch.FloatTensor(img_states)  # change to tensor, torch.Size([b, 1, 128, 128])
        #img_states = img_states.permute(0, 3, 1, 2)  # just put in the right order [b, 3, 128, 128]
        img_states = img_states.to(self.device)  # send batch to GPU

        actions = np.array(actions)
        actions = torch.FloatTensor(actions)
        actions = actions.to(self.device)  # send batch to GPU

        img_next_states = np.array(img_next_states)
        img_next_states = torch.FloatTensor(img_next_states)
        #img_next_states = img_next_states.permute(0, 3, 1, 2)
        img_next_states = img_next_states.to(self.device)  # send batch to GPU

        rewards = np.array(rewards).reshape(-1, 1)
        rewards = torch.FloatTensor(rewards)
        rewards = rewards.to(self.device)  # send batch to GPU

        dones = np.array(dones).reshape(-1, 1)
        dones = torch.FloatTensor(dones)
        dones = dones.to(self.device)  # send batch to GPU

        return img_states, actions, rewards, img_next_states, dones


    def get_action_from_policy(self, state_img_pixels):
        state_image_tensor  = torch.FloatTensor(state_img_pixels)
        state_image_tensor  = state_image_tensor.unsqueeze(0)  # torch.Size([1, 1, 128, 128])
        #state_image_tensor  = state_image_tensor.permute(0, 3, 1, 2).to(self.device)  # torch.Size([1, 3, 128, 128])
        state_image_tensor  = state_image_tensor.to(self.device)

        #self.vae.eval()
        #self.actor.eval()
        with torch.no_grad():
            _, _, _, z_state  = self.vae.forward(state_image_tensor)
            state_space_input = z_state #.unsqueeze(0)
            action = self.actor.forward(state_space_input)
            action = action.cpu().data.numpy()
        return action[0]


    def test_ensamble(self):
        if len(self.memory.memory_buffer_experiences) <= self.batch_size:
            return
        else:
            self.update_counter += 1

            self.simple_encoder.train()
            self.actor.train()
            self.special_critic.train()

            img_states, actions, rewards, img_next_states, dones = self.memory.sample_frame_vector_experiences(self.batch_size)
            img_states, actions, rewards, img_next_states, dones = self.prepare_tensor(img_states, actions, rewards, img_next_states, dones)


            x_rec, z_current   = self.simple_encoder.forward(img_states)

            state_input_new = torch.cat([z_current, rewards], dim=1)

            with torch.no_grad():
                x_next_rec, z_next = self.simple_encoder.forward(img_next_states)
                next_actions = self.actor_target(z_next)
                target_noise = 0.2 * torch.randn_like(next_actions)
                target_noise = target_noise.clamp_(-0.5, 0.5)
                next_actions = next_actions + target_noise
                next_actions = next_actions.clamp_(-2, 2)

                next_q_values_q1, next_q_values_q2 = self.special_critic_target.forward(z_next, next_actions)
                q_min = torch.minimum(next_q_values_q1, next_q_values_q2)
                q_target = rewards + (self.gamma * (1 - dones) * q_min)

            q_vals_q1, q_vals_q2 = self.special_critic.forward(z_current, actions)

            critic_loss_1 = F.mse_loss(q_vals_q1, q_target)
            critic_loss_2 = F.mse_loss(q_vals_q2, q_target)

            critic_loss   = critic_loss_1 + critic_loss_2

            #encoder_loss  = F.mse_loss(x_rec, img_states)
            encoder_loss  = F.mse_loss(x_rec, img_states, reduction="sum")

            total_loss = critic_loss + encoder_loss


            self.encoder_critic_optimizer.zero_grad()
            total_loss.backward()
            self.encoder_critic_optimizer.step()

            if self.update_counter % self.policy_freq_update == 0:
                # ------- calculate the actor loss

                z_actor = z_current.detach()  # detach for update the actor
                actor_q1, actor_q2 = self.special_critic.forward(z_actor, self.actor.forward(z_actor))
                actor_q_min = torch.minimum(actor_q1, actor_q2)
                actor_loss = - actor_q_min.mean()

                # Actor step Update
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ------------------------------------- Update target networks --------------- #
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.special_critic_target.parameters(), self.special_critic.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def update_function(self):
        if len(self.memory.memory_buffer_experiences) <= self.batch_size:
            return
        else:
            img_states_np, actions_np, rewards_np, img_next_states_np, dones_np = self.memory.sample_frame_vector_experiences(self.batch_size)
            img_states_tn, actions_tn, rewards_tn, img_next_states_tn, dones_tn = self.prepare_tensor(img_states_np, actions_np, rewards_np, img_next_states_np, dones_np)

            self.learn_encoder_function(img_states_tn, actions_tn, rewards_tn, img_next_states_tn, dones_tn)
            self.policy_learning_function(img_states_tn, actions_tn, rewards_tn, img_next_states_tn, dones_tn)

    def plot_functions(self, rewards):
        plt.subplot(2, 2, 1)
        plt.title("VAE LOSS")
        plt.plot(self.encoder_loss)

        plt.subplot(2, 2, 2)
        plt.title("Rewards")
        plt.plot(rewards)

        plt.savefig(f"plot_results/vae_td3_pendulum_curves.png")


    def calculate_similarity(self, img_input):
        self.simple_encoder.eval()
        with torch.no_grad():
            img_tensor = torch.FloatTensor(img_input)
            img_tensor = img_tensor.unsqueeze(0).to(device)
            x_rec,  z = self.simple_encoder.forward(img_tensor)
            x_rec = x_rec[0].cpu().numpy()

        ssim_const = ssim(img_input, x_rec, data_range=img_input.max() - img_input.min(), channel_axis=0)

        return ssim_const


    def save_vae_model(self):
        torch.save(self.vae.state_dict(), "trained_models/vae_model.pth")
        print("VAE model has been saved")

    def load_vae_model(self):
        self.vae.load_state_dict(torch.load("trained_models/vae_model.pth"))
        print("VAE model has been loaded")



def pre_pro_image(image_array):
    crop_image = image_array[110:390, 110:390]
    resized    = cv2.resize(crop_image, (128, 128), interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #cv2.imshow("Normalized image", norm_image)
    #cv2.waitKey(10)
    state_image = np.expand_dims(norm_image, axis=0)  # (1, 128, 128)
    return state_image


def run_random_exploration(env, agent,  num_exploration_episodes, episode_horizont):
    print("exploration start")
    for episode in tqdm(range(1, num_exploration_episodes + 1)):
        env.reset()
        state_image = env.render(mode='rgb_array')  # return the rendered image and can be used as input-state image
        state_image = pre_pro_image(state_image)

        for step in range(1, episode_horizont + 1):
            action = env.action_space.sample()
            obs_next_state_vector, reward, done, _ = env.step(action)
            new_state_image = env.render(mode='rgb_array')
            new_state_image = pre_pro_image(new_state_image)
            agent.memory.save_frame_vector_experience_buffer(state_image, action, reward, new_state_image, done)
            state_image = new_state_image
            if done:
                break
    print("exploration end")



def run_training_rl_method(env, agent, num_episodes_training=100, episode_horizont=200):

    rewards = []
    for episode in range(1, num_episodes_training + 1):
        print(f"-----------------Episode {episode}-----------------------------")
        episode_reward   = 0
        env.reset()
        state_image = env.render(mode='rgb_array')  # return the rendered image so can be used as input-state image
        state_image = pre_pro_image(state_image)

        for step in range(1, episode_horizont + 1):
            action = agent.get_action_from_policy(state_image)
            noise  = np.random.normal(0, scale=0.1, size=1)
            action = action + noise
            action = np.clip(action, -2, 2)
            obs_next_state_vector, reward, done, info = env.step(action)
            new_state_image = env.render(mode='rgb_array')
            new_state_image = pre_pro_image(new_state_image)
            agent.memory.save_frame_vector_experience_buffer(state_image, action, reward, new_state_image, done)
            state_image      = new_state_image

            episode_reward += reward

            if done:
                break

            agent.update_function()

            # todo try this
            #similarity_state = agent.calculate_similarity(state_image)
            #if similarity_state >= 0.92:   # similarity_tolerance
                #agent.policy_learning_function()

        rewards.append(episode_reward)
        print(f"Episode {episode} End, Total reward: {episode_reward}")
        if episode % 50 == 0:
            agent.plot_functions(rewards)

    agent.plot_functions(rewards)
    agent.save_vae_model()


def evaluate_encoder_model(env, agent):
    env.reset()
    state_image = env.render(mode='rgb_array')
    state_image = state_image[110:390, 110:390]
    state_image = cv2.resize(state_image, (128, 128), interpolation=cv2.INTER_AREA)
    state_image = cv2.cvtColor(state_image, cv2.COLOR_BGR2GRAY)
    state_image = cv2.normalize(state_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    state_image_in = np.expand_dims(state_image, axis=0)  # (1, 128, 128)

    agent.load_vae_model()
    agent.vae.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_image_in)
        state_tensor = state_tensor.unsqueeze(0).to(device)
        x_rec, mu, log_var, z = agent.vae.forward(state_tensor)
        x_rec = x_rec.permute(0, 2, 3, 1)
        x_rec = x_rec.cpu().numpy()
    #ssim_const = ssim(state_image, x_rec[0], data_range=state_image.max() - state_image.min(), channel_axis=2)
    #print("Similarity Index:", ssim_const)

    plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    plt.title("Input")
    plt.imshow(state_image)
    plt.subplot(1, 2, 2)  # index 2
    plt.title("Reconstruction")
    plt.imshow(x_rec[0])
    plt.show()


def main_run():
    env = gym.make('Pendulum-v1')
    agent = VAE_RL_AGENT()
    episode_horizont = 200
    num_exploration_episodes = 100
    num_training_episodes    = 2000

    run_random_exploration(env, agent, num_exploration_episodes, episode_horizont)
    run_training_rl_method(env, agent, num_training_episodes, episode_horizont)
    #evaluate_encoder_model(env, agent)
    env.close()


if __name__ == '__main__':
    main_run()