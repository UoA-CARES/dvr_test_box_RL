
import cv2
import gym
import copy
import math

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F

from memory_utilities import MemoryClass
from networks_architectures import SoftQNetworkSAC, PolicyNetworkSACDeterministic, VanillaVAE

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Working with GPU")
else:
    device = torch.device('cpu')
    print("Working with CPU")



class VAE_SAC_AGENT:
    def __init__(self):

        self.device = device
        self.batch_size = 32
        self.gamma = 0.99
        self.tau   = 0.005

        self.G = 10


        self.latent_vector_size = 64
        self.z_in_size  = self.latent_vector_size
        self.num_action = 1

        self.update_counter = 0
        self.freq_update    = 2

        self.noise = torch.Tensor(self.num_action).to(self.device) # tensor to save noise

        # ---- Initialization replay memory --- #
        self.max_memory_size = 10_000
        self.memory = MemoryClass(self.max_memory_size)

        self.lr_vae = 1e-3
        self.vae    = VanillaVAE(self.latent_vector_size).to(self.device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=self.lr_vae)

        self.soft_q_lr  = 1e-3
        self.soft_q_net = SoftQNetworkSAC(self.z_in_size, self.num_action).to(self.device)
        self.soft_q_net_target = copy.deepcopy(self.soft_q_net)
        self.soft_q_optimizer  = optim.Adam(self.soft_q_net.parameters(), lr=self.soft_q_lr)

        self.policy_lr  = 1e-3
        self.policy_net = PolicyNetworkSACDeterministic(self.z_in_size, self.num_action).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)



    def prepare_tensor(self, img_states, actions, rewards, img_next_states, dones):
        img_states = np.array(img_states)
        img_states = torch.FloatTensor(img_states)  # change to tensor, torch.Size([b, 1, 128, 128])
        img_states = img_states.to(self.device)  # send batch to GPU

        actions = np.array(actions)
        actions = torch.FloatTensor(actions)
        actions = actions.to(self.device)  # send batch to GPU

        img_next_states = np.array(img_next_states)
        img_next_states = torch.FloatTensor(img_next_states)
        img_next_states = img_next_states.to(self.device)  # send batch to GPU

        rewards = np.array(rewards).reshape(-1, 1)
        rewards = torch.FloatTensor(rewards)
        rewards = rewards.to(self.device)  # send batch to GPU

        dones = np.array(dones).reshape(-1, 1)
        dones = torch.FloatTensor(dones)
        dones = dones.to(self.device)  # send batch to GPU

        return img_states, actions, rewards, img_next_states, dones

    def sample_from_policy_deterministic(self, state):
        mean   = self.policy_net.forward(state)
        noise  = self.noise.normal_(0., std=0.1)
        noise  = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def learn_vae_function(self, img_states, actions, rewards, img_next_states, dones):
        x_rec, mu, log_var, z = self.vae.forward(img_states)

        rec_loss   = F.binary_cross_entropy(x_rec, img_states, reduction="sum")
        kld_loss   = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = rec_loss + kld_loss

        self.vae_optimizer.zero_grad()
        total_loss.backward()
        self.vae_optimizer.step()
        #print("VAE training Loss:", total_loss.item())


    def learn_rl_function(self, img_states, actions, rewards, img_next_states, dones):
        for it in range(1, self.G+1):
            self.update_counter += 1  # this is used for delay

            x_rec, mu, log_var, z_current_state = self.vae.forward(img_states)  # still gradient here

            with torch.no_grad():
                x_next_rec, _, _, z_next_state = self.vae.forward(img_next_states)

                next_state_action, next_state_log_pi, _ = self.sample_from_policy_deterministic(z_next_state)
                qf1_next_target, qf2_next_target        = self.soft_q_net_target(z_next_state, next_state_action)

                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value       = rewards + (1 - dones) * self.gamma * min_qf_next_target

            qf1, qf2 = self.soft_q_net(z_current_state, actions)
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
            qf_loss  = qf1_loss + qf2_loss

            self.soft_q_optimizer.zero_grad()
            qf_loss.backward()
            self.soft_q_optimizer.step()

            z_policy       = z_current_state.detach()
            pi, log_pi, _  = self.sample_from_policy_deterministic(z_policy)
            qf1_pi, qf2_pi = self.soft_q_net(z_policy, pi)
            min_qf_pi      = torch.min(qf1_pi, qf2_pi)

            policy_loss = - min_qf_pi.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            if self.update_counter % self.freq_update == 0:
                for target_param, param in zip(self.soft_q_net_target.parameters(), self.soft_q_net.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)



    def update_function(self):
        if len(self.memory.memory_buffer_experiences) <= self.batch_size:
            return
        else:
            img_states_np, actions_np, rewards_np, img_next_states_np, dones_np = self.memory.sample_frame_vector_experiences(self.batch_size)
            img_states_tn, actions_tn, rewards_tn, img_next_states_tn, dones_tn = self.prepare_tensor(img_states_np, actions_np, rewards_np, img_next_states_np, dones_np)

            self.learn_vae_function(img_states_tn, actions_tn, rewards_tn, img_next_states_tn, dones_tn)
            self.learn_rl_function(img_states_tn, actions_tn, rewards_tn, img_next_states_tn, dones_tn)




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
    for _ in tqdm(range(1, num_exploration_episodes + 1)):
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


def run_training_rl_method(env, agent, num_episodes_training, episode_horizont):
    rewards = []
    for episode in range(1, num_episodes_training + 1):
        print(f"-----------------Episode {episode}-----------------------------")
        episode_reward = 0
        env.reset()
        state_image = env.render(mode='rgb_array')
        state_image = pre_pro_image(state_image)
        for step in range(1, episode_horizont + 1):
            action = env.action_space.sample()
            #action = agent.get_action_from_policy(state_image)
            noise  = np.random.normal(0, scale=0.1, size=1)
            action = action + noise
            action = np.clip(action, -2, 2)
            obs_next_state_vector, reward, done, info = env.step(action)
            new_state_image = env.render(mode='rgb_array')
            new_state_image = pre_pro_image(new_state_image)
            agent.memory.save_frame_vector_experience_buffer(state_image, action, reward, new_state_image, done)
            state_image = new_state_image

            episode_reward += reward
            if done:
                break
            agent.update_function()

        rewards.append(episode_reward)
        print(f"Episode {episode} End, Total reward: {episode_reward}")



def main_run():
    env   = gym.make('Pendulum-v1')
    agent = VAE_SAC_AGENT()

    episode_horizont         = 200
    num_exploration_episodes = 10
    num_training_episodes    = 2000

    #run_random_exploration(env, agent, num_exploration_episodes, episode_horizont)
    run_training_rl_method(env, agent, num_training_episodes, episode_horizont)
    env.close()


if __name__ == '__main__':
    main_run()