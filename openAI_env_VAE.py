
import cv2
import gym
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

import torch
import torch.optim as optim
import torch.nn.functional as F

from memory_utilities import MemoryClass
from networks_architectures import VanillaVAE, Actor, Critic, VAE_Critic,  VAE_Critic_Ensemble, SimpleEncoder


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
        self.latent_vector_size = 8

        self.z_in_size  = self.latent_vector_size
        self.num_action = 1

        # ---- Initialization replay memory --- #
        self.max_memory_size = 50_000
        self.memory = MemoryClass(self.max_memory_size)

        # ---- Initialization and build VAE Model --- #
        self.lr_vae_critic = 1e-3
        self.lr_actor      = 1e-4

        self.vae = VanillaVAE(self.latent_vector_size).to(self.device)
        #self.simple_encoder = SimpleEncoder(self.latent_vector_size).to(self.device)

        self.critic = Critic(self.z_in_size, self.num_action).to(self.device)
        self.actor  = Actor(self.z_in_size, self.num_action).to(self.device)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target  = copy.deepcopy(self.actor)

        # self.encoder_critic_optimizer = optim.Adam(list(self.simple_encoder.parameters()) + list(self.critic.parameters()), lr=self.lr_vae_critic)
        self.vae_critic_optimizer = optim.Adam(list(self.vae.parameters()) + list(self.critic.parameters()), lr=self.lr_vae_critic)
        self.actor_optimizer      = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        self.vae_critic_loss = []
        #self.encoder_critic_loss = []

        # ---- Initialization and build CRITIC-VAE Model --- #
        # this idea is not clear yet
        '''
        self.vae       = VanillaVAE(self.latent_vector_size).to(self.device)
        self.critic_q1 = Critic(self.z_in_size, self.num_action).to(self.device)
        self.critic_q1_target = copy.deepcopy(self.critic_q1)

        self.vae_critic_ensemble = VAE_Critic_Ensemble(self.vae, self.critic_q1)
        self.vae_critic_optimizer = optim.Adam(self.vae_critic_ensemble.parameters(), lr=0.001)
        '''
        #for p1, p2 in zip(self.my_critic_q1.parameters(),  self.my_critic_q2.parameters()):
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%")
            #print(torch.equal(p1, p2))
            #print("=========================")


        # ---- Initialization and build Actor Critic Models --- #
        # Main networks
        #self.actor = Actor(self.z_in_size, self.num_action).to(self.device)
        #self.critic_q1 = Critic(self.z_in_size, self.num_action).to(self.device)
        #self.critic_q2 = Critic(self.z_in_size, self.num_action).to(self.device)

        # Target networks
        #self.actor_target     = copy.deepcopy(self.actor)
        #self.critic_target_q1 = copy.deepcopy(self.critic_q1)
        #self.critic_target_q2 = copy.deepcopy(self.critic_q2)

        # Target networks
        #self.actor_target     = Actor(self.z_in_size, self.num_action).to(self.device)
        #self.critic_target_q1 = Critic(self.z_in_size, self.num_action).to(self.device)
        #self.critic_target_q2 = Critic(self.z_in_size, self.num_action).to(self.device)

        # Initialization of the target networks as copies of the original networks
        #for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            #target_param.data.copy_(param.data)

        #for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
            #target_param.data.copy_(param.data)

        #for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
            #target_param.data.copy_(param.data)

        #self.actor_learning_rate  = 1e-4
        #self.critic_learning_rate = 1e-3

        #self.actor_optimizer    = optim.Adam(self.actor.parameters(),     lr=self.actor_learning_rate)
        #self.critic_optimizer_1 = optim.Adam(self.critic_q1.parameters(), lr=self.critic_learning_rate)
        #self.critic_optimizer_2 = optim.Adam(self.critic_q2.parameters(), lr=self.critic_learning_rate)

        self.N = 1  # internal loop for training VAE
        self.G = 1  # internal loop for Policy update


    def learn_encoder_function(self):
        if len(self.memory.memory_buffer_frames) <= self.minimal_buffer_size:
            return
        else:
            self.vae.train()
            for _ in range(1, self.N+1):
                # sample from memory a batch of previous image-only experiences
                img_states  = self.memory.sample_frames_experiences(self.batch_size)
                #img_states  = self.memory.sample_frame_vector_experiences(self.batch_size)

                img_states  = np.array(img_states)
                img_states  = torch.FloatTensor(img_states)   # change to tensor
                img_states  = img_states.permute(0, 3, 1, 2)  # just put in the right order [b, 3, 128, 128]
                img_states  = img_states.to(device)           # send batch to GPU

                x_rec, mu, log_var, z = self.vae.forward(img_states)

                # ---------------- Loss Function Reconstruction + KL --------------------------#
                #rec_loss = F.mse_loss(x_rec, img_states, reduction="sum")
                rec_loss = F.binary_cross_entropy(x_rec, img_states, reduction="sum")
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                total_loss = rec_loss + kld_loss
                # ------------------------------------------------------------------------------#
                self.vae.train()
                self.vae_optimizer.zero_grad()
                total_loss.backward()
                self.vae_optimizer.step()

                #print("VAE training Loss:", total_loss.item(), rec_loss.item(), kld_loss.item())
                self.vae_loss.append(total_loss.item())
                # --------------------------------------------------------------------------------

    def policy_learning_function(self):
        if len(self.memory.memory_buffer_vector) <= self.batch_size:
            return
        else:
            self.actor.train()
            self.critic_q1.train()
            self.critic_q2.train()

            for it in range(1, self.G+1):
                self.update_counter += 1  # this is used for delay/

                z_states, actions, rewards, z_next_states, dones = self.memory.sample_vector_experiences(self.batch_size)

                z_states = np.array(z_states)
                z_states = torch.FloatTensor(z_states)
                z_states = z_states.to(self.device)

                actions = np.array(actions)
                actions = torch.FloatTensor(actions)
                actions = actions.to(self.device)  # send batch to GPU

                z_next_states = np.array(z_next_states)
                z_next_states = torch.FloatTensor(z_next_states)
                z_next_states = z_next_states.to(self.device)

                rewards = np.array(rewards).reshape(-1, 1)
                rewards = torch.FloatTensor(rewards)
                rewards = rewards.to(self.device)  # send batch to GPU

                dones = np.array(dones).reshape(-1, 1)
                dones = torch.FloatTensor(dones)
                dones = dones.to(self.device)  # send batch to GPU

                with torch.no_grad():
                    next_actions = self.actor_target(z_next_states)
                    target_noise = 0.2 * torch.randn_like(next_actions)
                    target_noise = target_noise.clamp_(-0.5, 0.5)
                    next_actions = next_actions + target_noise
                    next_actions = next_actions.clamp_(-1, 1)

                    next_q_values_q1 = self.critic_target_q1.forward(z_next_states, next_actions)
                    next_q_values_q2 = self.critic_target_q2.forward(z_next_states, next_actions)
                    q_min = torch.minimum(next_q_values_q1, next_q_values_q2)

                    Q_target = rewards + (self.gamma * (1 - dones) * q_min)

                Q_vals_q1 = self.critic_q1.forward(z_states, actions)
                Q_vals_q2 = self.critic_q2.forward(z_states, actions)

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
                    actor_loss = - self.critic_q1.forward(z_states, self.actor.forward(z_states)).mean()

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

    def prepare_tensor(self, img_states, actions, rewards, img_next_states, dones):

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

        return img_states, actions, rewards, img_next_states, dones


    def get_action_from_policy(self, state_img_pixels):

        state_image_tensor  = torch.FloatTensor(state_img_pixels)
        state_image_tensor  = state_image_tensor.unsqueeze(0)
        state_image_tensor  = state_image_tensor.permute(0, 3, 1, 2).to(self.device)  # torch.Size([1, 3, 128, 128])

        self.vae.eval()
        #self.simple_encoder.eval()
        self.actor.eval()

        with torch.no_grad():
            _, _, _, z_state  = self.vae.forward(state_image_tensor)  # torch.Size([1, 16])
            #_, z_state  = self.simple_encoder.forward(state_image_tensor)  # torch.Size([1, 16])
            #state_space_input = z_state.unsqueeze(0)
            action = self.actor.forward(z_state)
            action = action.cpu().data.numpy()
        return action[0]


    def update_function(self):
        #self.learn_all_online()
        #self.learn_encoder_function()
        #self.policy_learning_function()
        self.train_special_function()
        #self.test_ensamble()


    def test_ensamble(self):
        if len(self.memory.memory_buffer_experiences) <= self.minimal_buffer_size:
            return
        else:

            self.update_counter += 1

            img_states, actions, rewards, img_next_states, dones = self.memory.sample_frame_vector_experiences(self.batch_size)
            img_states, actions, rewards, img_next_states, dones = self.prepare_tensor(img_states, actions, rewards, img_next_states, dones)

            x_rec, mu, log_var, z = self.vae_critic_ensemble.enconde_image(img_states)

            with torch.no_grad():
                x_rec_next, _, _, z_next = self.vae_critic_ensemble.enconde_image(img_next_states)
                next_actions  = self.actor_target(z_next)
                target_noise  = 0.2 * torch.randn_like(next_actions)
                target_noise  = target_noise.clamp_(-0.5, 0.5)
                next_actions  = next_actions + target_noise
                next_actions  = next_actions.clamp_(-1, 1)
                next_q_values = self.critic_q1_target.forward(z_next, next_actions)
                q_target = rewards + (self.gamma * (1 - dones) * next_q_values)

            q_vals_q1 = self.vae_critic_ensemble.critic_function_q1(z, actions)

            # Loss together
            rec_loss = F.binary_cross_entropy(x_rec, img_states, reduction="sum")
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            critic_loss = F.mse_loss(q_vals_q1, q_target)
            total_loss_vae_critic = rec_loss + kld_loss + critic_loss

            self.vae_critic_optimizer.zero_grad()
            total_loss_vae_critic.backward()
            self.vae_critic_optimizer.step()

            if self.update_counter % self.policy_freq_update == 0:
                # ------- calculate the actor loss
                actor_loss = - self.vae_critic_ensemble.critic_function_q1(z.detach(), self.actor.forward(z.detach())).mean()

                #  ------- Actor step Update  -------  #
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ------------------------------------- Update target networks --------------- #
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_q1_target.parameters(), self.vae_critic_ensemble.critic_model_1.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def train_special_function(self):
        if len(self.memory.memory_buffer_experiences) <= self.minimal_buffer_size:
            return
        else:
            for _ in range(1, self.G + 1):

                self.update_counter += 1
                img_states, actions, rewards, img_next_states, dones = self.memory.sample_frame_vector_experiences(self.batch_size)
                img_states, actions, rewards, img_next_states, dones = self.prepare_tensor(img_states, actions, rewards, img_next_states, dones)

                #  =========================== train special critic-vae ======================================
                self.vae.train()
                self.critic.train()
                self.actor.train()

                x_rec, mu, log_var, z = self.vae.forward(img_states)
                #x_rec, z = self.simple_encoder.forward(img_states)

                # --VAE Loss Function Reconstruction + KL -- #
                rec_loss = F.binary_cross_entropy(x_rec, img_states, reduction="sum")
                #rec_loss = F.mse_loss(x_rec, img_states, reduction="sum")  # todo try this one too

                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss_vae = rec_loss + kld_loss
                #loss_vae = rec_loss

                with torch.no_grad():
                    x_rec_next, mu_next, log_var_next, z_next = self.vae.forward(img_next_states)
                    #x_rec_next, z_next = self.simple_encoder.forward(img_next_states)
                    next_actions       = self.actor_target.forward(z_next)

                    #next_actions = next_actions.cpu().numpy()  # tensor to numpy
                    #next_actions = next_actions + (np.random.normal(0, scale=0.2, size=self.num_action))
                    #next_actions = np.clip(next_actions, -2, 2)
                    #next_actions = torch.FloatTensor(next_actions).to(self.device)

                    target_noise = 0.1 * torch.randn_like(next_actions)
                    target_noise = target_noise.clamp_(-0.5, 0.5)
                    next_actions = next_actions + target_noise
                    next_actions = next_actions.clamp_(-2, 2)

                    next_q_values = self.critic_target(z_next, next_actions)
                    q_target      = rewards + (self.gamma * (1 - dones) * next_q_values)

                q_val = self.critic(z, actions)

                critic_loss = F.mse_loss(q_val, q_target)

                total_loss_vae_critic = (0.1 * loss_vae) + (1.0 * critic_loss)


                self.vae_critic_optimizer.zero_grad()
                total_loss_vae_critic.backward()
                self.vae_critic_optimizer.step()

                #self.encoder_critic_optimizer.zero_grad()
                #total_loss_vae_critic.backward()
                #self.encoder_critic_optimizer.step()


                if self.update_counter % self.policy_freq_update == 0:
                    z_in_actor = z.detach()
                    actor_loss = - self.critic.forward(z_in_actor, self.actor.forward(z_in_actor)).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # ------------------------------------- Update target networks --------------- #
                    for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                    for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def learn_all_online(self):
        if len(self.memory.memory_buffer_experiences) <= self.minimal_buffer_size:
            return
        else:
            img_states, actions, rewards, img_next_states, dones = self.memory.sample_frame_vector_experiences(self.batch_size)
            img_states, actions, rewards, img_next_states, dones = self.prepare_tensor(img_states, actions, rewards, img_next_states, dones)

            #  =========================== train vae model ======================================
            self.vae.train()
            x_state_rec, mu_state, log_var_state, z_input = self.vae.forward(img_states)
            x_next_rect, mu_next, log_var_next, z_target  = self.vae.forward(img_next_states)

            # --Loss Function Reconstruction + KL -- #
            rec_loss = F.binary_cross_entropy(x_state_rec, img_states, reduction="sum")
            kld_loss = -0.5 * torch.sum(1 + log_var_state - mu_state.pow(2) - log_var_state.exp())
            total_loss_vae = rec_loss + kld_loss

            self.vae_optimizer.zero_grad()
            total_loss_vae.backward()
            self.vae_optimizer.step()

            #print("VAE training Loss:", total_loss_vae.item(), rec_loss.item(), kld_loss.item())
            self.vae_loss.append(total_loss_vae.item())

            #  =========================== train actor critic model =================================== #
            z_in   = z_input.detach()
            z_next = z_target.detach()  # all good, here no grands

            self.actor.train()
            self.critic_q1.train()
            self.critic_q2.train()

            self.update_counter += 1

            with torch.no_grad():
                next_actions = self.actor_target(z_next)
                target_noise = 0.2 * torch.randn_like(next_actions)
                target_noise = target_noise.clamp_(-0.5, 0.5)
                next_actions = next_actions + target_noise
                next_actions = next_actions.clamp_(-1, 1)

                next_q_values_q1 = self.critic_target_q1.forward(z_next, next_actions)
                next_q_values_q2 = self.critic_target_q2.forward(z_next, next_actions)
                q_min = torch.minimum(next_q_values_q1, next_q_values_q2)
                Q_target = rewards + (self.gamma * (1 - dones) * q_min)

            Q_vals_q1 = self.critic_q1.forward(z_in, actions)
            Q_vals_q2 = self.critic_q2.forward(z_in, actions)

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
                actor_loss = - self.critic_q1.forward(z_in, self.actor.forward(z_in)).mean()

                # Actor Update
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

    def plot_functions(self, rewards):
        #plt.subplot(2, 2, 1)
        #plt.title("VAE LOSS")
        #plt.plot(self.vae_critic_loss)

        plt.subplot(2, 2, 2)
        plt.title("Rewards")
        plt.plot(rewards)

        plt.savefig(f"plot_results/pendulum_curves.png")


    def calculate_z_and_similarity(self, img_input):
        self.vae.eval()
        with torch.no_grad():
            img_tensor = torch.FloatTensor(img_input)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.permute(0, 3, 1, 2).to(device)

            x_rec, _, _, z = self.vae.forward(img_tensor)
            x_rec = x_rec.permute(0, 2, 3, 1)
            x_rec = x_rec[0].cpu().numpy()
            z = z.cpu().numpy()

        ssim_const = ssim(img_input, x_rec, data_range=img_input.max() - img_input.min(), channel_axis=2)

        return ssim_const, z[0]


    def save_vae_model(self):
        torch.save(self.vae.state_dict(), "trained_models/vae_model.pth")
        print("VAE model has been saved")

    def load_vae_model(self):
        self.vae.load_state_dict(torch.load("trained_models/vae_model.pth"))
        print("VAE model has been loaded")



def pre_pro_image(image_array):
    img        = cv2.resize(image_array, (128, 128), interpolation=cv2.INTER_AREA)
    norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #cv2.imshow("Normalized image", norm_image)
    #cv2.waitKey(10)
    return norm_image


def run_random_exploration(env, agent,  num_exploration_episodes, episode_horizont):
    for episode in tqdm(range(1, num_exploration_episodes + 1)):
        #print(episode, "Exploration")
        env.reset()
        state_image = env.render(mode='rgb_array')  # return the rendered image and can be used as input-state image
        state_image = pre_pro_image(state_image)
        for step in range(1, episode_horizont + 1):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            new_state_image = env.render(mode='rgb_array')
            new_state_image = pre_pro_image(new_state_image)
            agent.memory.save_frame_experience_buffer(state_image)
            state_image = new_state_image
            if done:
                break


def run_training_rl_method(env, agent, num_episodes_training=100, episode_horizont=200):
    rewards = []

    for episode in range(1, num_episodes_training + 1):

        print(f"-----------------Episode {episode}-----------------------------")
        env.reset()
        episode_reward = 0
        state_image = env.render(mode='rgb_array')  # return the rendered image so can be used as input-state image
        state_image = pre_pro_image(state_image)

        for step in range(1, episode_horizont + 1):

            action = agent.get_action_from_policy(state_image)
            noise  = np.random.normal(0, scale=0.1, size=1)
            action = action + noise
            action = np.clip(action, -2, 2)

            obs_vector, reward, done, info = env.step(action)

            new_state_image = env.render(mode='rgb_array')
            new_state_image = pre_pro_image(new_state_image)

            agent.memory.save_frame_vector_experience_buffer(state_image, action, reward, new_state_image, done)
            state_image = new_state_image
            episode_reward += reward

            # -----------#
            #similarity_state, z_state           = agent.calculate_z_and_similarity(state_image)
            #similarity_next_state, z_next_state = agent.calculate_z_and_similarity(new_state_image)

            #if similarity_state >= 0.90:   # similarity_tolerance:
                #agent.memory.save_vector_experience_buffer(z_state, action, reward, z_next_state, done)

            if done:
                break

            # ----- Update Function------#
            agent.update_function()

        rewards.append(episode_reward)
        print(f"Episode {episode} End, Total reward: {episode_reward}")
        if episode % 50 == 0:
            agent.plot_functions(rewards)

    # todo save RL models
    agent.plot_functions(rewards)
    agent.save_vae_model()


def evaluate_vae_model(env, agent):
    env.reset()
    state_image = env.render(mode='rgb_array')
    state_image = pre_pro_image(state_image)

    #agent.load_vae_model()
    agent.vae.eval()
    #agent.simple_encoder.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_image)
        state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.permute(0, 3, 1, 2).to(device)

        x_rec, _, _, _ = agent.vae.forward(state_tensor)
        #x_rec, _ = agent.simple_encoder.forward(state_tensor)

        x_rec = x_rec.permute(0, 2, 3, 1)
        #x_rec = x_rec.view(-1, 128, 128, 3)
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

    #num_exploration_episodes  = 100
    #run_random_exploration(env, agent, num_exploration_episodes, episode_horizont)
    # run_training_vae_only(agent)

    num_episodes_training     = 2000
    episode_horizont          = 200
    run_training_rl_method(env, agent, num_episodes_training, episode_horizont)
    evaluate_vae_model(env, agent)
    env.close()


if __name__ == '__main__':
    main_run()