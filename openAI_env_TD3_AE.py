
"""
TD3 with deterministic AE for pendulum env only
This is my version and  re-implementation of the paper  https://arxiv.org/pdf/1910.01741.pdf
however I removed or changed many part here. The original paper use SAC

Every state is rendered and passed as input to the autoencoder, after preprocessing it

original image 510 x510 x 3
Input for the encoder = 3 stacked frames, gray scalded, normalized and resized (84 , 83)
The input is batch-size x 3 x 84 x 84, where the stacked number takes the place of the channel for covnet

status = working
keys= initial exploration is very important, the activation function in the decoder helps
Status = Working
"""

import gym
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Working with GPU")
else:
    device = torch.device('cpu')
    print("Working with CPU")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Memory:
    def __init__(self, replay_max_size=40_000):
        self.replay_max_size = replay_max_size
        self.memory_buffer   = deque(maxlen=replay_max_size)

    def save_experience_to_buffer(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory_buffer.append(experience)

    def sample_experiences_from_buffer(self, sample_size):
        state_batch      = []
        action_batch     = []
        reward_batch     = []
        next_state_batch = []
        done_batch       = []

        batch = random.sample(self.memory_buffer, sample_size)
        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        state_batch      = np.array(state_batch)
        action_batch     = np.array(action_batch)
        reward_batch     = np.array(reward_batch).reshape(-1, 1)
        done_batch       = np.array(done_batch).reshape(-1, 1)
        next_state_batch = np.array(next_state_batch)

        state_batch_tensor      = torch.FloatTensor(state_batch).to(device)
        action_batch_tensor     = torch.FloatTensor(action_batch).to(device)
        reward_batch_tensor     = torch.FloatTensor(reward_batch).to(device)
        done_batch_tensor       = torch.FloatTensor(done_batch).to(device)
        next_batch_state_tensor = torch.FloatTensor(next_state_batch).to(device)

        # just put in the right order [b, H*, W, C] --->  [b, c, H*, W]
        #state_batch_tensor      = state_batch_tensor.permute(0, 3, 1, 2)
        #next_batch_state_tensor = next_batch_state_tensor.permute(0, 3, 1, 2)
        return state_batch_tensor, action_batch_tensor, reward_batch_tensor, next_batch_state_tensor, done_batch_tensor

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class FrameStack:
    def __init__(self, k=3, env=gym.make('Pendulum-v1')):
        self.env = env
        self.k   = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

    def reset(self):
        self.env.reset()
        obs = self.env.render(mode='rgb_array')
        obs = self.preprocessing_image(obs)
        for _ in range(self.k):
            self.frames_stacked.append(obs)
        #stacked_vector = np.concatenate(list(self.frames_stacked), axis=0)
        stacked_vector = np.array(list(self.frames_stacked))
        return stacked_vector

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self.env.render(mode='rgb_array')
        obs = self.preprocessing_image(obs)
        self.frames_stacked.append(obs)
        stacked_vector = np.array(list(self.frames_stacked))
        #stacked_vector = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_vector, reward, done, info

    def preprocessing_image(self, image_array):
        resized     = cv2.resize(image_array, (84, 84), interpolation=cv2.INTER_AREA)
        gray_image  = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        norm_image  = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        state_image = norm_image
        return state_image
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.num_filters = 32
        self.num_layers  = 4
        self.latent_dim  = latent_dim

        self.fc_1 = nn.Linear(self.latent_dim, 39200)

        '''
        self.deconvs = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.deconvs.append(nn.ConvTranspose2d(self.num_filters, self.num_filters, 3, stride=1))
        self.deconvs.append(nn.ConvTranspose2d(self.num_filters, 3, 3, stride=2, output_padding=1))
        '''

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=3, kernel_size=3, stride=2, output_padding=1),
            nn.Sigmoid(),  # original paper no use activation function here. I added it and helps
        )

    def forward(self, x):
        '''
        h = torch.relu(self.fc_1(x))
        x = h.view(-1, 32, 35, 35)
        for i in range(0, self.num_layers - 1):
            x = torch.relu(self.deconvs[i](x))
        obs = self.deconvs[-1](x)  # there is no activation function on the original paper
        obs = torch.sigmoid(obs)
        return obs
        '''
        x = torch.relu(self.fc_1(x))
        x = x.view(-1, 32, 35, 35)
        x = self.deconvs(x)
        return x
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.num_layers  = 4
        self.num_filters = 32
        self.latent_dim  = latent_dim

        self.cov_net = nn.ModuleList([nn.Conv2d(3, self.num_filters, 3, stride=2)])
        for i in range(self.num_layers - 1):
            self.cov_net.append(nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1))

        self.fc  = nn.Linear(39200, self.latent_dim)
        self.ln  = nn.LayerNorm(self.latent_dim)

    def forward_conv(self, x):
        conv = torch.relu(self.cov_net[0](x))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.cov_net[i](conv))
        h = torch.flatten(conv, start_dim=1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        h_fc   = self.fc(h)
        h_norm = self.ln(h_fc)
        out    = torch.tanh(h_norm)
        return out

    def copy_conv_weights_from(self, model_source):
        for i in range(self.num_layers):
            tie_weights(src=model_source.cov_net[i], trg=self.cov_net[i])
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(Actor, self).__init__()
        self.encoder_net = Encoder(latent_dim)
        self.act_net = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim),
        )
        self.apply(weight_init)

    def forward(self, state, detach_encoder=False):
        z_vector = self.encoder_net(state, detach=detach_encoder)
        output   = self.act_net(z_vector)
        output   = torch.tanh(output)
        return output * 2.0  # 2.0 is for pendulum env range

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class Critic(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(Critic, self).__init__()

        self.encoder_net = Encoder(latent_dim)
        self.hidden_dim  = 1024
        '''
        self.Q1 = nn.Sequential(
            nn.Linear(input_critic_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        self.Q2 = nn.Sequential(
            nn.Linear(input_critic_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        '''
        self.Q1 = QFunction(latent_dim, action_dim, self.hidden_dim)
        self.Q2 = QFunction(latent_dim, action_dim, self.hidden_dim)

        self.apply(weight_init)

    def forward(self, state, action, detach_encoder=False):
        z_vector = self.encoder_net(state, detach=detach_encoder)
        q1       = self.Q1(z_vector, action)
        q2       = self.Q2(z_vector, action)
        return q1, q2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class RLAgent:
    def __init__(self):
        self.G = 1

        self.encoder_lr = 1e-3
        self.decoder_lr = 1e-3

        self.critic_lr  = 1e-3  # 1e-3
        self.actor_lr   = 1e-4  # 1e-4

        self.tau         = 0.005
        self.tau_encoder = 0.001
        self.gamma       = 0.99

        self.update_counter     = 0
        self.policy_freq_update = 2

        self.batch_size = 32   # 32
        self.latent_dim = 50   # 50
        self.action_dim = 1

        self.memory = Memory()

        # main networks
        self.actor  = Actor(self.latent_dim, self.action_dim).to(device)
        self.critic = Critic(self.latent_dim, self.action_dim).to(device)
        # target networks
        self.actor_target  = Actor(self.latent_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.latent_dim, self.action_dim).to(device)

        # tie encoders between actor and critic
        # with this, any changes in the critic encoder
        # will also be affecting the actor-encoder during the whole training
        self.actor.encoder_net.copy_conv_weights_from(self.critic.encoder_net)

        # copy weights and bias from main to target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        # main Decoder
        self.decoder = Decoder(self.latent_dim).to(device)
        self.decoder.apply(weight_init)

        # Optimizer
        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder_net.parameters(), lr=self.encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(),            lr=self.decoder_lr, weight_decay=1e-7)
        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(),              lr=self.actor_lr, betas=(0.9, 0.999))
        self.critic_optimizer  = torch.optim.Adam(self.critic.parameters(),             lr=self.critic_lr, betas=(0.9, 0.999))

        self.actor.train(True)
        self.critic.train(True)
        self.critic_target.train(True)
        self.actor_target.train(True)
        self.decoder.train(True)

    def select_action_from_policy(self, state_image_pixel):
        with torch.no_grad():
            state_image_tensor  = torch.FloatTensor(state_image_pixel)
            state_image_tensor  = state_image_tensor.unsqueeze(0).to(device)
            action = self.actor.forward(state_image_tensor)
            action = action.cpu().data.numpy()
        return action[0]

    def update_function(self):
        if len(self.memory.memory_buffer) <= self.batch_size:
            return
        else:
            for _ in range(1, self.G+1):
                self.update_counter += 1
                state_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.memory.sample_experiences_from_buffer(self.batch_size)

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # update the critic part
                with torch.no_grad():
                    next_actions = self.actor_target(next_states_batch)
                    target_noise = 0.2 * torch.randn_like(next_actions)
                    target_noise = target_noise.clamp_(-0.5, 0.5)
                    next_actions = next_actions + target_noise
                    next_actions = next_actions.clamp_(-2, 2)

                    next_q_values_q1, next_q_values_q2 = self.critic_target(next_states_batch, next_actions)
                    q_min = torch.minimum(next_q_values_q1, next_q_values_q2)
                    q_target = rewards_batch + (self.gamma * (1 - dones_batch) * q_min)

                q1, q2 = self.critic(state_batch, actions_batch)

                critic_loss_1 = F.mse_loss(q1, q_target)
                critic_loss_2 = F.mse_loss(q2, q_target)
                critic_loss_total = critic_loss_1 + critic_loss_2

                self.critic_optimizer.zero_grad()
                critic_loss_total.backward()
                self.critic_optimizer.step()

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # Update the actor and soft updates of targets networks
                if self.update_counter % self.policy_freq_update == 0:
                    action_actor       = self.actor(state_batch, detach_encoder=True)
                    actor_q1, actor_q2 = self.critic(state_batch, action_actor, detach_encoder=True)

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

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # Update the autoencoder part
                z_vector = self.critic.encoder_net(state_batch)
                rec_obs  = self.decoder(z_vector)

                rec_loss    = F.mse_loss(state_batch, rec_obs)
                latent_loss = (0.5 * z_vector.pow(2).sum(1)).mean()  # add L2 penalty on latent representation

                ae_loss = rec_loss + 1e-6 * latent_loss

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                ae_loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

    def save_models(self):
        torch.save(self.actor.state_dict(), f'trained_models/AE-TD3_actor_pendulum.pht')
        torch.save(self.critic.encoder_net.state_dict(), f'trained_models/AE-TD3_encoder_pendulum.pht')
        torch.save(self.decoder.state_dict(), f'trained_models/AE-TD3_decoder_pendulum.pht')
        print("models have been saved...")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_reward(reward_vector):
    plt.title("Rewards")
    plt.plot(reward_vector)
    plt.savefig(f"plot_results/AE-TD3_pendulum_reward_curve.png")
    np.savetxt(f"plot_results/AE-TD3_pendulum_reward_curve.txt", reward_vector)
    #plt.show()

def plot_reconstructions(input_img, reconstruction_img):
    input_img          = np.transpose(input_img, (1, 2, 0))
    reconstruction_img = np.transpose(reconstruction_img, (1, 2, 0))

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(input_img)

    plt.subplot(1, 2, 2)
    plt.title("Reconstruction")
    plt.imshow(reconstruction_img)

    plt.savefig(f"plot_results/AE-TD3_pendulum_image_reconstruction.png")
    #plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def run_random_exploration(env, agent, frames_stack, num_exploration_episodes=200, episode_horizont=200):
    print("exploration start")
    for episode in tqdm(range(1, num_exploration_episodes + 1)):
        state_image = frames_stack.reset()
        for step in range(1, episode_horizont + 1):
            action = env.action_space.sample()
            new_state_image, reward, done, _ = frames_stack.step(action)
            agent.memory.save_experience_to_buffer(state_image, action, reward, new_state_image, done)
            state_image = new_state_image
            if done:
                break
    print("exploration end")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def run_training_rl_method(env, agent, frames_stack, num_episodes_training=500, episode_horizont=200):
    total_reward = []
    for episode in range(1, num_episodes_training + 1):
        state_image = frames_stack.reset()
        episode_reward = 0
        for step in range(1, episode_horizont + 1):
            action = agent.select_action_from_policy(state_image)
            noise  = np.random.normal(0, scale=0.1, size=1)
            action = action + noise
            action = np.clip(action, -2, 2)
            new_state_image, reward, done, _ = frames_stack.step(action)
            agent.memory.save_experience_to_buffer(state_image, action, reward, new_state_image, done)
            state_image = new_state_image
            episode_reward += reward
            if done:
                break
            agent.update_function()
        total_reward.append(episode_reward)
        print(f"Episode {episode} End, Total reward: {episode_reward}")
    agent.save_models()
    plot_reward(total_reward)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def autoencoder_evaluation(agent, frames_stack):
    state_image        = frames_stack.reset()
    state_image_tensor = torch.FloatTensor(state_image)
    state_image_tensor = state_image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        z_vector = agent.critic.encoder_net(state_image_tensor)
        rec_obs  = agent.decoder(z_vector)
        rec_obs = rec_obs.cpu().numpy()

    plot_reconstructions(state_image, rec_obs[0])
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def main():
    env   = gym.make('Pendulum-v1')
    agent = RLAgent()
    frames_stack = FrameStack(env=env)

    run_random_exploration(env, agent, frames_stack)
    run_training_rl_method(env, agent, frames_stack)
    autoencoder_evaluation(agent, frames_stack)
    env.close()


if __name__ == '__main__':
    main()