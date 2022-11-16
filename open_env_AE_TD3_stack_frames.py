'''
This use a stack of 3 frames as input of autoencoder
'''

import gym
import cv2
import torch
import torch.nn as nn

import random
import numpy as np
from tqdm import tqdm
from collections import deque

import torch.nn.functional as F
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
        self.k   = k  # number of frames stacked
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

        self.fc_1    = nn.Linear(self.latent_dim, 39200)

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
            nn.Sigmoid(),  # # original paper no use activation function here. I added it
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

        self.num_layers = 4
        num_filters     = 32

        self.cov_net = nn.ModuleList([nn.Conv2d(3, num_filters, 3, stride=2)])

        for i in range(self.num_layers - 1):
            self.cov_net.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        '''
        self.cov_net = nn.ModuleList([nn.Conv2d(3,  32, 3, stride=2),
                                      nn.Conv2d(32, 32, 3, stride=1),
                                      nn.Conv2d(32, 32, 3, stride=1),
                                      nn.Conv2d(32, 32, 3, stride=1),
                                      ])
        '''
        self.fc  = nn.Linear(39200, latent_dim)
        self.ln  = nn.LayerNorm(latent_dim)

    def forward_conv(self, x):
        '''
        x = torch.relu(self.cov_net[0](x))
        x = torch.relu(self.cov_net[1](x))
        x = torch.relu(self.cov_net[2](x))
        x = torch.relu(self.cov_net[3](x))
        x = torch.flatten(x, start_dim=1)
        return x
        '''
        conv = torch.relu(self.cov_net[0](x))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.cov_net[i](conv))
        h = torch.flatten(conv, start_dim=1)
        return h

    def forward(self, obs, detach=False):
        '''
        x = self.forward_conv(x)
        if detach_encoder:
            x = x.detach()
        x = self.fc(x)
        x = self.ln(x)
        x = torch.tanh(x)
        '''
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        h_fc   = self.fc(h)
        h_norm = self.ln(h_fc)
        out    = torch.tanh(h_norm)
        return out

    def copy_conv_weights_from(self, model_source):
        for i in range(self.num_layers):
            #self.cov_net[i].weight = model_source.cov_net[i].weight
            #self.cov_net[i].bias   = model_source.cov_net[i].bias
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

        self.encoder_net  = Encoder(latent_dim)
        input_critic_size = latent_dim + action_dim

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
        self.Q1 = QFunction(latent_dim, 1, 1024)
        self.Q2 = QFunction(latent_dim, 1, 1024)

        self.apply(weight_init)

    def forward(self, state, action, detach_encoder=False):
        z_vector = self.encoder_net(state, detach=detach_encoder)
        q1 = self.Q1(z_vector, action)
        q2 = self.Q2(z_vector, action)
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

        self.batch_size = 32  # 128
        self.latent_dim = 50   # 50
        self.action_dim = 1

        self.memory = Memory()

        self.actor  = Actor(self.latent_dim, self.action_dim).to(device)
        self.critic = Critic(self.latent_dim, self.action_dim).to(device)

        self.actor_target  = Actor(self.latent_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.latent_dim, self.action_dim).to(device)

        # tie encoders between actor and critic
        self.actor.encoder_net.copy_conv_weights_from(self.critic.encoder_net)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.decoder = Decoder(self.latent_dim).to(device)
        self.decoder.apply(weight_init)

        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder_net.parameters(), lr=self.encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.decoder_lr, weight_decay=1e-7)

        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, betas=(0.9, 0.999))
        self.critic_optimizer  = torch.optim.Adam(self.critic.parameters(),  lr=self.critic_lr, betas=(0.9, 0.999))

        self.actor.train(True)
        self.critic.train(True)
        self.critic_target.train(True)
        self.actor_target.train(True)
        self.decoder.train(True)


    def select_action_from_policy(self, state_image_pixel):
        with torch.no_grad():
            state_image_tensor  = torch.FloatTensor(state_image_pixel)
            state_image_tensor  = state_image_tensor.unsqueeze(0).to(device)
            #state_image_tensor  = state_image_tensor.permute(0, 3, 1, 2).to(device)
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

                rec_loss = F.mse_loss(state_batch, rec_obs)

                latent_loss = (0.5 * z_vector.pow(2).sum(1)).mean()  # add L2 penalty on latent representation

                ae_loss = rec_loss + 1e-6 * latent_loss

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                ae_loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_reward(reward_vector):
    plt.title("Rewards")
    plt.plot(reward_vector)
    plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def run_random_exploration(env, agent, frames_stack, num_exploration_episodes=100, episode_horizont=200):
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
def run_training_rl_method(env, agent, frames_stack, num_episodes_training=400, episode_horizont=200):
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
    plot_reward(total_reward)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def evaluation(env, agent, frames_stack):
    state_image = frames_stack.reset()
    state_image_tensor = torch.FloatTensor(state_image)
    state_image_tensor = state_image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        z_vector = agent.critic.encoder_net(state_image_tensor)
        rec_obs = agent.decoder(z_vector)

    rec_obs = rec_obs.cpu().numpy()
    plt.imshow(rec_obs[0][1])
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def main():
    env   = gym.make('Pendulum-v1')
    agent = RLAgent()
    frames_stack = FrameStack(env=env)

    run_random_exploration(env, agent, frames_stack)
    run_training_rl_method(env, agent, frames_stack)
    evaluation(env, agent, frames_stack)
    env.close()


if __name__ == '__main__':
    main()