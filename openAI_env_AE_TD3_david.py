import gym
import cv2
import copy
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

        # just put in the right order [b, 3, H, W]
        state_batch_tensor      = state_batch_tensor.permute(0, 3, 1, 2)
        next_batch_state_tensor = next_batch_state_tensor.permute(0, 3, 1, 2)

        return state_batch_tensor, action_batch_tensor, reward_batch_tensor, next_batch_state_tensor, done_batch_tensor
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.cov_net = nn.ModuleList([nn.Conv2d(3,  32, 3, stride=2),
                                      nn.Conv2d(32, 32, 3, stride=1),
                                      nn.Conv2d(32, 32, 3, stride=1),
                                      nn.Conv2d(32, 32, 3, stride=1),
                                      ])
        '''
        self.cov_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        '''
        self.fc  = nn.Linear(39200, latent_dim)
        self.ln  = nn.LayerNorm(latent_dim)

    def forward_conv(self, x):
        x = torch.relu(self.cov_net[0](x))
        x = torch.relu(self.cov_net[1](x))
        x = torch.relu(self.cov_net[2](x))
        x = torch.relu(self.cov_net[3](x))  # torch.Size([1, 32, 35, 35])
        x = torch.flatten(x, start_dim=1)  # torch.Size([batch_size, 39200])
        return x

    def forward(self, x, detach_encoder=False):
        '''
        x = self.cov_net(x)  # torch.Size([1, 32, 35, 35])
        if detach_encoder:
            x = torch.flatten(x, start_dim=1)  # torch.Size([batch_size, 39200])
            x = x.detach()
        else:
            x = torch.flatten(x, start_dim=1)  # torch.Size([batch_size, 39200])
        x = self.fc(x)
        x = self.ln(x)
        x = torch.tanh(x)    # output between -1 ~ 1
        return x
        '''

        x = self.forward_conv(x)
        if detach_encoder:
            x = x.detach()

        x = self.fc(x)
        x = self.ln(x)
        x = torch.tanh(x)
        return x

    def copy_conv_weights_from(self, model_source):
        for i in range(4):
            self.cov_net[i].weight = model_source.cov_net[i].weight
            self.cov_net[i].bias   = model_source.cov_net[i].bias


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.fc_1 = nn.Linear(self.latent_dim, 39200)
        self.decov_net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=0, output_padding=1),
            nn.Sigmoid(),  # I put sigmoid because can help to get the reconstruction  between 0~1
        )

    def forward(self, x):
        x = self.fc_1(x)
        x = torch.relu(x)  # no sure about this activation function
        x = x.view(-1, 32, 35, 35)
        x = self.decov_net(x)
        return x
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(Actor, self).__init__()

        self.encoder_net = Encoder(latent_dim)
        self.act_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )
        self.apply(weight_init)

    def forward(self, state, detach_encoder=False):
        z_vector = self.encoder_net(state, detach_encoder)
        output   = self.act_net(z_vector)
        output   = torch.tanh(output) * 2.0  # 2.0 is for pendulum env range
        return output
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Critic(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(Critic, self).__init__()

        self.encoder_net  = Encoder(latent_dim)
        input_critic_size = latent_dim + action_dim

        self.Q1 = nn.Sequential(
            nn.Linear(input_critic_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.Q2 = nn.Sequential(
            nn.Linear(input_critic_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.apply(weight_init)

    def forward(self, state, action, detach_encoder=False):
        z_vector = self.encoder_net.forward(state, detach_encoder)
        z_vector = torch.cat([z_vector, action], dim=1)
        q1 = self.Q1(z_vector)
        q2 = self.Q2(z_vector)
        return q1, q2
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class RLAgent:

    def __init__(self):
        self.G  = 1

        self.encoder_lr = 1e-3
        self.decoder_lr = 1e-3

        self.critic_lr  = 1e-3  # 1e-3
        self.actor_lr   = 1e-4  # 1e-4

        self.tau         = 0.005
        self.tau_encoder = 0.001
        self.gamma       = 0.99

        self.update_counter     = 0
        self.policy_freq_update = 2

        self.batch_size = 32  # 32
        self.latent_dim = 50  # 50 todo play with this number
        self.action_dim = 1

        # load and create models
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

        # Check params
        #for p1, p2 in zip(self.critic.encoder_net.cov_net.parameters(), self.actor_target.encoder_net.cov_net.parameters()):
            #print(torch.equal(p1, p2))

        '''
        self.encoder = Encoder(self.latent_dim).to(device)
        self.actor_target  = Actor(self.encoder, self.latent_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.encoder, self.latent_dim, self.action_dim).to(device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Check params
        #for p1, p2 in zip(self.actor.encoder_net.parameters(), self.critic_target.encoder_net.parameters()):
            #print(torch.equal(p1, p2))
        '''

        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder_net.parameters(), lr=self.encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.decoder_lr, weight_decay=1e-7)
        self.critic_optimizer  = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(),   lr=self.actor_lr)


    def update_function(self):
        if len(self.memory.memory_buffer) <= self.batch_size:
            return
        else:
            for _ in range(1, self.G+1):

                self.update_counter += 1

                state_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.memory.sample_experiences_from_buffer(self.batch_size)

                # Update the autoencoder part
                z_vector = self.critic.encoder_net(state_batch, detach_encoder=False)
                rec_obs  = self.decoder(z_vector)

                rec_loss    = F.mse_loss(state_batch, rec_obs)
                latent_loss = (0.5 * z_vector.pow(2).sum(1)).mean()  # add L2 penalty on latent representation

                ae_loss = rec_loss + 1e-6 * latent_loss

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                ae_loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                #print("Rec Loss:", rec_loss.item(), "Latent Loss:", latent_loss.item())
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

                # update the critic part
                with torch.no_grad():
                    next_actions = self.actor_target.forward(next_states_batch)
                    target_noise = 0.2 * torch.randn_like(next_actions)
                    target_noise = target_noise.clamp_(-0.5, 0.5)
                    next_actions = next_actions + target_noise
                    next_actions = next_actions.clamp_(-2, 2)

                    next_q_values_q1, next_q_values_q2 = self.critic_target.forward(next_states_batch, next_actions)
                    q_min    = torch.minimum(next_q_values_q1, next_q_values_q2)

                    q_target = rewards_batch + (self.gamma * (1 - dones_batch) * q_min)

                q1, q2 = self.critic.forward(state_batch, actions_batch, detach_encoder=False)

                critic_loss_1 = F.mse_loss(q1, q_target)
                critic_loss_2 = F.mse_loss(q2, q_target)
                critic_loss_total = critic_loss_1 + critic_loss_2

                self.critic_optimizer.zero_grad()
                critic_loss_total.backward()
                self.critic_optimizer.step()
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # Update the actor and soft updates of targets networks
                if self.update_counter % self.policy_freq_update == 0:
                    action_actor       = self.actor.forward(state_batch, detach_encoder=True)
                    actor_q1, actor_q2 = self.critic.forward(state_batch, action_actor, detach_encoder=True)  # try with False too, maybe

                    actor_q_min = torch.minimum(actor_q1, actor_q2)
                    actor_loss  = - actor_q_min.mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    # ------------------------------------- Update target networks --------------- #

                    for target_param, param in zip(self.critic_target.Q1.parameters(), self.critic.Q1.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                    for target_param, param in zip(self.critic_target.Q2.parameters(), self.critic.Q2.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                    for target_param, param in zip(self.critic.encoder_net.parameters(), self.critic_target.encoder_net.parameters()):
                        target_param.data.copy_(param.data * self.tau_encoder + target_param.data * (1.0 - self.tau_encoder))

                    for target_param, param in zip(self.actor.encoder_net.parameters(), self.actor_target.encoder_net.parameters()):
                        target_param.data.copy_(param.data * self.tau_encoder + target_param.data * (1.0 - self.tau_encoder))

                    '''
                    for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                    for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
                    '''
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def select_action_from_policy(self, state_image_pixel):
        state_image_tensor  = torch.FloatTensor(state_image_pixel)
        state_image_tensor  = state_image_tensor.unsqueeze(0)  # torch.Size([1, 84, 84, 3])
        state_image_tensor  = state_image_tensor.permute(0, 3, 1, 2).to(device)  # torch.Size([1, 3, 84, 84])
        with torch.no_grad():
            action = self.actor.forward(state_image_tensor, detach_encoder=True)
            action = action.cpu().data.numpy()
        return action[0]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def pre_pro_image(image_array):
    resized     = cv2.resize(image_array, (84, 84), interpolation=cv2.INTER_AREA)
    norm_image  = resized / 255.0
    state_image = norm_image
    return state_image

def plot_reward(reward_vector):
    plt.title("Rewards")
    plt.plot(reward_vector)
    plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def run_random_exploration(env, agent,  num_exploration_episodes=50, episode_horizont=200):
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
            agent.memory.save_experience_to_buffer(state_image, action, reward, new_state_image, done)
            state_image = new_state_image
            if done:
                break
    print("exploration end")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def run_training_rl_method(env, agent, num_episodes_training=200, episode_horizont=200):
    total_reward = []
    for episode in range(1, num_episodes_training + 1):
        env.reset()
        state_image = env.render(mode='rgb_array')
        state_image = pre_pro_image(state_image)
        episode_reward = 0
        for step in range(1, episode_horizont + 1):
            action = agent.select_action_from_policy(state_image)
            noise  = np.random.normal(0, scale=0.1, size=1)
            action = action + noise
            action = np.clip(action, -2, 2)
            obs_next_state_vector, reward, done, info = env.step(action)
            new_state_image = env.render(mode='rgb_array')
            new_state_image = pre_pro_image(new_state_image)
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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def evaluation(env, agent):
    # evaluation encoder and decoder
    env.reset()
    state_image_pixel = env.render(mode='rgb_array')
    state_image_pixel = pre_pro_image(state_image_pixel)
    state_image_tensor = torch.FloatTensor(state_image_pixel)
    state_image_tensor = state_image_tensor.unsqueeze(0)  # torch.Size([1, 84, 84, 3])
    state_image_tensor = state_image_tensor.permute(0, 3, 1, 2).to(device)  # torch.Size([1, 3, 84, 84])
    with torch.no_grad():
        z_vector = agent.critic.encoder_net(state_image_tensor, detach_encoder=False)
        rec_obs  = agent.decoder(z_vector)
    rec_obs = rec_obs.permute(0, 2, 3, 1)
    rec_obs = rec_obs.cpu().numpy()
    plt.imshow(rec_obs[0])
    plt.show()


def main():
    env   = gym.make('Pendulum-v1')
    agent = RLAgent()
    run_random_exploration(env, agent)
    run_training_rl_method(env, agent)
    evaluation(env, agent)
    env.close()


if __name__ == '__main__':
    main()