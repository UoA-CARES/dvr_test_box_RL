
import cv2
import torch
import torch.nn as nn
import gym

from collections import deque
import random
import numpy as np
from tqdm import tqdm

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

    def __init__(self, replay_max_size=20_000):
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

        state_batch_tensor      = torch.FloatTensor(state_batch)
        action_batch_tensor     = torch.FloatTensor(action_batch).to(device)
        reward_batch_tensor     = torch.FloatTensor(reward_batch).to(device)
        done_batch_tensor       = torch.FloatTensor(done_batch).to(device)
        next_batch_state_tensor = torch.FloatTensor(next_state_batch)

        # just put in the right order [b, 3, H, W]
        state_batch_tensor      = state_batch_tensor.permute(0, 3, 1, 2).to(device)
        next_batch_state_tensor = next_batch_state_tensor.permute(0, 3, 1, 2).to(device)

        return state_batch_tensor, action_batch_tensor, reward_batch_tensor, next_batch_state_tensor, done_batch_tensor

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.latent_dim  = latent_dim

        self.cov_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,  stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.fc_1 = nn.Linear(46656, self.latent_dim)
        self.ln   = nn.LayerNorm(self.latent_dim)


    def forward(self, x,  detach=False):
        x = self.cov_net(x)                # torch.Size([batch_size, 64, 27, 27])
        x = torch.flatten(x, start_dim=1)  # torch.Size([batch_size, 46656])
        if detach:
            x = x.detach()
        x = self.fc_1(x)
        x = self.ln(x)
        out = torch.tanh(x)  # not 100% sure but the tanh may help to normalize the output between -1 and 1
        return out

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.latent_dim  = latent_dim

        self.fc_1 = nn.Linear(self.latent_dim, 46656)

        self.decov_net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=0, output_padding=1),
            #nn.ReLU(),  # original paper puts Relu here
            nn.Sigmoid(),  # I put sigmoid because can help to get the reconstruction  between 0~1
        )

    def forward(self, x):
        x = self.fc_1(x)
        x = torch.relu(x)
        x = x.view(-1, 64, 27, 27)
        x = self.decov_net(x)
        return x

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(Actor, self).__init__()

        self.encoder = Encoder(latent_dim)

        self.act_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, state, detach_encoder=False):
        z_vector_in  = self.encoder(state, detach_encoder)
        output = self.act_net(z_vector_in)
        output = torch.tanh(output) * 2.0
        return output


class Critic(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(Critic, self).__init__()

        input_critic_size = latent_dim + action_dim

        self.encoder = Encoder(latent_dim)

        self.Q1 = nn.Sequential(
            nn.Linear(input_critic_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.Q2 = nn.Sequential(
            nn.Linear(input_critic_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state, action, detach_encoder=False):
        z_vector = self.encoder.forward(state, detach_encoder)
        x_in = torch.cat([z_vector, action], dim=1)
        q1   = self.Q1(x_in)
        q2   = self.Q2(x_in)
        return q1, q2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class RLAgent(object):
    """TD3 + AE algorithm"""
    def __init__(self):
        encoder_lr = 1e-3
        decoder_lr = 1e-3
        actor_lr   = 1e-4
        critic_lr  = 1e-3

        self.decoder_latent_lambda = 1e-6

        self.update_counter     = 0
        self.policy_freq_update = 2

        self.gamma = 0.99
        self.tau = 0.005

        self.memory = Memory()

        self.batch_size = 32
        self.latent_dim = 50
        self.action_dim = 1

        self.critic = Critic(self.latent_dim, self.action_dim).to(device)
        self.actor  = Actor(self.latent_dim,  self.action_dim).to(device)

        # tie encoders between actor and critic
        self.actor.encoder.load_state_dict(self.critic.encoder.state_dict())

        self.critic_target = Critic(self.latent_dim, self.action_dim).to(device)
        self.actor_target  = Actor(self.latent_dim,  self.action_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        #print(self.actor_target.encoder.fc_1.bias)
        #print(self.critic_target.encoder.fc_1.bias)

        self.decoder = Decoder(self.latent_dim).to(device)

        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder.parameters(), lr=encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(),        lr=decoder_lr)
        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(),          lr=actor_lr)
        self.critic_optimizer  = torch.optim.Adam(self.critic.parameters(),         lr=critic_lr)

        self.actor.train(True)
        self.critic.train(True)
        self.decoder.train(True)
        self.actor_target.train(True)
        self.critic_target.train(True)

    def update_function(self):
        if len(self.memory.memory_buffer) <= self.batch_size:
            return
        else:

            self.update_counter += 1

            state_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.memory.sample_experiences_from_buffer(self.batch_size)

            #self.actor.encoder.load_state_dict(self.critic.encoder.state_dict())

            print("inicio")
            for p1, p2 in zip(self.critic.encoder.parameters(), self.actor.encoder.parameters()):
                print(torch.equal(p1, p2))

            # first update the critic part
            with torch.no_grad():
                next_actions = self.actor_target(next_states_batch)
                target_noise = 0.2 * torch.randn_like(next_actions)
                target_noise = target_noise.clamp_(-0.5, 0.5)
                next_actions = next_actions + target_noise
                next_actions = next_actions.clamp_(-2, 2)

                next_q_values_q1, next_q_values_q2 = self.critic_target.forward(next_states_batch, next_actions)
                q_min   = torch.minimum(next_q_values_q1, next_q_values_q2)
                q_target = rewards_batch + (self.gamma * (1 - dones_batch) * q_min)

            q1, q2 = self.critic.forward(state_batch, actions_batch)

            critic_loss_1 = F.mse_loss(q1, q_target)
            critic_loss_2 = F.mse_loss(q2, q_target)
            critic_loss_total = critic_loss_1 + critic_loss_2

            self.critic_optimizer.zero_grad()
            critic_loss_total.backward()
            self.critic_optimizer.step()

            '''
            # second update the actor and soft updates of targets networks
            if self.update_counter % self.policy_freq_update == 0:
                action_actor       = self.actor.forward(state_batch, detach_encoder=True)
                actor_q1, actor_q2 = self.critic.forward(state_batch, action_actor, detach_encoder=True)
                actor_q_min = torch.minimum(actor_q1, actor_q2)
                actor_loss  = - actor_q_min.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ------------------------------------- Update target networks --------------- #
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


            # Third update the autoencoder
            z_vector = self.critic.encoder(state_batch)
            rec_obs  = self.decoder(z_vector)

            rec_loss    = F.mse_loss(state_batch, rec_obs)
            latent_loss = (0.5 * z_vector.pow(2).sum(1)).mean()  # add L2 penalty on latent representation

            ae_loss = rec_loss + self.decoder_latent_lambda * latent_loss

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            ae_loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            
            '''

    def select_action_from_policy(self, state_image_pixel):
        state_image_tensor  = torch.FloatTensor(state_image_pixel)
        state_image_tensor  = state_image_tensor.unsqueeze(0)  # torch.Size([1, 64, 64, 3])
        state_image_tensor  = state_image_tensor.permute(0, 3, 1, 2).to(device)  # torch.Size([1, 3, 64, 64])
        with torch.no_grad():
            action = self.actor.forward(state_image_tensor)
            action = action.cpu().data.numpy()
        return action[0]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def pre_pro_image(image_array):
    #crop_image = image_array[110:390, 110:390]
    resized     = cv2.resize(image_array, (64, 64), interpolation=cv2.INTER_AREA)
    norm_image  = resized / 255.0
    #norm_image  = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    state_image = norm_image  # (64, 64, 3)
    #state_image = np.expand_dims(state_image, axis=0)  # (1, 64, 64, 3)
    #cv2.imshow("Normalized image", state_image)
    #cv2.waitKey(10)
    return state_image

def plot_reward(reward_vector):
    plt.title("Rewards")
    plt.plot(reward_vector)
    plt.show()


def run_random_exploration(env, agent,  num_exploration_episodes=100, episode_horizont=200):
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


def run_training_rl_method(env, agent, num_episodes_training=1000, episode_horizont=200):
    total_reward = []
    for episode in range(1, num_episodes_training + 1):
        env.reset()
        state_image = env.render(mode='rgb_array')
        state_image = pre_pro_image(state_image)
        episode_reward = 0
        for step in range(1, episode_horizont + 1):
            #action = env.action_space.sample()
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


def main():
    env   = gym.make('Pendulum-v1')
    agent = RLAgent()

    #run_random_exploration(env, agent)
    run_training_rl_method(env, agent)
    env.close()


if __name__ == '__main__':
    main()