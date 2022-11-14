import gym
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random


class Memory:
    def __init__(self, replay_max_size=40_000, device="gpu"):
        self.device          = device
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

        state_batch_tensor      = torch.FloatTensor(state_batch).to(self.device)
        action_batch_tensor     = torch.FloatTensor(action_batch).to(self.device)
        reward_batch_tensor     = torch.FloatTensor(reward_batch).to(self.device)
        done_batch_tensor       = torch.FloatTensor(done_batch).to(self.device)
        next_batch_state_tensor = torch.FloatTensor(next_state_batch).to(self.device)

        # just put in the right order [b, 3, H, W]
        #state_batch_tensor      = state_batch_tensor.permute(0, 3, 1, 2)
        #next_batch_state_tensor = next_batch_state_tensor.permute(0, 3, 1, 2)

        return state_batch_tensor, action_batch_tensor, reward_batch_tensor, next_batch_state_tensor, done_batch_tensor
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class FrameStack:
    def __init__(self, k=3, env=gym.make('Pendulum-v1')):
        self.k = k
        self.frames_stacked = deque([], maxlen=k)
        self.env = env

    def reset(self):
        self.env.reset()
        obs = self.env.render(mode='rgb_array')
        obs = self.pre_pro_image(obs)
        for _ in range(self.k):
            self.frames_stacked.append(obs)
        #stacked_vector = np.concatenate(list(self.frames_stacked), axis=0)
        stacked_vector = np.array(list(self.frames_stacked))
        return stacked_vector

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self.env.render(mode='rgb_array')
        obs = self.pre_pro_image(obs)
        self.frames_stacked.append(obs)
        stacked_vector = np.array(list(self.frames_stacked))
        #stacked_vector = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_vector, reward, done, info

    def pre_pro_image(self, image_array):
        resized = cv2.resize(image_array, (84, 84), interpolation=cv2.INTER_AREA)
        gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        norm_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
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

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi



class Decoder(nn.Module):
    def __init__(self, encoder_feature_dim):
        super(Decoder, self).__init__()

        self.feature_dim = encoder_feature_dim

        self.num_layers = 4
        num_filters     = 32

        self.fc = nn.Linear(self.feature_dim, 39200)

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1))
        self.deconvs.append(nn.ConvTranspose2d(num_filters, 3, 3, stride=2, output_padding=1))

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h
        x = h.view(-1, 32, 35, 35)

        for i in range(0, self.num_layers - 1):
            x = torch.relu(self.deconvs[i](x))
            self.outputs['deconv%s' % (i + 1)] = x

        obs = self.deconvs[-1](x)
        self.outputs['obs'] = obs
        return obs



class Encoder(nn.Module):
    def __init__(self, encoder_feature_dim):
        super(Encoder, self).__init__()

        self.feature_dim = encoder_feature_dim

        self.num_layers = 4
        num_filters     = 32

        self.convs = nn.ModuleList([nn.Conv2d(3, num_filters, 3, stride=2)])

        for i in range(self.num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        self.fc = nn.Linear(num_filters * 35 * 35, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def forward_conv(self, obs):
        self.outputs['obs'] = obs
        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv
        h = torch.flatten(conv, start_dim=1)  # torch.Size([batch_size, 39200])
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc
        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm
        out = torch.tanh(h_norm)
        self.outputs['tanh'] = out
        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

class Actor(nn.Module):
    def __init__(self, encoder_feature_dim):
        super(Actor, self).__init__()

        self.encoder = Encoder(encoder_feature_dim)

        self.log_std_min = -20
        self.log_std_max = 2

        action_shape = 1

        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * action_shape)  # multiplico por dos ya que esto regresa mu and log
        )
        self.outputs = dict()

        self.apply(weight_init)


    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        self.outputs['mu']  = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std   = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi      = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        mu = mu * 2.0  # this is for pendulum only

        return mu, pi, log_pi, log_std


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


class Critic(nn.Module):
    def __init__(self, encoder_feature_dim):
        super(Critic, self).__init__()

        action_shape = 1
        hidden_dim   = 1024
        self.encoder = Encoder(encoder_feature_dim)

        self.Q1 = QFunction(encoder_feature_dim, action_shape, hidden_dim)
        self.Q2 = QFunction(encoder_feature_dim, action_shape, hidden_dim)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        q1  = self.Q1(obs, action)
        q2  = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class RLAgent:
    def __init__(self, device):
        action_shape = 1

        encoder_lr = 1e-3
        decoder_lr = 1e-3

        actor_lr   = 1e-3
        critic_lr  = 1e-3

        self.G = 1
        self.update_counter     = 0
        self.policy_freq_update = 2

        self.device = device
        self.gamma  = 0.99

        self.tau         = 0.005
        self.tau_encoder = 0.001

        self.memory = Memory(replay_max_size=40_000, device=self.device)

        self.batch_size = 32
        self.latent_dim = 50

        encoder_feature_dim = 50

        self.actor  = Actor(encoder_feature_dim).to(self.device)
        self.critic = Critic(encoder_feature_dim).to(self.device)
        self.critic_target = Critic(encoder_feature_dim).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        init_temperature = 0.01
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True

        self.target_entropy = -np.prod(action_shape)

        self.decoder = Decoder(encoder_feature_dim).to(device)
        self.decoder.apply(weight_init)


        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder.parameters(), lr=encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=decoder_lr, weight_decay=1e-7)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(0.9, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=(0.9, 0.999))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3, betas=(0.9, 0.999))

        self.actor.train(True)
        self.critic.train(True)
        self.decoder.train(True)
        self.critic_target.train(True)

    @property
    def alpha(self):
        return self.log_alpha.exp()


    def select_action_from_policy(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs)
            obs = obs.unsqueeze(0).to(self.device)
            #obs = obs.permute(0, 3, 1, 2).to(self.device)  # torch.Size([1, 3, 84, 84])
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            action = mu.cpu().data.numpy().flatten()
        return action

    def update_function(self):
        if len(self.memory.memory_buffer) <= self.batch_size:
            return
        else:
            for _ in range(1, self.G+1):
                self.update_counter += 1

                state_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.memory.sample_experiences_from_buffer(self.batch_size)

                with torch.no_grad():
                    _, policy_action, log_pi, _ = self.actor(next_states_batch)
                    target_Q1, target_Q2        = self.critic_target(next_states_batch, policy_action)
                    target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
                    target_Q = rewards_batch + (1 - dones_batch) * self.gamma * target_V

                current_Q1, current_Q2 = self.critic(state_batch, actions_batch)
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                if self.update_counter % self.policy_freq_update == 0:
                    _, pi, log_pi, log_std = self.actor(state_batch, detach_encoder=True)
                    actor_Q1, actor_Q2     = self.critic(state_batch, pi, detach_encoder=True)
                    actor_Q                = torch.min(actor_Q1, actor_Q2)
                    actor_loss             = (self.alpha.detach() * log_pi - actor_Q).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    self.log_alpha_optimizer.zero_grad()
                    alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
                    alpha_loss.backward()
                    self.log_alpha_optimizer.step()

                    for param, target_param in zip(self.critic.Q1.parameters(), self.critic_target.Q1.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.critic.Q2.parameters(), self.critic_target.Q2.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.critic.encoder.parameters(), self.critic_target.encoder.parameters()):
                        target_param.data.copy_(self.tau_encoder * param.data + (1 - self.tau_encoder) * target_param.data)

                h = self.critic.encoder(state_batch)
                rec_obs = self.decoder(h)

                rec_loss = F.mse_loss(state_batch, rec_obs)

                latent_loss = (0.5 * h.pow(2).sum(1)).mean()
                loss        = rec_loss + 1e-6 * latent_loss

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()


def pre_pro_image(image_array, bits=5):
    resized     = cv2.resize(image_array, (84, 84), interpolation=cv2.INTER_AREA)
    norm_image  = resized / 255.0
    state_image = norm_image  # (84, 84, 3)
    #cv2.imshow("Normalized image", state_image)
    #cv2.waitKey(10)
    return state_image


def plot_reward(reward_vector):
    plt.title("Rewards")
    plt.plot(reward_vector)
    plt.show()


def run_training_rl_method(env, agent, frames_stack, num_episodes_training=2000, episode_horizont=200):
    total_reward = []
    for episode in range(1, num_episodes_training + 1):
        #env.reset()
        #state_image = env.render(mode='rgb_array')
        #state_image = pre_pro_image(state_image)
        state_image = frames_stack.reset()
        episode_reward = 0
        for step in range(1, episode_horizont + 1):
            # action = env.action_space.sample()
            action = agent.select_action_from_policy(state_image)

            #obs_next_state_vector, reward, done, info = env.step(action)
            #new_state_image = env.render(mode='rgb_array')
            #new_state_image = pre_pro_image(new_state_image)
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


def run_random_exploration(env, agent, frames_stack,  num_exploration_episodes=200, episode_horizont=200):
    print("exploration start")
    for episode in range(1, num_exploration_episodes + 1):
        #env.reset()
        #state_image = env.render(mode='rgb_array')  # return the rendered image and can be used as input-state image
        #state_image = pre_pro_image(state_image)
        state_image = frames_stack.reset()
        for step in range(1, episode_horizont + 1):
            action = env.action_space.sample()
            new_state_image, reward, done, _ = frames_stack.step(action)
            #new_state_image = env.render(mode='rgb_array')
            #new_state_image = pre_pro_image(new_state_image)
            agent.memory.save_experience_to_buffer(state_image, action, reward, new_state_image, done)
            state_image = new_state_image
            if done:
                break
    print("exploration end")


def main():
    env = gym.make('Pendulum-v1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    min_action_value = env.action_space.low.min()  # --> -2
    max_action_value = env.action_space.high.max()  # --> +2

    obs_shape    = env.observation_space.shape  # --> 3
    action_shape = env.action_space.shape  # --> 1

    agent = RLAgent(device)
    frames_stack = FrameStack(env=env)
    run_random_exploration(env, agent, frames_stack)
    run_training_rl_method(env, agent, frames_stack)
    env.close()


if __name__ == '__main__':
    main()
