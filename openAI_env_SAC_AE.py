import gym
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



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



class Encoder(nn.Module):
    def __init__(self, encoder_feature_dim):
        super(Encoder, self).__init__()

        self.feature_dim = encoder_feature_dim

        self.num_layers = 2
        num_filters = 32

        self.convs = nn.ModuleList([nn.Conv2d(3, num_filters, 3, stride=2)])

        for i in range(self.num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        self.fc = nn.Linear(num_filters * 39 * 39, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def forward_conv(self, obs):
        self.outputs['obs'] = obs
        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        print(conv.shape)
        print(conv.size(1))
        print(conv.size(0))

        h = conv.view(conv.size(1), -1)
        print(h.shape)
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


class Actor(nn.Module):
    def __init__(self, encoder_feature_dim):
        super(Actor, self).__init__()

        self.encoder = Encoder(encoder_feature_dim)

        self.log_std_min = -10
        self.log_std_max = 2

        hidden_dim = 256

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,  1)
        )
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs_in, compute_pi=True, compute_log_pi=True, detach_encoder=False):

        obs_z = self.encoder(obs_in, detach=detach_encoder)

        '''
        mu, log_std = self.trunk(obs_z).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std
        '''






class RLAgent(object):
    def __init__(self, device):
        encoder_lr = 1e-3
        decoder_lr = 1e-3
        actor_lr = 1e-4
        critic_lr = 1e-3

        self.device = device
        self.gamma = 0.99
        self.tau = 0.005

        self.batch_size = 32
        self.latent_dim = 50

        encoder_feature_dim = 50

        self.actor = Actor(encoder_feature_dim).to(self.device)

    def select_action_from_policy(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs)
            obs = obs.unsqueeze(0)
            obs = obs.permute(0, 3, 1, 2).to(self.device)  # torch.Size([1, 3, 84, 84])

            self.actor(obs, compute_pi=False, compute_log_pi=False)

            #mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)

            #action = mu.cpu().data.numpy().flatten()

        #print(action)





def pre_pro_image(image_array, bits=5):
    #image_array_tensor = torch.FloatTensor(image_array.copy())
    #assert image_array_tensor.dtype == torch.float32
    #bins = 2 ** bits
    #obs = torch.floor(image_array_tensor / 2 ** (8 - bits))
    #obs = obs / bins
    #obs = obs + torch.rand_like(obs) / bins
    #obs = obs - 0.5

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


def run_training_rl_method(env, agent, num_episodes_training=1000, episode_horizont=200):
    total_reward = []
    for episode in range(1, num_episodes_training + 1):
        env.reset()
        state_image = env.render(mode='rgb_array')
        state_image = pre_pro_image(state_image)
        episode_reward = 0
        for step in range(1, episode_horizont + 1):
            pass
            # action = env.action_space.sample()
            action = agent.select_action_from_policy(state_image)
            '''
            
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
    '''


def main():
    env = gym.make('Pendulum-v1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    min_action_value = env.action_space.low.min()  # --> -2
    max_action_value = env.action_space.high.max()  # --> +2

    obs_shape = env.observation_space.shape  # --> 3
    action_shape = env.action_space.shape  # --> 1

    agent = RLAgent(device)

    run_training_rl_method(env, agent)


if __name__ == '__main__':
    main()
