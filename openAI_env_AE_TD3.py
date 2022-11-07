
import cv2
import torch
import torch.nn as nn
import gym

from collections import deque
import random
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Working with GPU")
else:
    device = torch.device('cpu')
    print("Working with CPU")


class Memory:

    def __init__(self, replay_max_size=10_000):
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
        done_batch_tensor       = torch.FloatTensor(done_batch)
        next_batch_state_tensor = torch.FloatTensor(next_state_batch).to(device)

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


    def forward(self, x):
        x = self.cov_net(x)                # torch.Size([batch_size, 64, 27, 27])
        x = torch.flatten(x, start_dim=1)  # torch.Size([batch_size, 46656])
        x = self.fc_1(x)
        x = self.ln(x)
        out = torch.tanh(x)  # not 100% sure but this may help to normalize the output between -1 and 1
        return out

    def copy_conv_weights_from(self, source):
        """still no sure why we need this"""
        """Tie convolutional layers but only tie conv layers"""




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
            nn.ReLU(),  # no sure but could also be a nn.Sigmoid here
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

    def forward(self, state):
        z_vector    = self.encoder.forward(state)
        z_vector_in = z_vector.detach()

        output = self.act_net(z_vector_in)
        output = torch.tanh(output) * 2.0
        return output





class Critic(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(Critic, self).__init__()

        self.encoder = Encoder(latent_dim)

        input_critic_size = latent_dim + action_dim

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

    def forward(self, state, action):
        z_vector = self.encoder.forward(state)
        x_in = torch.cat([z_vector, action], dim=1)
        q1   = self.Q1(x_in)
        q2   = self.Q2(x_in)
        return q1, q2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class RLAgent(object):
    """TD3 + AE algorithm"""
    def __init__(self):
        self.memory = Memory()

        self.batch_size = 64
        self.latent_dim = 32
        self.action_dim = 1

        self.critic = Critic(self.latent_dim, self.action_dim).to(device)
        self.actor  = Actor(self.latent_dim,  self.action_dim).to(device)

        self.critic_target = Critic(self.latent_dim, self.action_dim).to(device)
        self.actor_target  = Actor(self.latent_dim,  self.action_dim).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)


    def update_function(self):
        if len(self.memory.memory_buffer) <= self.batch_size:
            return
        else:
            state_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.memory.sample_experiences_from_buffer(self.batch_size)

            #q1, q2 = self.critic(state_batch, actions_batch)

            action = self.actor(state_batch)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def pre_pro_image(image_array):
    #crop_image = image_array[110:390, 110:390]
    resized     = cv2.resize(image_array, (64, 64), interpolation=cv2.INTER_AREA)
    #gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm_image  = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    state_image = norm_image  # (64, 64, 3)
    #state_image = np.expand_dims(state_image, axis=0)  # (1, 64, 64, 3)
    #cv2.imshow("Normalized image", state_image)
    #cv2.waitKey(10)
    return state_image


def run_training_rl_method(env, agent, num_episodes_training=100, episode_horizont=200):
    for episode in range(1, num_episodes_training + 1):
        env.reset()
        state_image = env.render(mode='rgb_array')
        state_image = pre_pro_image(state_image)
        for step in range(1, episode_horizont + 1):
            action = env.action_space.sample()
            obs_next_state_vector, reward, done, info = env.step(action)
            new_state_image = env.render(mode='rgb_array')
            new_state_image = pre_pro_image(new_state_image)
            agent.memory.save_experience_to_buffer(state_image, action, reward, new_state_image, done)
            state_image = new_state_image
            if done:
                break
            agent.update_function()





def main():
    env   = gym.make('Pendulum-v1')
    agent = RLAgent()
    run_training_rl_method(env, agent)
    env.close()




if __name__ == '__main__':
    main()