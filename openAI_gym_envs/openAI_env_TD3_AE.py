"""
good
TD3 with deterministic AE for pendulum and BipedalWalker  env only
This is my version and  re-implementation of the paper  https://arxiv.org/pdf/1910.01741.pdf
however I removed or changed many part here. The original paper use SAC

Every state is rendered and passed as input to the autoencoder, after preprocessing it

original image 510 x 510 x 3 -->  Pendulum
               400 x 600 x 3 -->  BipedalWalker

Input for the encoder = 3 stacked frames, gray scalded, normalized and resized (84 , 84)
The input is batch-size x 3 x 84 x 84, where the stacked number takes the place of the channel for covnet

keys= initial exploration is very important, the activation function in the decoder helps
Status = Working
"""

import gym
import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from openAI_memory_utilities import Memory, FrameStack
from openAI_architectures_utilities import Actor, Critic, Decoder
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class RLAgent:
    def __init__(self, env, memory_size, device, batch_size, G, k):

        # env info
        self.max_action_value = env.action_space.high.max()
        self.env_name         = env.unwrapped.spec.id

        self.k      = k  # number of stacked frames

        self.G      = G
        self.env    = env
        self.device = device

        self.encoder_lr = 1e-3
        self.decoder_lr = 1e-3

        self.critic_lr = 1e-3  # 1e-3
        self.actor_lr  = 1e-4  # 1e-4

        self.tau         = 0.005 # 0.005
        self.tau_encoder = 0.005 # 0.001
        self.gamma       = 0.99

        self.update_counter     = 0
        self.policy_freq_update = 2

        self.batch_size = batch_size  # 32
        self.latent_dim = 50  # 50
        self.action_dim = env.action_space.shape[0]

        self.memory = Memory(memory_size, self.device)

        # main networks
        self.actor  = Actor(self.latent_dim, self.action_dim, self.max_action_value, self.k).to(self.device)
        self.critic = Critic(self.latent_dim, self.action_dim, self.k).to(self.device)

        # target networks
        self.actor_target  = Actor(self.latent_dim, self.action_dim, self.max_action_value, self.k).to(self.device)
        self.critic_target = Critic(self.latent_dim, self.action_dim, self.k).to(self.device)

        # tie encoders between actor and critic
        # with this, any changes in the critic encoder
        # will also be affecting the actor-encoder during the whole training

        #self.actor.encoder_net.copy_conv_weights_from(self.critic.encoder_net)
        self.actor.encoder_net.copy_all_weights_from(self.critic.encoder_net)  # original paper just copy cov weights

        # copy weights and bias from main to target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        # main Decoder
        self.decoder = Decoder(self.latent_dim, self.k).to(device)

        # Optimizer with default values
        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder_net.parameters(), lr=self.encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.decoder_lr, weight_decay=1e-7)
        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(),   lr=self.actor_lr)
        self.critic_optimizer  = torch.optim.Adam(self.critic.parameters(),  lr=self.critic_lr)

        self.actor.train(True)
        self.critic.train(True)
        self.critic_target.train(True)
        self.actor_target.train(True)
        self.decoder.train(True)

    def select_action_from_policy(self, state_image_pixel):
        with torch.no_grad():
            state_image_tensor = torch.FloatTensor(state_image_pixel)
            state_image_tensor = state_image_tensor.unsqueeze(0).to(self.device)
            action = self.actor(state_image_tensor)
            action = action.cpu().data.numpy()#.flatten()
        return action[0]

    def update_function(self):
        if len(self.memory.memory_buffer) <= self.batch_size:
            return
        else:
            for _ in range(1, self.G + 1):
                self.update_counter += 1

                state_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.memory.sample_experiences_from_buffer(self.batch_size)

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # update the critic part
                with torch.no_grad():
                    next_actions = self.actor_target(next_states_batch)
                    target_noise = 0.2 * torch.randn_like(next_actions)
                    target_noise = target_noise.clamp(-0.5, 0.5)
                    next_actions = next_actions + target_noise
                    next_actions = torch.clamp(next_actions, min=-self.max_action_value, max=self.max_action_value)

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

                    actor_q_min = torch.minimum(actor_q1, actor_q2)
                    actor_loss  = - actor_q_min.mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # ------------------------------------- Update target networks --------------- #
                    # change this Feb
                    '''
                    #problems
                    for param, target_param in zip(self.critic.Q1.parameters(), self.critic_target.Q1.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.critic.Q2.parameters(), self.critic_target.Q2.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.critic.encoder_net.parameters(), self.critic_target.encoder_net.parameters()):
                        target_param.data.copy_(self.tau_encoder * param.data + (1 - self.tau_encoder) * target_param.data)

                    for param, target_param in zip(self.actor.encoder_net.parameters(), self.actor_target.encoder_net.parameters()):
                        target_param.data.copy_(self.tau_encoder * param.data + (1 - self.tau_encoder) * target_param.data) 
                    '''
                    # working
                    # assuming tau encoder and tau actor-critic are the same
                    for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                    for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                    '''
                     #working
                    for target_param, param in zip(self.critic_target.Q1.parameters(), self.critic.Q1.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                    for target_param, param in zip(self.critic_target.Q2.parameters(), self.critic.Q2.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                    for target_param, param in zip(self.actor_target.act_net.parameters(), self.actor.act_net.parameters()):
                        target_param.data.copy_(param.data * self.tau_encoder + target_param.data * (1.0 - self.tau_encoder))

                    # Because the encoder may have a different tau
                    for target_param, param in zip(self.critic_target.encoder_net.parameters(), self.critic.encoder_net.parameters()):
                        target_param.data.copy_(param.data * self.tau_encoder + target_param.data * (1.0 - self.tau_encoder))

                    for target_param, param in zip(self.actor_target.encoder_net.parameters(), self.actor.encoder_net.parameters()):
                        target_param.data.copy_(param.data * self.tau_encoder + target_param.data * (1.0 - self.tau_encoder))
                    '''


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

    def save_models(self):
        torch.save(self.actor.state_dict(), f'trained_models/AE-TD3_actor_{self.env_name}.pht')
        torch.save(self.critic.encoder_net.state_dict(), f'trained_models/AE-TD3_encoder_{self.env_name}.pht')
        torch.save(self.decoder.state_dict(), f'trained_models/AE-TD3_decoder_{self.env_name}.pht')
        print("models have been saved...")

    def load_models(self):
        self.actor.load_state_dict(torch.load(f'trained_models/AE-TD3_actor_{self.env_name}.pht'))
        self.critic.encoder_net.load_state_dict(torch.load(f'trained_models/AE-TD3_encoder_{self.env_name}.pht'))
        self.decoder.load_state_dict(torch.load(f'trained_models/AE-TD3_decoder_{self.env_name}.pht'))
        print("models has been loaded...")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_reward(reward_vector, env_name):
    plt.title("Rewards")
    plt.plot(reward_vector)
    plt.savefig(f"plot_results/AE-TD3_{env_name}_reward_curve.png")
    np.savetxt(f"plot_results/AE-TD3_{env_name}_reward_curve.txt", reward_vector)
    # plt.show()


def plot_reconstructions(input_img, reconstruction_img, env_name):

    input_img = input_img[2]
    reconstruction_img = reconstruction_img[0][2]

    plt.subplot(1, 2, 1)
    plt.title("Image Input")
    plt.imshow(input_img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Image Reconstruction")
    plt.imshow(reconstruction_img, cmap='gray')

    plt.savefig(f"plot_results/AE-TD3_{env_name}_image_reconstruction.png")
    plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def run_random_exploration(env, agent, frames_stack, num_exploration_episodes, episode_horizont):
    print("exploration start")
    for _ in tqdm(range(1, num_exploration_episodes + 1)):
    #while len(agent.memory.memory_buffer) < agent.memory.replay_max_size:
        #print(f"{len(agent.memory.memory_buffer)}/{agent.memory.replay_max_size}")
        state_image = frames_stack.reset()
        for step in range(1, episode_horizont + 1):
            action = env.action_space.sample()
            new_state_image, reward, done, _, _ = frames_stack.step(action)
            agent.memory.save_experience_to_buffer(state_image, action, reward, new_state_image, done)
            state_image = new_state_image
            if done:
                break
    print("exploration end")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def run_training_rl_method(env, agent, max_action_value, env_name, frames_stack, num_episodes_training, episode_horizont):
    total_reward = []
    for episode in range(1, num_episodes_training + 1):
        state_image = frames_stack.reset()
        episode_reward = 0

        for step in range(1, episode_horizont + 1):

            action = agent.select_action_from_policy(state_image)
            noise  = np.random.normal(0, scale=0.1 * max_action_value, size=env.action_space.shape[0])
            action = action + noise
            action = np.clip(action, -max_action_value, max_action_value)
            new_state_image, reward, done, _, _ = frames_stack.step(action)
            agent.memory.save_experience_to_buffer(state_image, action, reward, new_state_image, done)
            state_image = new_state_image
            episode_reward += reward

            agent.update_function()

            if done:
                break


        total_reward.append(episode_reward)
        print(f"Episode {episode} End, Total reward: {episode_reward}")
    agent.save_models()
    plot_reward(total_reward, env_name)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def autoencoder_evaluation(agent, frames_stack, env_name, device):
    agent.load_models()
    state_image = frames_stack.reset()
    state_image_tensor = torch.FloatTensor(state_image)
    state_image_tensor = state_image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        z_vector = agent.critic.encoder_net(state_image_tensor)
        rec_obs  = agent.decoder(z_vector)
        rec_obs  = rec_obs.cpu().numpy()
    plot_reconstructions(state_image, rec_obs, env_name)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def policy_env_evaluation_function(agent, env, frames_stack):
    agent.load_models()

    episodes_test = 10
    seed          = 200
    env.seed(seed)

    for episode in range(1, episodes_test + 1):
        episode_reward = 0
        state_image    = frames_stack.reset()
        done = False
        while not (done is True):
            action = agent.select_action_from_policy(state_image)
            new_state_image, reward, done, _ = frames_stack.step(action)
            state_image = new_state_image
            episode_reward += reward
        print(episode_reward)


def define_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--k',          type=int, default=3)  # number of stacked frames
    parser.add_argument('--G',          type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed',       type=int, default=571)
    parser.add_argument('--env_name',   type=str, default='Pendulum-v1')  # BipedalWalker-v3
    args   = parser.parse_args()
    return args


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args   = define_parse_args()

    env      = gym.make(args.env_name, render_mode="rgb_array")
    env_name = args.env_name
    max_action_value = env.action_space.high.max()  # --> 2 for pendulum, 1 for Walker

    if env_name == "Pendulum-v1":
        num_exploration_episodes = 50 # 300
        num_training_episodes    = 50
        episode_horizont         = 200
        memory_size              = 1_000_000 # int(num_exploration_episodes*episode_horizont)
    else:
        num_exploration_episodes = 125
        num_training_episodes    = 1000
        episode_horizont         = 1600
        memory_size              = 200_000 # 320_000

    #env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent        = RLAgent(env, memory_size, device, args.batch_size, args.G, args.k)
    frames_stack = FrameStack(args.k, env)

    run_random_exploration(env, agent, frames_stack, num_exploration_episodes=num_exploration_episodes, episode_horizont=episode_horizont)
    run_training_rl_method(env, agent, max_action_value, env_name, frames_stack, num_episodes_training=num_training_episodes, episode_horizont=episode_horizont)
    #autoencoder_evaluation(agent, frames_stack, env_name, device)
    #policy_env_evaluation_function(agent, env, frames_stack)
    env.close()


if __name__ == '__main__':
    main()
