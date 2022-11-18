"""
SAC using the env observation space from environment
observation_space_size = 3
action_space_size      = 1
SAC version 2
the difference wrt versio one is that here there is not a value function network
and there is an alpha value which is automatically upload
REFERENCES CODE:
    https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py#L32
    https://spinningup.openai.com/en/latest/algorithms/sac.html

status = working
"""
import gym
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from memory_utilities import MemoryClass
from networks_architectures import SoftQNetworkSAC, PolicyNetworkSAC, PolicyNetworkSACDeterministic


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Working with GPU")
else:
    device = torch.device('cpu')
    print("Working with CPU")


class SACAgent:
    def __init__(self):
        self.device = device

        self.G          = 1  # internal loop for Policy update
        self.num_action = 1
        self.obs_size   = 3  # this is the obs space vector size

        # -------- Hyper-parameters --------------- #
        self.batch_size = 32

        self.gamma           = 0.99
        self.tau             = 0.005
        self.max_memory_size = 10_000

        self.update_counter = 0
        self.freq_update    = 2

        self.alpha_lr  = 1e-3
        self.soft_q_lr = 1e-3
        self.policy_lr = 1e-3

        self.noise = torch.Tensor(self.num_action).to(self.device)

        # -------- Helper Functions --------------- #
        self.memory = MemoryClass(self.max_memory_size)

        self.soft_q_net        = SoftQNetworkSAC(self.obs_size, self.num_action).to(self.device)  # Q function, could be considered Critic
        self.soft_q_net_target = copy.deepcopy(self.soft_q_net)

        self.policy_net = PolicyNetworkSAC(self.obs_size, self.num_action).to(self.device)  # could be considered Actor

        self.target_entropy = -self.num_action  # −dim(A)
        self.alpha          = 0.0
        self.log_alpha      = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)

        self.soft_q_optimizer  = optim.Adam(self.soft_q_net.parameters(), lr=self.soft_q_lr)
        self.policy_optimizer  = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.alpha_optimizer   = optim.Adam([self.log_alpha], lr=self.alpha_lr)

        self.policy_net_deterministic       = PolicyNetworkSACDeterministic(self.obs_size, self.num_action).to(self.device)
        self.policy_deterministic_optimizer = optim.Adam(self.policy_net_deterministic.parameters(), lr=self.policy_lr)


    def sample_from_policy(self, state):
        mean, log_std = self.policy_net(state)
        std    = log_std.exp()
        normal = Normal(mean, std)
        x_t    = normal.rsample()
        y_t    = torch.tanh(x_t)

        epsilon      = 1e-6
        action_scale = 2.0
        action = y_t * action_scale

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * action_scale
        return action, log_prob, mean


    def sample_from_policy_deterministic(self, state):
        mean   = self.policy_net_deterministic(state)
        noise  = self.noise.normal_(0., std=0.1)
        noise  = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean


    def get_action_from_policy(self, state):
        state_tensor = torch.FloatTensor(state)
        state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.to(self.device)

        action, _, _ = self.sample_from_policy(state_tensor)
        #action, _, _ = self.sample_from_policy_deterministic(state_tensor)
        action = action.detach().cpu().numpy()
        return action[0]


    def update_function(self):
        self.policy_learning_function()
        #self.policy_deterministic_function()


    def policy_deterministic_function(self):
        # Here alpha value is no automatically adapted so it will be not  an automatic_entropy_tuning
        if len(self.memory.memory_buffer_experiences) <= self.batch_size:
            return
        else:
            for it in range(1, self.G + 1):
                self.update_counter += 1

                states, actions, rewards, next_states, dones = self.memory.sample_frame_vector_experiences(self.batch_size)
                states, actions, rewards, next_states, dones = self.prepare_tensor(states, actions, rewards, next_states, dones)

                with torch.no_grad():
                    next_state_action, next_state_log_pi, _ = self.sample_from_policy_deterministic(next_states)
                    qf1_next_target, qf2_next_target        = self.soft_q_net_target(next_states, next_state_action)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    next_q_value       = rewards + (1 - dones) * self.gamma * min_qf_next_target

                qf1, qf2 = self.soft_q_net(states, actions)
                qf1_loss = F.mse_loss(qf1, next_q_value)
                qf2_loss = F.mse_loss(qf2, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                self.soft_q_optimizer.zero_grad()
                qf_loss.backward()
                self.soft_q_optimizer.step()

                pi, log_pi, _  = self.sample_from_policy_deterministic(states)
                qf1_pi, qf2_pi = self.soft_q_net(states, pi)
                min_qf_pi      = torch.min(qf1_pi, qf2_pi)

                policy_loss = - min_qf_pi.mean()

                self.policy_deterministic_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_deterministic_optimizer.step()

                if self.update_counter % self.freq_update == 0:
                    for target_param, param in zip(self.soft_q_net_target.parameters(), self.soft_q_net.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


    def policy_learning_function(self):
        if len(self.memory.memory_buffer_experiences) <= self.batch_size:
            return
        else:
            for it in range(1, self.G + 1):
                self.update_counter += 1

                states, actions, rewards, next_states, dones = self.memory.sample_frame_vector_experiences(self.batch_size)
                states, actions, rewards, next_states, dones = self.prepare_tensor(states, actions, rewards, next_states, dones)

                with torch.no_grad():
                    next_state_action, next_state_log_pi, _ = self.sample_from_policy(next_states)
                    qf1_next_target, qf2_next_target        = self.soft_q_net_target(next_states, next_state_action)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi

                    #next_q_value = rewards + dones * self.gamma * (min_qf_next_target) # this not work is actually a problem
                    next_q_value = rewards + (1 - dones) * self.gamma * min_qf_next_target

                qf1, qf2 = self.soft_q_net(states, actions)
                qf1_loss = F.mse_loss(qf1, next_q_value)
                qf2_loss = F.mse_loss(qf2, next_q_value)
                qf_loss  = qf1_loss + qf2_loss

                self.soft_q_optimizer.zero_grad()
                qf_loss.backward()
                self.soft_q_optimizer.step()

                pi, log_pi, _  = self.sample_from_policy(states)
                qf1_pi, qf2_pi = self.soft_q_net(states, pi)
                min_qf_pi      = torch.min(qf1_pi, qf2_pi)

                policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp()

                if self.update_counter % self.freq_update == 0:
                    for target_param, param in zip(self.soft_q_net_target.parameters(), self.soft_q_net.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


    def prepare_tensor(self, states, actions, rewards, next_states, dones):
        states  = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        dones   = np.array(dones).reshape(-1, 1)
        next_states = np.array(next_states)

        states  = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones   = torch.FloatTensor(dones).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        return states, actions, rewards, next_states, dones


    def plot_functions(self, rewards):
        plt.title("Rewards")
        plt.plot(rewards)
        plt.savefig(f"plot_results/Normal-SAC_pendulum_reward_curve.png")
        np.savetxt(f"plot_results/Normal-SAC_pendulum_reward_curve.txt", rewards)

    def save_model(self):
        torch.save(self.policy_net.state_dict(), f'trained_models/Normal-SAC_policy_net_pendulum.pht')
        print("models have been saved...")



def run_exploration(env, agent,  num_exploration_episodes, episode_horizont):
    print("exploration start")
    for episode in tqdm(range(1, num_exploration_episodes + 1)):
        state = env.reset()
        for step in range(1, episode_horizont):
            action = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)
            agent.memory.save_frame_vector_experience_buffer(state, action, reward, new_state, done)
            state = new_state
            if done:
                break
    print("exploration end")



def run_training_rl_method(env, agent, num_episodes_training=100, episode_horizont=200):
    rewards = []
    for episode in range(1, num_episodes_training + 1):
        print(f"-----------------Episode {episode}-----------------------------")
        episode_reward = 0
        state = env.reset()
        for step in range(1, episode_horizont + 1):
            action = agent.get_action_from_policy(state)
            new_state, reward, done, _ = env.step(action)
            agent.memory.save_frame_vector_experience_buffer(state, action, reward, new_state, done)
            state = new_state
            episode_reward += reward
            if done:
                break
            agent.update_function()
        rewards.append(episode_reward)
        print(f"Episode {episode} End, Total reward: {episode_reward}")
    agent.plot_functions(rewards)
    agent.save_model()




def main_run():
    env = gym.make('Pendulum-v1')
    agent = SACAgent()

    episode_horizont          = 200
    num_episodes_exploration  = 200
    num_episodes_training     = 500

    run_exploration(env, agent, num_episodes_exploration, episode_horizont)
    run_training_rl_method(env, agent, num_episodes_training, episode_horizont)
    env.close()


if __name__ == '__main__':
    main_run()