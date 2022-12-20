"""
TD3 using the env observation space from environment

env = gym.make('Pendulum-v1')
env = gym.make("BipedalWalker-v3")

Status = Working

"""
import gym
import copy
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F

from openAI_architectures_utilities  import Actor_Normal, Critic_Normal
from openAI_memory_utilities import Memory


class TD3Agent:
    def __init__(self, env, memory_size, device, batch_size):

        self.device = device
        self.env    = env

        # -------- Hyper-parameters --------------- #
        self.action_dim       = env.action_space.shape[0]
        self.obs_size         = env.observation_space.shape[0]  # this is the obs space vector size
        self.max_action_value = env.action_space.high.max()
        self.env_name         = env.unwrapped.spec.id

        self.G = 10  # internal loop for Policy update

        self.gamma      = 0.99
        self.tau        = 0.005
        self.batch_size = batch_size

        self.update_counter     = 0
        self.policy_freq_update = 2
        self.max_memory_size    = memory_size

        self.lr_critic  = 3e-4 #1e-3
        self.lr_actor   = 3e-4 #1e-4

        # -------- Models --------------- #
        self.memory = Memory(self.max_memory_size, self.device)

        # ---- Initialization Models --- #
        self.critic = Critic_Normal(self.obs_size, self.action_dim).to(self.device)
        self.actor  = Actor_Normal(self.obs_size,  self.action_dim, self.max_action_value).to(self.device)

        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.actor.train()
        self.critic.train()

    def update_function(self):
        self.policy_learning_function()

    def policy_learning_function(self):
        if len(self.memory.memory_buffer) <= self.batch_size:
            return

        else:
            self.update_counter += 1  # this is used for delay/

            states, actions, rewards, next_states, dones = self.memory.sample_experiences_from_buffer(self.batch_size)

            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                target_noise = 0.2 * torch.randn_like(next_actions)
                target_noise = target_noise.clamp(-0.5, 0.5)
                next_actions = next_actions + target_noise
                next_actions = next_actions.clamp(-self.max_action_value, self.max_action_value)

                next_q_values_q1, next_q_values_q2 = self.critic_target(next_states, next_actions)
                q_min = torch.min(next_q_values_q1, next_q_values_q2)

                q_target = rewards + (self.gamma * (1 - dones) * q_min)

            q_vals_q1, q_vals_q2 = self.critic(states, actions)

            critic_loss_1 = F.mse_loss(q_vals_q1, q_target)
            critic_loss_2 = F.mse_loss(q_vals_q2, q_target)
            critic_loss_total = critic_loss_1 + critic_loss_2

            self.critic_optimizer.zero_grad()
            critic_loss_total.backward()
            torch.nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=10.0) # still no sure about this
            self.critic_optimizer.step()

            if self.update_counter % self.policy_freq_update == 0:
                # ------- calculate the actor loss
                actor_q1, actor_q2 = self.critic(states, self.actor(states))

                # original paper work here with Q1 only
                #actor_loss = - actor_q1.mean()

                actor_q_min = torch.min(actor_q1, actor_q2)
                actor_loss  = - actor_q_min.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=10.0)
                self.actor_optimizer.step()
                # ------------------------------------- Update target networks --------------- #
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_models(self):
        torch.save(self.actor.state_dict(), f'trained_models/Normal-TD3_actor_{self.env_name}.pht')
        print("models have been saved...")

    def load_models(self):
        self.actor.load_state_dict(torch.load(f'trained_models/Normal-TD3_actor_{self.env_name}.pht'))
        print("models has been loaded...")


    def get_action_from_policy(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.to(self.device)
            action = self.actor(state_tensor)
            action = action.cpu().data.numpy().flatten()
        return action

    def plot_functions(self, rewards):
        plt.title("Rewards")
        plt.plot(rewards)
        plt.savefig(f"plot_results/Normal-TD3_{self.env_name}_reward_curve.png")
        np.savetxt(f"plot_results/Normal-TD3_{self.env_name}_reward_curve.txt", rewards)
        print("plots have been saved...")


def run_training_rl_method(env, agent, max_value, num_episodes_training, episode_horizont):
    rewards = []
    for episode in range(1, num_episodes_training + 1):
        print(f"-----------------Episode {episode}-----------------------------")
        episode_reward = 0
        state = env.reset()
        for step in range(1, episode_horizont + 1):
            action = agent.get_action_from_policy(state)
            noise  = np.random.normal(0, scale=0.1*max_value, size=env.action_space.shape[0])
            action = action + noise
            action = np.clip(action, -max_value, max_value)
            new_state, reward, done, _ = env.step(action)
            agent.memory.save_experience_to_buffer(state, action, reward, new_state, done)
            state = new_state
            episode_reward += reward
            for _ in range(1, 10):
                agent.update_function()
            if done:
                break
        rewards.append(episode_reward)
        print(f"Episode {episode} End, Total reward: {episode_reward}")
    agent.save_models()
    agent.plot_functions(rewards)



def run_exploration(env, agent,  num_exploration_episodes, episode_horizont):
    print("exploration start")
    for _ in tqdm(range(1, num_exploration_episodes + 1)):
        state = env.reset()
        for step in range(1, episode_horizont):
            action = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)
            agent.memory.save_experience_to_buffer(state, action, reward, new_state, done)
            state = new_state
            if done:
                break
    print("exploration end")

def evaluation_function(agent, env):
    agent.load_models()

    episodes_test = 10
    seed = 200
    env.seed(seed)

    for episode in range(1, episodes_test + 1):
        episode_reward = 0
        state = env.reset()
        done  = False
        while not (done is True):
            env.render()
            action = agent.get_action_from_policy(state)
            new_state, reward, done, _ = env.step(action)
            state = new_state
            episode_reward += reward
        print(episode_reward)


def main_run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #env = gym.make('Pendulum-v1')
    env = gym.make("BipedalWalker-v3")

    env_name         = env.unwrapped.spec.id
    max_action_value = env.action_space.high.max()
    batch_size = 256
    seed       = 100


    if env_name == "Pendulum-v1":
        num_episodes_exploration = 100
        num_episodes_training    = 20
        episode_horizont         = 200
        memory_size              = 20_000
    else:
        num_episodes_exploration = 100
        num_episodes_training    = 1000
        episode_horizont         = 1600
        memory_size              = int(num_episodes_exploration*episode_horizont)

    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = TD3Agent(env, memory_size, device, batch_size)
    run_exploration(env, agent, num_episodes_exploration, episode_horizont)
    run_training_rl_method(env, agent, max_action_value, num_episodes_training, episode_horizont)
    evaluation_function(agent, env)
    env.close()


if __name__ == '__main__':
    main_run()