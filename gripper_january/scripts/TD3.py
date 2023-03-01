
import copy
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

#from Networks_Architectures import Actor_Normal as Actor
#from Networks_Architectures import Critic_Normal as Critic

from Networks_Architectures import Actor_Lineal  as Actor
from Networks_Architectures import Critic_Lineal as Critic


class TD3:
    def __init__(self, device, obs_dim, act_dim):

        self.device  = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.max_action_value =  1
        self.min_action_value = -1

        self.gamma      = 0.99
        self.tau        = 0.005  # 0.005

        self.lr_actor  = 1e-4 # 1e-3 # 3e-4 # 1e-4
        self.lr_critic = 1e-3 # 1e-3 # 3e-4 # 1e-3

        self.update_counter     = 0
        self.policy_freq_update = 2

        # main networks
        self.actor  = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = Critic(self.obs_dim, self.act_dim).to(self.device)

        # target networks
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.actor.train(True)
        self.critic.train(True)
        self.critic_target.train(True)
        self.actor_target.train(True)

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.to(self.device)
            _, action = self.actor(state_tensor)
            action = action.cpu().data.numpy().flatten()
        return action

    def action_sample(self):
        action = []
        for i in range(0, self.act_dim):
            a = np.clip(random.uniform(-1, 1), self.min_action_value, self.max_action_value)
            action.append(a)
        return action

    def learn(self, experiences):
        self.update_counter += 1

        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)

        # Convert into tensor
        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        #dones       = torch.LongTensor(dones).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Reshape in the right order
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones   = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            _, next_actions = self.actor_target(next_states)
            target_noise = 0.2 * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -0.5, 0.5)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=self.min_action_value, max=self.max_action_value)

            target_q_values_one, target_q_values_two = self.critic_target(next_states, next_actions)

            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)
            q_target        = rewards + self.gamma * (1 - dones) * target_q_values

        q_vals_q1, q_vals_q2 = self.critic(states, actions)

        critic_loss_1     = F.mse_loss(q_vals_q1, q_target)
        critic_loss_2     = F.mse_loss(q_vals_q2, q_target)
        critic_loss_total = critic_loss_1 + critic_loss_2

        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        #torch.nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()


        # Delayed policy updates
        if self.update_counter % self.policy_freq_update == 0:
            # ------- calculate the actor loss
            pre_activation, actor_action  = self.actor(states)
            actor_q1, actor_q2 = self.critic(states, actor_action)

            actor_q_min = torch.minimum(actor_q1, actor_q2)
            actor_loss  = - actor_q_min.mean()

            # idea: Saturation Penalty
            #-------------------------------------------------------------------------------------
            upper_saturation =  1.5 # 3
            lower_saturation = -1.5 #-3
            k_saturation     =  100 # this number is a empirical value could be changed

            saturation_penalty_up = pre_activation - upper_saturation
            saturation_penalty_up = torch.maximum(saturation_penalty_up, torch.tensor(0))

            saturation_penalty_down = -pre_activation + lower_saturation
            saturation_penalty_down = torch.maximum(saturation_penalty_down, torch.tensor(0))

            saturation_penalty = saturation_penalty_up + saturation_penalty_down
            saturation_penalty = torch.square(saturation_penalty).mean()
            saturation_penalty = k_saturation * saturation_penalty
            #-------------------------------------------------------------------------------------

            total_actor_loss = actor_loss + saturation_penalty
            print(f"Actor Loss={actor_loss} Penalty={saturation_penalty}")

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=0.1)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # ------------------------------------- Update target networks --------------- #
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_models(self, filename):
        torch.save(self.actor.state_dict(),  f'models/{filename}_actor.pht')
        torch.save(self.critic.state_dict(), f'models/{filename}_critic.pht')
        print("models has been saved...")

    def load_models(self, filename):
        # to do this
        pass
