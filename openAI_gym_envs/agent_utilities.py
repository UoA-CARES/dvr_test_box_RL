
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F

from openAI_architectures_utilities  import Actor_Normal, Critic_Normal, Actor, Critic, Decoder


class TD3:
    def __init__(self, obs_dim, act_dim, max_act_value, device, env_name):
        self.obs_dim       = obs_dim
        self.act_dim       = act_dim
        self.max_act_value = max_act_value
        self.device        = device
        self.env_name      = env_name

        self.lr_critic  = 3e-4 #1e-3
        self.lr_actor   = 3e-4 #1e-4

        self.gamma = 0.99
        self.tau   = 0.005

        self.update_counter     = 0
        self.policy_freq_update = 2

        self.actor  = Actor_Normal(self.obs_dim,  self.act_dim, self.max_act_value).to(self.device)
        self.critic = Critic_Normal(self.obs_dim, self.act_dim).to(self.device)

        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.actor.train(True)
        self.critic.train(True)


    def get_action_from_policy(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.to(self.device)
            _, action = self.actor(state_tensor)
            action = action.cpu().data.numpy().flatten()
        return action

    def update_policy(self, memory_buffer, batch_size):
        self.update_counter += 1  # this is used for delay update the actor
        states, actions, rewards, next_states, dones = memory_buffer.sample_experiences_from_buffer(batch_size)


        with torch.no_grad():
            _, next_actions = self.actor_target(next_states)
            target_noise = 0.2 * torch.randn_like(next_actions)
            target_noise = target_noise.clamp(-0.5, 0.5)
            next_actions = next_actions + target_noise
            next_actions = next_actions.clamp(-self.max_act_value, self.max_act_value)


            next_q_values_q1, next_q_values_q2 = self.critic_target(next_states, next_actions)
            q_min = torch.min(next_q_values_q1, next_q_values_q2)
            q_target = rewards + (self.gamma * (1 - dones) * q_min)

        q_vals_q1, q_vals_q2 = self.critic(states, actions)

        critic_loss_1 = F.mse_loss(q_vals_q1, q_target)
        critic_loss_2 = F.mse_loss(q_vals_q2, q_target)
        critic_loss_total = critic_loss_1 + critic_loss_2


        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=1.0)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.update_counter % self.policy_freq_update == 0:
            # ------- calculate the actor loss
            pre_activation, action_actor = self.actor(states)
            actor_q1, actor_q2 = self.critic(states, action_actor)

            # original paper work here with Q1 only
            # actor_loss = - actor_q1.mean()

            actor_q_min = torch.min(actor_q1, actor_q2)
            actor_loss = - actor_q_min.mean()

            # idea: Saturation Penalty
            upper_saturation  = torch.tensor(2.5)
            lower_saturation  = torch.tensor(-2.5)
            saturation_penalty = torch.max((pre_activation-upper_saturation), torch.tensor(0)) + torch.max((-pre_activation+lower_saturation), torch.tensor(0))
            saturation_penalty = torch.square(saturation_penalty).mean()

            total_actor_loss = actor_loss + (0.3 * saturation_penalty)

            self.actor_optimizer.zero_grad()
            #actor_loss.backward()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=1.0) # still no sure about this 0.1
            self.actor_optimizer.step()
            # ------------------------------------- Update target networks --------------- #
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_models(self):
        torch.save(self.actor.state_dict(), f'trained_models/Normal-TD3_actor_{self.env_name}.pht')
        print("models have been saved...")


class TD3AE:
    def __init__(self, obs_dim, act_dim, max_act_value, device, env_name):

        self.obs_dim       = obs_dim
        self.act_dim       = act_dim
        self.max_act_value = max_act_value
        self.device        = device
        self.env_name      = env_name

        self.encoder_lr = 1e-3
        self.decoder_lr = 1e-3

        self.critic_lr = 1e-3 #3e-4  # 1e-3
        self.actor_lr  = 1e-4 #3e-4  # 1e-4

        self.tau         = 0.005 # 0.005
        self.tau_encoder = 0.001 # 0.001
        self.gamma       = 0.99

        self.update_counter     = 0
        self.policy_freq_update = 2

        self.latent_dim = 50  # 50

        # main networks
        self.actor  = Actor(self.latent_dim, self.act_dim, self.max_act_value).to(self.device)
        self.critic = Critic(self.latent_dim, self.act_dim).to(self.device)

        # target networks
        self.actor_target  = Actor(self.latent_dim, self.act_dim, self.max_act_value).to(self.device)
        self.critic_target = Critic(self.latent_dim, self.act_dim).to(self.device)

        # tie encoders between actor and critic
        # with this, any changes in the critic encoder
        # will also be affecting the actor-encoder during the whole training
        self.actor.encoder_net.copy_conv_weights_from(self.critic.encoder_net)

        # copy weights and bias from main to target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        # main Decoder
        self.decoder = Decoder(self.latent_dim).to(device)

        # Optimizer
        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder_net.parameters(), lr=self.encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(),  lr=self.decoder_lr, weight_decay=1e-7)
        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(),    lr=self.actor_lr,  betas=(0.9, 0.999))
        self.critic_optimizer  = torch.optim.Adam(self.critic.parameters(),   lr=self.critic_lr, betas=(0.9, 0.999))

        self.actor.train(True)
        self.critic.train(True)
        self.critic_target.train(True)
        self.actor_target.train(True)
        self.decoder.train(True)

    def get_action_from_policy(self, state_image_pixel):
        with torch.no_grad():
            state_image_tensor = torch.FloatTensor(state_image_pixel)
            state_image_tensor = state_image_tensor.unsqueeze(0).to(self.device)
            action = self.actor(state_image_tensor)
            action = action.cpu().data.numpy().flatten()
        return action


    def update_policy(self, memory_buffer, batch_size):

        self.update_counter += 1  # this is used for delay update the actor
        states, actions, rewards, next_states, dones = memory_buffer.sample_experiences_from_buffer(batch_size)


        #print(dones)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_noise = 0.2 * torch.randn_like(next_actions)
            target_noise = target_noise.clamp(-0.5, 0.5)
            next_actions = next_actions + target_noise
            next_actions = next_actions.clamp(-self.max_act_value, self.max_act_value)

            next_q_values_q1, next_q_values_q2 = self.critic_target(next_states, next_actions)
            q_min = torch.min(next_q_values_q1, next_q_values_q2)

            q_target = rewards + (self.gamma * (1 - dones) * q_min)

        q_vals_q1, q_vals_q2 = self.critic(states, actions)

        critic_loss_1 = F.mse_loss(q_vals_q1, q_target)
        critic_loss_2 = F.mse_loss(q_vals_q2, q_target)
        critic_loss_total = critic_loss_1 + critic_loss_2

        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.update_counter % self.policy_freq_update == 0:
            # ------- calculate the actor loss
            actor_q1, actor_q2 = self.critic(states, self.actor(states))
            actor_q_min = torch.min(actor_q1, actor_q2)
            actor_loss = - actor_q_min.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ------------------------------------- Update target networks --------------- #
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Update the autoencoder part
        z_vector = self.critic.encoder_net(states)
        rec_obs  = self.decoder(z_vector)

        rec_loss = F.mse_loss(states, rec_obs)
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
