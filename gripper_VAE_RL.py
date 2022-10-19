"""
Description:
            VAE-Using test bed camera
            input image size = 1024 * 960
            input NN size    = 128 * 128
            latent vector    = 32
"""
import cv2

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from gripper_function_utilities import Utilities
from gripper_environment_utilities import RL_ENV

from memory_utilities import MemoryClass
from networks_architectures import VanillaVAE, ForwardModelPrediction

# todo should i use or not the eval() mode when i need a network during the training ?
class ReductionLearning:
    def __init__(self):
        # values for loops


        self.batch_size          = 8
        self.minimal_buffer_size = 256
        self.latent_vector_size  = 32

        self.exploration_episodes = 10_000  # todo do i need this ?

        self.camera_index = 2
        self.max_memory_size = 10_000

        self.device = Utilities().detect_device()
        #self.vision = VisionCamera(self.camera_index)
        self.memory = MemoryClass(self.max_memory_size)
        self.env = RL_ENV()

        # ---- Initialization and build  VAE Model --- #
        self.learning_rate_vae = 0.0001
        self.vae = VanillaVAE(self.latent_vector_size).to(self.device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate_vae)
        self.vae_loss = []

        # ---- Initialization and build  Forward Predictive Model --- #
        self.learning_rate_forward = 0.001
        self.forward_prediction_model = ForwardModelPrediction().to(self.device)
        self.forward_prediction_optimizer = optim.Adam(self.forward_prediction_model.parameters(), lr=self.learning_rate_forward)
        self.forward_prediction_loss = []

        self.novelty_values  = []
        self.surprise_values = []


    def learn_vae_model_function(self):
        # Full reconstruction VAE model
        # input random sampled batch of preprocessed images (batch, 3, 128, 128)
        if len(self.memory.memory_buffer) <= self.minimal_buffer_size:
            return
        else:
            # sample from memory a batch but care about image-only
            img_states, _, _, _, _ = self.memory.sample_full_experiences(self.batch_size)

            img_states = np.array(img_states)
            img_states = torch.FloatTensor(img_states)  # change to tensor
            img_states = img_states.permute(0, 3, 1, 2)  # just put in the right order [b, 3, 128, 128]
            img_states = img_states.to(self.device)  # send batch to GPU

            x_rec, mu, log_var, z = self.vae.forward(img_states)

            # ---------------- Loss Function Reconstruction + KL --------------------------#
            # rec_loss = F.mse_loss(x_rec, img_states, reduction="sum")
            rec_loss = F.binary_cross_entropy(x_rec, img_states, reduction="sum")
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            total_loss = rec_loss + kld_loss
            # ------------------------------------------------------------------------------#
            self.vae.train()
            self.vae_optimizer.zero_grad()
            total_loss.backward()
            self.vae_optimizer.step()

            print("VAE training Loss:", total_loss.item(), rec_loss.item(), kld_loss.item())
            self.vae_loss.append(total_loss.item())

            # --------------------------------------------------------------------------------

    def learn_predictive_model_function(self):
        if len(self.memory.memory_buffer) <= self.minimal_buffer_size:
            return
        else:
            img_states, actions, _, img_next_states, _ = self.memory.sample_full_experiences(self.batch_size)

            img_states = np.array(img_states)
            img_states = torch.FloatTensor(img_states)  # change to tensor
            img_states = img_states.permute(0, 3, 1, 2)  # just put in the right order [b, 3, 128, 128]
            img_states = img_states.to(self.device)  # send batch to GPU

            actions = np.array(actions)
            actions = torch.FloatTensor(actions)
            actions = actions.to(self.device)  # send batch to GPU

            img_next_states = np.array(img_next_states)
            img_next_states = torch.FloatTensor(img_next_states)
            img_next_states = img_next_states.permute(0, 3, 1, 2)
            img_next_states = img_next_states.to(self.device)  # send batch to GPU

        with torch.no_grad():
            self.vae.eval()
            _, _, _, z_input  = self.vae.forward(img_states)
            _, _, _, z_target = self.vae.forward(img_next_states)


        distribution_probability_model = self.forward_prediction_model.forward(z_input, actions)
        loss_neg_log_likelihood = - distribution_probability_model.log_prob(z_target)
        loss_neg_log_likelihood = torch.mean(loss_neg_log_likelihood)

        self.forward_prediction_model.train()
        self.forward_prediction_optimizer.zero_grad()
        loss_neg_log_likelihood.backward()
        self.forward_prediction_optimizer.step()

        print("Forward Prediction Loss:", loss_neg_log_likelihood.item())
        self.forward_prediction_loss.append(loss_neg_log_likelihood.item())


    def policy_learning_function(self):

        #if len(self.memory.memory_buffer) <= self.minimal_buffer_size:
        if len(self.memory.memory_buffer) <= 8:
            return
        else:
            img_states, actions, rewards, img_next_states, dones = self.memory.sample_full_experiences(self.batch_size)

            img_states = np.array(img_states)
            img_states = torch.FloatTensor(img_states)  # change to tensor
            img_states = img_states.permute(0, 3, 1, 2)  # just put in the right order [b, 3, 128, 128]
            img_states = img_states.to(self.device)  # send batch to GPU

            actions = np.array(actions)
            actions = torch.FloatTensor(actions)
            actions = actions.to(self.device)  # send batch to GPU

            img_next_states = np.array(img_next_states)
            img_next_states = torch.FloatTensor(img_next_states)
            img_next_states = img_next_states.permute(0, 3, 1, 2)
            img_next_states = img_next_states.to(self.device)  # send batch to GPU

            rewards = np.array(rewards).reshape(-1, 1)
            rewards = torch.FloatTensor(rewards)
            rewards = rewards.to(self.device)

            with torch.no_grad():
                self.vae.eval()
                _, _, _, z_state      = self.vae.forward(img_states)
                _, _, _, z_next_state = self.vae.forward(img_next_states)

            # todo need to create the state_space observation here
            # observation space (encode_image vector, valve_angle, target_angle, novelty, surprise)
            # valve and target angle could come with the sampled batch



    def calculate_novelty(self):
        pass

    def calculate_surprise(self):
        pass

    def learn_all_online(self):
        # todo still no sure if this is correct
        if len(self.memory.memory_buffer) <= self.batch_size:
            return
        else:
            img_states, actions, _, img_next_states, _ = self.memory.sample_full_experiences(self.batch_size)

            img_states = np.array(img_states)
            img_states = torch.FloatTensor(img_states)  # change to tensor
            img_states = img_states.permute(0, 3, 1, 2)  # just put in the right order [b, 3, 128, 128]
            img_states = img_states.to(self.device)  # send batch to GPU

            actions = np.array(actions)
            actions = torch.FloatTensor(actions)
            actions = actions.to(self.device)  # send batch to GPU

            img_next_states = np.array(img_next_states)
            img_next_states = torch.FloatTensor(img_next_states)
            img_next_states = img_next_states.permute(0, 3, 1, 2)
            img_next_states = img_next_states.to(self.device)  # send batch to GPU

            #  =========== train vae model only  ===========
            for param in self.vae.parameters():
                param.requires_grad = True
            for param in self.forward_prediction_model.parameters():
                param.requires_grad = False

            x_rec, mu, log_var, z_input = self.vae.forward(img_states)
            _, _, _, z_target           = self.vae.forward(img_next_states)

            # ---------------- Loss Function Reconstruction + KL --------------------------#
            # rec_loss = F.mse_loss(x_rec, img_states, reduction="sum")
            rec_loss = F.binary_cross_entropy(x_rec, img_states, reduction="sum")
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            total_loss_vae = rec_loss + kld_loss
            # ------------------------------------------------------------------------------#
            self.vae.train()
            self.vae_optimizer.zero_grad()
            total_loss_vae.backward()
            self.vae_optimizer.step()

            #  =========== train forward prediction model only  ===========
            for param in self.vae.parameters():
                param.requires_grad = False
            for param in self.forward_prediction_model.parameters():
                param.requires_grad = True

            z_in  = z_input.detach()
            z_out = z_target.detach()

            distribution_probability_model = self.forward_prediction_model.forward(z_in, actions)
            loss_neg_log_likelihood = - distribution_probability_model.log_prob(z_out)
            loss_neg_log_likelihood = torch.mean(loss_neg_log_likelihood)

            self.forward_prediction_model.train()
            self.forward_prediction_optimizer.zero_grad()
            loss_neg_log_likelihood.backward()
            self.forward_prediction_optimizer.step()


        # I can add the actor critic model here
        #  =========== train actor critic model only  ===========



    def update_models(self):
        #self.learn_vae_model_function()
        #self.learn_predictive_model_function()
        self.policy_learning_function()
        #self.learn_all_online()


    def run_exploration_frames(self):
        state_image = self.vision.get_camera_image()
        state_image = self.vision.pre_pro_image(state_image)
        for _ in tqdm(range(1, self.exploration_episodes+1)):
            action = self.env.generate_sample_act()
            self.env.env_step(action)
            new_state_image = self.vision.get_camera_image()
            new_state_image = self.vision.pre_pro_image(new_state_image)
            self.memory.save_frame_experience_buffer(state_image)
            state_image = new_state_image


    def rl_idea_training(self, horizontal_steps=50, num_episodes=100):

        state_raw_image = self.env.vision_config.get_camera_image()
        state_image     = self.env.vision_config.pre_pro_image(state_raw_image)

        target_angle    = self.env.define_goal_angle()
        valve_angle     = self.env.get_valve_angle()

        # todo set the reset function here

        for episode in range(1, horizontal_steps+1):

            action = self.env.generate_sample_act()  # todo update this with an action from policy

            self.env.env_step(action)

            new_state_raw_image = self.env.vision_config.get_camera_image()
            new_state_image     = self.env.vision_config.pre_pro_image(new_state_raw_image)
            new_valve_angle     = self.env.get_valve_angle()

            reward, done = self.env.calculate_extrinsic_reward(target_angle, new_valve_angle)

            #self.memory.save_full_experience_buffer(state_image, action, reward, new_state_image, done, valve_angle, new_valve_angle, target_angle)
            # todo could also create other buffer with (z,a,r,zt,done, target, valve_angle) where z is added valve and target valve

            state_image = new_state_image
            valve_angle = new_valve_angle


            if done:
                print("done TRUE, breaking loop")
                break

            #self.update_models()

            '''
            while True:
                img_aruco, valve_angle, flag_detection = self.vision.get_aruco_angle()
                if flag_detection:
                    cv2.imshow("Render Image", img_aruco)
                    cv2.waitKey(10)
                    break
                else:
                    pass
            '''


    def vae_evaluation(self):
        Utilities().load_vae_model(self.vae)

        state_image = self.vision.get_camera_image()
        state_image = self.vision.pre_pro_image(state_image)

        self.vae.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_image)
            state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.permute(0, 3, 1, 2).to(self.device)
            x_rec, _, _, _ = self.vae.forward(state_tensor)
            x_rec = x_rec.permute(0, 2, 3, 1)
            x_rec = x_rec.cpu().numpy()

        plt.subplot(1, 2, 1)  # row 1, col 2 index 1
        plt.title("Input")
        #plt.imshow(state_image)
        plt.imshow((state_image * 255).astype(np.uint8))
        plt.subplot(1, 2, 2)  # index 2
        plt.title("Reconstruction")
        #plt.imshow(x_rec[0])
        plt.imshow((x_rec[0] * 255).astype(np.uint8))
        plt.show()


def main_run():
    model = ReductionLearning()
    model.rl_idea_training()


if __name__ == '__main__':
    main_run()
