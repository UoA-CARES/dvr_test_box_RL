
import cv2
import gym
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

import torch
import torch.optim as optim
import torch.nn.functional as F

from memory_utilities import MemoryClass
from networks_architectures import VanillaVAE

import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Working with GPU")
else:
    device = torch.device('cpu')
    print("Working with CPU")


class VAE_RL_AGENT:
    def __init__(self):

        self.batch_size         = 8
        self.learning_rate      = 0.0001
        self.latent_vector_size = 32
        self.max_memory_size    = 20_000

        self.memory = MemoryClass(self.max_memory_size)
        self.vae    = VanillaVAE(self.latent_vector_size).to(device)

        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        #self.vae_optimizer = optim.SGD(self.vae.parameters(), lr=self.learning_rate)
        self.vae_loss = []

        self.N = 1  # internal loop for training VAE


    def learn_encoder_function(self):
        if len(self.memory.memory_buffer_frames) <= self.batch_size:
            return
        else:
            for _ in range(1, self.N+1):
                # sample from memory a batch of previous image-only experiences
                img_states  = self.memory.sample_frames_experiences(self.batch_size)
                img_states  = np.array(img_states)
                img_states  = torch.FloatTensor(img_states)   # change to tensor
                img_states  = img_states.permute(0, 3, 1, 2)  # just put in the right order [b, 3, 128, 128]
                img_states  = img_states.to(device)           # send batch to GPU

                x_rec, mu, log_var, z = self.vae.forward(img_states)

                # ---------------- Loss Function Reconstruction + KL --------------------------#
                #rec_loss = F.mse_loss(x_rec, img_states, reduction="sum")
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


    def calculate_z_and_similarity(self, img_input):
        self.vae.eval()
        with torch.no_grad():
            img_tensor = torch.FloatTensor(img_input)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.permute(0, 3, 1, 2).to(device)

            x_rec, _, _, z = self.vae.forward(img_tensor)
            x_rec = x_rec.permute(0, 2, 3, 1)
            x_rec = x_rec[0].cpu().numpy()

        ssim_const = ssim(img_input, x_rec, multichannel=True, data_range=img_input.max() - x_rec.min())
        z = z.cpu().numpy()
        return ssim_const, z[0]


    def save_vae_model(self):
        torch.save(self.vae.state_dict(), "trained_models/vae_model.pth")
        print("VAE model has been saved")

    def load_vae_model(self):
        self.vae.load_state_dict(torch.load("trained_models/vae_model.pth"))
        print("VAE model has been loaded")



def pre_pro_image(image_array):
    img        = cv2.resize(image_array, (128, 128), interpolation=cv2.INTER_AREA)
    norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #cv2.imshow("Normalized image", norm_image)
    #cv2.waitKey(10)
    return norm_image


def run_random_exploration(env, agent,  num_exploration_episodes, episode_horizont):
    for episode in tqdm(range(1, num_exploration_episodes + 1)):
        #print(episode, "Exploration")
        env.reset()
        state_image = env.render(mode='rgb_array')  # return the rendered image and can be used as input-state image
        state_image = pre_pro_image(state_image)
        for step in range(1, episode_horizont + 1):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            new_state_image = env.render(mode='rgb_array')
            new_state_image = pre_pro_image(new_state_image)
            agent.memory.save_frame_experience_buffer(state_image)
            state_image = new_state_image
            if done:
                break


def run_training_rl_method(env, agent, num_episodes_training, episode_horizont):

    for episode in range(1, num_episodes_training + 1):
        print(f"-----------------Episode {episode}-----------------------------")
        env.reset()
        state_image = env.render(mode='rgb_array')  # return the rendered image so can be used as input-state image
        state_image = pre_pro_image(state_image)

        for step in range(1, episode_horizont + 1):

            action = env.action_space.sample()  # todo change action from policy
            obs, reward, done, _ = env.step(action)
            new_state_image = env.render(mode='rgb_array')
            new_state_image = pre_pro_image(new_state_image)

            agent.memory.save_frame_experience_buffer(state_image)

            # ----- Learning Function VAE------#
            agent.learn_encoder_function()

            # -----------#
            similarity, z_state = agent.calculate_z_and_similarity(state_image)
            _, z_next_state     = agent.calculate_z_and_similarity(new_state_image)
            if similarity >= 0.98:   # similarity_tolerance:
                print("similarity:", similarity)
                agent.memory.save_vector_experience_buffer(z_state, action, reward, z_next_state, done)

            state_image = new_state_image

            if done:

                break

    agent.save_vae_model()


def run_training_vae_only(agent):
    for epochs in range(1, 20_000):
        agent.learn_encoder_function()
    agent.save_vae_model()


def evaluate_vae_model(env, agent):
    env.reset()
    state_image = env.render(mode='rgb_array')
    state_image = pre_pro_image(state_image)

    agent.load_vae_model()
    agent.vae.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_image)
        state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.permute(0, 3, 1, 2).to(device)
        x_rec, _, _, _ = agent.vae.forward(state_tensor)
        x_rec = x_rec.permute(0, 2, 3, 1)
        #x_rec = x_rec.view(-1, 128, 128, 3)
        x_rec = x_rec.cpu().numpy()

    plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    plt.title("Input")
    plt.imshow(state_image)
    plt.subplot(1, 2, 2)  # index 2
    plt.title("Reconstruction")
    plt.imshow(x_rec[0])
    plt.show()



def main_run():
    env = gym.make('Pendulum-v1')
    agent = VAE_RL_AGENT()

    num_exploration_episodes  = 100
    num_episodes_training     = 1000
    episode_horizont          = 200

    run_random_exploration(env, agent, num_exploration_episodes, episode_horizont)
    run_training_rl_method(env, agent, num_episodes_training, episode_horizont)
    #run_training_vae_only(agent)
    evaluate_vae_model(env, agent)
    env.close()


if __name__ == '__main__':
    main_run()