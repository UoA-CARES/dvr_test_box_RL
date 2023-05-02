import gym
import torch

import numpy as np
import cv2

import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt

from Novelty import Deep_Novelty
from skimage.metrics import structural_similarity as ssim


def preprocessing_image(image_array):
    resized    = cv2.resize(image_array, (84, 84), interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image_out  = np.expand_dims(norm_image, axis=0) # to make it (1, channel, w, h) because i am using a single img here
    return image_out


def evaluation(env, autoencoder_model):
    autoencoder_model.load_model()

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    _, _ = env.reset()
    state = env.render()
    state = preprocessing_image(state)

    reconstruction = autoencoder_model.get_reconstruction_from_model(state)
    reconstruction = reconstruction.cpu().numpy()

    input_img = state[0]
    reconstruction_img = reconstruction[0][0]

    #difference = ((input_img - reconstruction_img) * 255).astype("uint8")
    difference = (input_img - reconstruction_img)

    ssim_index = ssim(input_img, reconstruction_img, data_range=input_img.max() - input_img.min())
    print("Structural similarity index", ssim_index)

    plt.subplot(1, 3, 1)
    plt.title("Image Input")
    plt.imshow(input_img, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Image Reconstruction")
    plt.imshow(reconstruction_img, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.imshow(difference, cmap='gray')

    plt.show()





def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_gym_name = "Pendulum-v1"  # BipedalWalker-v3, Pendulum-v1, HalfCheetah-v4"

    env = gym.make(env_gym_name, render_mode="rgb_array")  # for AE needs to be rgb_array in render_mode

    latent_dim = 50

    autoencoder_model = Deep_Novelty(
        latent_dim=latent_dim,
        device=device,
        k=1
    )

    evaluation(env, autoencoder_model)


if __name__ == '__main__':
    main()
