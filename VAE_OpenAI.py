
import gym
import cv2
import pickle
import random
import json
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
from networks_architectures import Encoder, Decoder



if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Working with GPU")
else:
    device = torch.device('cpu')
    print("Working with CPU")


def save_img_memory(image, episode, step):
    cv2.imwrite(f"image_experiences/image{episode}_{step}.png", image)


def save_experience_buffer(state_image, action, reward, next_state_image, done, buffer):
    experience = (state_image, action, reward, next_state_image, done)
    buffer.append(experience)


def save_experience_memory_pickle(buffer_full):
    with open("data_experiences/data_experiences.npy", "wb") as f:
        pickle.dump(buffer_full, f)  # generated file too big
        #np.save(f, buffer_full)
        # todo try with nzp_compress


def sample_experiences(buffer, sample_size):

    state_batch  = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    done_batch       = []

    batch = random.sample(buffer, sample_size)

    for experience in batch:
        state, action, reward, next_state, done = experience
        state_batch.append(state)
        action_batch.append(action)
        reward_batch.append(reward)
        next_state_batch.append(next_state)
        done_batch.append(done)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch


def learn_encoder_function(buffer, encoder, decoder):

    sample_size = 5

    if len(buffer) <= sample_size:
        return
    else:

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = sample_experiences(buffer, sample_size)

        state = np.array(state_batch)
        state = torch.FloatTensor(state)
        state = state.permute(0, 3, 1, 2)  # batch, channel, H, W
        state = state.to(device)

        z, mu, log_var = encoder.forward(state)

        x_rec = decoder.forward(z)

        print(x_rec.shape)





def main_run():
    num_episodes = 50
    episode_horizont = 200
    memory_buffer = deque(maxlen=10_000)

    env = gym.make('Pendulum-v1')
    env.reset()

    encoder = Encoder()
    decoder = Decoder()

    encoder.to(device)
    decoder.to(device)

    for episode in range(1, num_episodes + 1):
        env.reset()
        state_image = env.render(mode='rgb_array')  # return the rendered image and can be used as input-state image

        for step in range(1, episode_horizont + 1):

            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            new_state_image = env.render(mode='rgb_array')

            # save image and experience here
            save_experience_buffer(state_image, action, reward, new_state_image, done, memory_buffer)
            save_img_memory(state_image, episode, step)

            state_image = new_state_image

            learn_encoder_function(memory_buffer, encoder, decoder)

            if done:
                break

    env.close()


if __name__ == '__main__':
    main_run()
