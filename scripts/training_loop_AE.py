
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from cares_reinforcement_learning.util import MemoryBuffer

from networks import Actor
from networks import Critic
from networks import Decoder

from AE_TD3 import AE_TD3



def train():
    pass

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make('Pendulum-v1')

    #obs_size    = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    latent_size = 50

    lr_actor   = 1e-3
    lr_critic  = 1e-4
    lr_decoder = 1e-3
    lr_encoder = 1e-3

    gamma = 0.99
    tau   = 0.005  # 0.005


    memory      = MemoryBuffer()
    actor_net   = Actor(latent_size, action_size, lr_actor)
    critic_net  = Critic(latent_size, action_size, lr_critic)
    decoder_net = Decoder(latent_size, lr_decoder)

    agent = AE_TD3(
        actor_network=actor_net,
        critic_network=critic_net,
        decoder_network=decoder_net,
        gamma=gamma,
        tau=tau,
        action_num=action_size,
        latent_size=latent_size,
        device=device,
    )

    train()


if __name__ == '__main__':
    main()