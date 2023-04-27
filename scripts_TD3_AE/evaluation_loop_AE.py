
import gym
import torch

import numpy as np

from AE_TD3 import AE_TD3
from FrameStack import FrameStack

from networks import Actor
from networks import Critic
from networks import Decoder

from cares_reinforcement_learning.util import helpers as hlp


def evaluation_autoencoder():
    pass
    # todo



def evaluation_policy(env, agent, env_name):

    seed = 5059 # to load the model
    file_name = env_name+"_"+str(seed)
    agent.load_models(filename=file_name)

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    evaluation_seed =  898
    torch.manual_seed(evaluation_seed)
    np.random.seed(evaluation_seed)
    env.action_space.seed(evaluation_seed)

    k = 3
    max_steps_evaluation = 1000
    frames_stack = FrameStack(env, k, evaluation_seed)

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    state = frames_stack.reset()

    for total_step_counter in range(max_steps_evaluation):
        episode_timesteps += 1

        action = agent.get_action_from_policy(state, evaluation=True)  # algorithm range [-1, 1]
        action_env = hlp.denormalize(action, max_action_value, min_action_value)

        next_state, reward, done, truncated, info = frames_stack.step(action_env)
        state = next_state
        episode_reward += reward

        if done  or truncated:
            print(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")
            state = frames_stack.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env_gym_name = "Pendulum-v1"  # BipedalWalker-v3, Pendulum-v1, HalfCheetah-v4"
    env = gym.make(env_gym_name, render_mode="rgb_array")  # for AE needs to be rgb_array in render_mode

    action_size = env.action_space.shape[0]
    latent_size = 50  # 50

    gamma = 0.99
    tau = 0.005

    actor_net   = Actor(latent_size, action_size)
    critic_net  = Critic(latent_size, action_size)
    decoder_net = Decoder(latent_size)

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

    evaluation_policy(env, agent, env_gym_name)




if __name__ == '__main__':
    main()
