
import gym
import torch
import logging
logging.basicConfig(level=logging.INFO)

from cares_reinforcement_learning.util import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp

from networks import Actor
from networks import Critic
from networks import Decoder

from AE_TD3 import AE_TD3
from FrameStack import FrameStack



def train(env, agent):
    max_steps_training    = 50_000
    max_steps_exploration = 1_000
    batch_size            = 32

    seed = 232
    G    = 10
    k    = 3

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    memory       = MemoryBuffer()
    frames_stack = FrameStack(env, k)

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state = frames_stack.reset()

    historical_reward = {"step": [], "episode_reward": []}

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action_env = env.action_space.sample()  # action range the env uses [e.g. -2 , 2 for pendulum]
            action     = hlp.normalize(action_env, max_action_value, min_action_value)  # algorithm range [-1, 1]

        else:
            action     = agent.get_action_from_policy(state)
            action_env = hlp.denormalize(action, max_action_value, min_action_value)

        next_state, reward, done, truncated, info = frames_stack.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        state = next_state

        episode_reward += reward

        if total_step_counter >= max_steps_exploration:
            for _ in range(G):
                experiences = memory.sample(batch_size)
                agent.train_policy(experiences)

        if done or truncated:
            logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state = frames_stack.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make('Pendulum-v1', render_mode="rgb_array")  # for AE needs to be rgb_array in render_mode

    action_size = env.action_space.shape[0]
    latent_size = 50

    lr_actor   = 1e-3
    lr_critic  = 1e-4
    lr_decoder = 1e-3
    lr_encoder = 1e-3

    gamma = 0.99
    tau   = 0.005

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
        lr_actor=1e-3,
        lr_critic=1e-4,
        lr_decoder=1e-3,
        lr_encoder=1e-3,
        device=device,

    )

    train(env, agent)


if __name__ == '__main__':
    main()
