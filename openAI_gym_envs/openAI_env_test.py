
"""
Description:
            Just a simple script to test if OpenAI envs are installed properly
"""


import gym
import ale_py

print('gym:', gym.__version__)  # gym: 0.21.0
print('ale_py:', ale_py.__version__)  # ale_py: 0.7.1

#env = gym.make('CartPole-v1')      # Discrete action space
#env = gym.make('CarRacing-v0')     # Continuous action space, need -->pip3 install Box2D gym
#env = gym.make('Pendulum-v1')      # Continuous action space
#env = gym.make("LunarLander-v2")   # Discrete action space
env = gym.make("Breakout-v0")      # Discrete action space
#env = gym.make("BipedalWalker-v3")  # Continuous action space


print("---- Environment Basic Info:")
print("Observation Space Shape:", env.observation_space.shape)
print("Action Space Shape:",      env.action_space.shape)
#print("Action Range:", "Min:",    env.action_space.low.min(), ", Max:", env.action_space.high.max())



for e in range(20):
    env.reset()
    for t in range(10000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info, = env.step(action)
        if done:
            print(f"Episode {e + 1} finished after {t + 1} timesteps")
            break


