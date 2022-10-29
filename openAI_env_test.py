
"""
Description:
            Just a simple script to test if OpenAI envs are installed properly
s
"""


import gym
import ale_py

print('gym:', gym.__version__)  # gym: 0.21.0
print('ale_py:', ale_py.__version__)  # ale_py: 0.7.1

env = gym.make('CartPole-v1')
#env = gym.make('CarRacing-v0')  # pip3 install Box2D gym
#env = gym.make('Pendulum-v1')
#env = gym.make("LunarLander-v2")
#env = gym.make("Breakout-v0")
#env = gym.make("BipedalWalker-v3")


for e in range(20):
    obser = env.reset()
    for t in range(10000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info, = env.step(action)
        print(observation)
        if done:
            print(f"Episode {e + 1} finished after {t + 1} timesteps")
            break


