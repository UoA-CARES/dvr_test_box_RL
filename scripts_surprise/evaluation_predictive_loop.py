import gym
import torch
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from Ensemble import Deep_Ensemble
from cares_reinforcement_learning.util import helpers as hlp


def evaluation(env, prediction_ensemble_model):
    prediction_ensemble_model.load_model()
    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    state, _   = env.reset()
    action_env = env.action_space.sample()
    action     = hlp.normalize(action_env, max_action_value, min_action_value)  # algorithm range [-1, 1]
    next_state, reward, done, truncated, info = env.step(action_env)

    next_state_predicted = prediction_ensemble_model.get_prediction_from_model(state, action)

    print(state)
    print(next_state)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_gym_name = "Pendulum-v1"  # BipedalWalker-v3, Pendulum-v1, HalfCheetah-v4"

    env = gym.make(env_gym_name, render_mode=None)  # for AE needs to be rgb_array in render_mode

    obs_size    = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    prediction_model = Deep_Ensemble(
        input_dim=obs_size+action_size,
        output_dim=obs_size,
        device=device,
        ensemble_size=5
    )

    evaluation(env, prediction_model)


if __name__ == '__main__':
    main()
