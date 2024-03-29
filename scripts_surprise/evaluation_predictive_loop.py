import gym
import torch
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from Surprise import Deep_Surprise
from cares_reinforcement_learning.util import helpers as hlp


def evaluation(env, prediction_ensemble_model):
    prediction_ensemble_model.load_model()

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    state, _   = env.reset()
    action_env = env.action_space.sample()
    action     = hlp.normalize(action_env, max_action_value, min_action_value)  # algorithm range [-1, 1]

    next_state, reward, done, truncated, info = env.step(action_env)

    # avr_mean, avr_std, avr_std_total = prediction_ensemble_model.get_prediction_from_model(state, action)
    # print("Predicted State", avr_mean)
    # print("Next State True", next_state)
    # print("Error in the prediction", (avr_mean - next_state))
    # print("Uncertainly in each element of prediction", avr_std)
    # print("Average Uncertainly in prediction", avr_std_total)
    # print("Average Error in prediction", np.mean(avr_mean - next_state))


    prediction_avr = prediction_ensemble_model.get_prediction_from_model_discrete(state, action)
    print("Predicted State", prediction_avr)
    print("Next State True", next_state)
    print("Error in the prediction", (prediction_avr - next_state))
    print("Average Error in prediction", np.mean(prediction_avr - next_state))
    mse = (np.square(prediction_avr - next_state)).mean()
    print("MSE Error in prediction", mse)




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_gym_name = "Pendulum-v1"  # BipedalWalker-v3, Pendulum-v1, HalfCheetah-v4"

    env = gym.make(env_gym_name, render_mode=None)  # for AE needs to be rgb_array in render_mode

    obs_size    = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    prediction_model = Deep_Surprise(
        input_dim=obs_size+action_size,
        output_dim=obs_size,
        device=device,
        ensemble_size=5
    )

    evaluation(env, prediction_model)


if __name__ == '__main__':
    main()
