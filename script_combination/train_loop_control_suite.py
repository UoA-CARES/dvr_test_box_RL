

import cv2
import torch
import numpy as np
from dm_control import suite
import dmc2gym


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    domain_name = "cartpole"
    task_name   = "balance"
    seed        = 571
    env         = suite.load(domain_name, task_name, task_kwargs={'random': seed})


    time_step   = env.reset()
    action_spec = env.action_spec()

    print()

    # observation = np.hstack(list(time_step.observation.values())) # # e.g. position, orientation, joint_angles
    #
    # observation_size = len(observation)
    # action_num       = action_spec.shape[0]
    #
    # max_action_value = action_spec.maximum[0]
    # min_action_value = action_spec.minimum[0]
    #







if __name__ == '__main__':
    main()