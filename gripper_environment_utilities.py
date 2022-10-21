

import cv2
from gripper_motor_utilities import Motor
from gripper_vision_utilities import VisionCamera

import numpy as np
import random


class RL_ENV:

    def __init__(self):

        self.camera_index = 0

        self.motors_config = Motor()
        self.vision_config = VisionCamera(self.camera_index)

        self.angle_valve_deg  = 0.0
        self.goal_angle_deg   = 0.0

    def generate_sample_act(self):
        act_m1 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m2 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m3 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m4 = np.clip(random.uniform(-1, 1), -1, 1)
        action_vector = np.array([act_m1, act_m2, act_m3, act_m4])
        return action_vector

    def env_reset(self):
        pass

    def env_step(self, actions):
        id_1_dxl_goal_position = (actions[0] - (-1)) * (700 - 300) / (1 - (-1)) + 300
        id_2_dxl_goal_position = (actions[1] - (-1)) * (700 - 300) / (1 - (-1)) + 300
        id_3_dxl_goal_position = (actions[2] - (-1)) * (700 - 300) / (1 - (-1)) + 300
        id_4_dxl_goal_position = (actions[3] - (-1)) * (700 - 300) / (1 - (-1)) + 300

        id_1_dxl_goal_position = int(id_1_dxl_goal_position)
        id_2_dxl_goal_position = int(id_2_dxl_goal_position)
        id_3_dxl_goal_position = int(id_3_dxl_goal_position)
        id_4_dxl_goal_position = int(id_4_dxl_goal_position)

        self.motors_config.move_motor_step(id_1_dxl_goal_position,
                                           id_2_dxl_goal_position,
                                           id_3_dxl_goal_position,
                                           id_4_dxl_goal_position)

    def state_space_function(self):
        pass

    def get_sample_reduction(self):
        pass

    def get_valve_angle(self):
        while True:
            valve_angle, vision_flag_status = self.vision_config.get_aruco_angle()
            if vision_flag_status:
                break
            else:
               pass
        return valve_angle[0]


    def define_goal_angle(self):
        random.seed(10)
        # self.goal_angle = random.randint(-180, 180)
        self.goal_angle_deg = random.randint(0, 360)
        print("New Goal Angle Generated", self.goal_angle_deg)
        return self.goal_angle_deg

    def calculate_extrinsic_reward(self, target_angle, valve_angle):
        angle_difference = np.abs(target_angle - valve_angle)

        if angle_difference <= 10:
            done = True
            reward_ext = np.float64(100)
        else:
            done = False
            reward_ext = -angle_difference

        return reward_ext, done



    def render(self):
        pass

