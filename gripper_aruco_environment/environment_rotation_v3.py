"""
Author: DVR
Date: 19 /12 / 2022
Modification:

Description: Environment for robot gripper, using vector state representation with 7 aruco markers
             Task  = Rotation valve only
             Input = state space vector representation
"""

import cv2
import math
import time
import random
import numpy as np

from motor_utilities_v3  import Motor
from vision_utilities_v3 import Vision


class RL_ENV:

    def __init__(self, camera_index=0, device_index=1):

        self.camera_index = camera_index
        self.device_index = device_index

        self.motors_config = Motor(self.device_index)
        self.vision_config = Vision(self.camera_index)

        self.goal_angle      = 0.0
        self.cylinder_angle  = 0.0
        self.counter_success = 0


    def generate_sample_act(self):
        act_m1 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m2 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m3 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m4 = np.clip(random.uniform(-1, 1), -1, 1)
        action_vector = np.array([act_m1, act_m2, act_m3, act_m4])
        return action_vector

    def reset_env(self):
        id_1_dxl_home_position = 310
        id_2_dxl_home_position = 310
        id_3_dxl_home_position = 690
        id_4_dxl_home_position = 690

        
        self.motors_config.move_motor_step(id_1_dxl_home_position, id_2_dxl_home_position,
                                           id_3_dxl_home_position, id_4_dxl_home_position)

        print("Sending Robot to Home Position")
        time.sleep(1.0)

        self.define_goal_angle()

    def define_goal_angle(self):
        #self.goal_angle = random.randint(-180, 180)
        self.goal_angle  = random.randint(0, 360)
        print("New Goal Angle Generated", self.goal_angle)
        #return self.goal_angle


    def state_space_function(self):
        while True:
            state_space_vector, raw_img, detection_status = self.vision_config.calculate_marker_pose(self.goal_angle)
            if detection_status:
                state_space_vector  = [element for state_space_list in state_space_vector for element in state_space_list]
                self.cylinder_angle = state_space_vector[-2:-1]
                break
            else:
                print("waiting for camera and marks")
        return np.array(state_space_vector), raw_img


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

    
    def calculate_reward(self):
        cylinder_angle_single    = self.cylinder_angle[0]
        difference_cylinder_goal = np.abs(cylinder_angle_single - self.goal_angle)

        if difference_cylinder_goal <= 5:
            done     = True
            reward_d = np.float64(100)
            #reward_d = -difference_cylinder_goal
        else:
            done = False
            #reward_d = -difference_cylinder_goal
            reward_d = np.float64(-1)

        return reward_d, done, difference_cylinder_goal


    def env_render(self, image=None, done=False, step=1, episode=1, cylinder=0, mode="exploration"):
        if done:
            self.counter_success += 1
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        target_angle = self.goal_angle
        cv2.circle(image, (560, 405), 97, color, 2)

        cv2.putText(image, f'Goal     Angle : {target_angle}', (580, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Cylinder Angle : {int(cylinder)}', (580, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Success  Counter : {int(self.counter_success)}', (580, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Episode : {str(episode)}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Steps : {str(step)}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Stage : {mode}', (900, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # plot target line
        x_clock = int(560 + 97 * math.cos(np.deg2rad(target_angle-90)))
        y_clock = int(405 + 97 * math.sin(np.deg2rad(target_angle-90)))
        interna_line_x = int(20 * math.cos(np.deg2rad(target_angle-90)))
        interna_line_y = int(20 * math.sin(np.deg2rad(target_angle-90)))
        cv2.line(image, (x_clock-interna_line_x, y_clock-interna_line_y), (x_clock+interna_line_x, y_clock+interna_line_y), (255, 0, 0), 2)

        x_clock_cylinder = int(560 + 97 * math.cos(np.deg2rad(cylinder-90)))
        y_clock_cylinder = int(405 + 97 * math.sin(np.deg2rad(cylinder-90)))
        interna_line_x_cylinder = int(20 * math.cos(np.deg2rad(cylinder-90)))
        interna_line_y_cylinder = int(20 * math.sin(np.deg2rad(cylinder-90)))

        cv2.line(image, (560, 405), (x_clock_cylinder+interna_line_x_cylinder, y_clock_cylinder+interna_line_y_cylinder), color, 2)

        cv2.imshow("State Image Rotation", image)
        cv2.waitKey(10)
