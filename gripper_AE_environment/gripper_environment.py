"""
Author: DVR
Date:
Modification: 17/11/2022

Description: Environment for robot gripper, autoencoder
             Task = Rotation valve only
             Input = Images only
"""
import cv2
import math
import time
import random
import numpy as np

from gripper_motor_utilities import Motor
from gripper_vision_utilities import VisionCamera


class ENV:

    def __init__(self, camera_index=0, device_index=0):

        self.camera_index = camera_index
        self.device_index = device_index

        self.motors_config = Motor(self.device_index)
        self.vision_config = VisionCamera(self.camera_index)

        self.angle_valve_deg  = 0.0
        self.goal_angle_deg   = 0.0
        self.counter_success  = 0


    def generate_sample_action(self):
        act_m1 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m2 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m3 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m4 = np.clip(random.uniform(-1, 1), -1, 1)
        action_vector = np.array([act_m1, act_m2, act_m3, act_m4])
        return action_vector

    def reset(self):
        id_1_dxl_home_position = 310
        id_2_dxl_home_position = 310
        id_3_dxl_home_position = 690
        id_4_dxl_home_position = 690
        self.motors_config.move_motor_step(id_1_dxl_home_position, id_2_dxl_home_position,
                                           id_3_dxl_home_position, id_4_dxl_home_position)

        print("Sending Robot to Home Position")

        time.sleep(1.0)  # just to make sure robot is moving to home position

    def step_action(self, actions):
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

    def get_valve_angle(self):
        while True:
            valve_angle, vision_flag_status = self.vision_config.get_aruco_angle()
            if vision_flag_status:
                break
            else:
               pass
        return valve_angle[0]


    def define_goal_angle(self):
        self.goal_angle_deg = random.randint(0, 360)
        #self.goal_angle_deg = random.randint(-180, 180)
        print("New Goal Angle Generated", self.goal_angle_deg)
        return self.goal_angle_deg


    def calculate_extrinsic_reward(self, target_angle, valve_angle):
        angle_difference = np.abs(target_angle - valve_angle)

        if angle_difference <= 5:
            done = True
            reward_ext = np.float64(100)
            #reward_ext = -angle_difference
        else:
            done = False
            reward_ext = np.float64(-1)
            #reward_ext = -angle_difference

        return reward_ext, done, angle_difference


    def render(self, image, step, episode, valve_angle, target_angle, done):
        if done:
            self.counter_success += 1
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.circle(image, (560, 405), 97, color, 2)

        cv2.putText(image, f'Goal  Angle : {target_angle}', (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Valve Angle : {int(valve_angle)}', (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Episode : {str(episode)}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Steps : {str(step)}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Success  Counter : {int(self.counter_success)}', (850, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # plot target line
        x_clock = int(560 + 97 * math.cos(np.deg2rad(target_angle-90)))
        y_clock = int(405 + 97 * math.sin(np.deg2rad(target_angle-90)))
        internal_line_x = int(20 * math.cos(np.deg2rad(target_angle-90)))
        internal_line_y = int(20 * math.sin(np.deg2rad(target_angle-90)))
        cv2.line(image, (x_clock-internal_line_x, y_clock-internal_line_y), (x_clock+internal_line_x, y_clock+internal_line_y), (255, 0, 0), 2)

        # plot current valve location
        x_clock_cylinder = int(560 + 97 * math.cos(np.deg2rad(valve_angle-90)))
        y_clock_cylinder = int(405 + 97 * math.sin(np.deg2rad(valve_angle-90)))
        internal_line_x_cylinder = int(20 * math.cos(np.deg2rad(valve_angle-90)))
        internal_line_y_cylinder = int(20 * math.sin(np.deg2rad(valve_angle-90)))
        cv2.line(image, (560, 405), (x_clock_cylinder+internal_line_x_cylinder, y_clock_cylinder+internal_line_y_cylinder), color, 2)

        cv2.imshow("Image Rotation", image)
        cv2.waitKey(10)

