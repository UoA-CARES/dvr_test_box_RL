
import logging
import numpy as np

from Camera import Camera
from gripper_configuration import Gripper, GripperError
from FrameStack import FrameStack

#from cares_lib.vision.ArucoDetector import ArucoDetector
from gripper_aruco_detector import ArucoDetector  # TODO use the lib from cares


class GripperEnvironment:
    def __init__(self, num_motors=4, camera_id=0, device_name="/dev/ttyUSB1", train_mode='vector', robot_id='RR'):

        self.gripper     = Gripper(num_motors=num_motors, device_name=device_name)
        self.camera      = Camera(camera_id=camera_id, robot_id=robot_id)
        self.frame_stack = FrameStack()

        self.aruco_detector = ArucoDetector(marker_size=18)
        self.target_angle   = self.choose_target_angle()

        self.train_mode        = train_mode
        self.object_marker_id  = 6
        self.marker_ids_vector = [0, 1, 2, 3, 4, 5, 6]

        self.noise_tolerance = 5


    def reset(self):
        try:
            current_servo_positions = self.gripper.home()
        except GripperError as error:
            logging.error(error)
            exit()

        marker_pose_all   = self.find_marker_pose(marker_ids_vector=self.marker_ids_vector)
        object_marker_yaw = marker_pose_all[self.object_marker_id][1][2]

        if self.train_mode == 'autoencoder':
            marker_coordinates_all = None
            frame = self.camera.get_frame()
            pre_pro_frame = self.frame_stack.pre_pro_image(frame)
            frame_stack   = self.frame_stack.stack_reset(pre_pro_frame)

        elif self.train_mode == 'vector':
            marker_coordinates_all = self.find_joint_coordinates(marker_pose_all)
            frame_stack = None

        self.target_angle = self.choose_target_angle()
        # Check if the current yaw of the object is not the same as the new target angle,
        # this is in order to avoid getting a reward after doing nothing
        while True:
            angle_difference = np.abs(self.target_angle - object_marker_yaw)
            if angle_difference <= self.noise_tolerance:
                self.target_angle = self.choose_target_angle()
            else:
                logging.info(f"New Goal Angle Generated : {self.target_angle}")
                break

        state = self.define_state_space(current_servo_positions, marker_coordinates_all, frame_stack, object_marker_yaw)
        return state


    def choose_target_angle(self):
        #return np.random.randint(low=0, high=360)
        target_angle = np.random.randint(1, 5)
        if target_angle == 1:
            return 90
        elif target_angle == 2:
            return 180
        elif target_angle == 3:
            return 270
        elif target_angle == 4:
            return 0

    def reward_function(self, target_angle, start_marker_pose, final_marker_pose):
        done = False
        valve_angle_before = start_marker_pose
        valve_angle_after  = final_marker_pose

        angle_difference = np.abs(target_angle - valve_angle_after)
        delta_changes    = np.abs(target_angle - valve_angle_before) - np.abs(target_angle - valve_angle_after)

        if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
            reward = 0
        else:
            reward = delta_changes

        if angle_difference <= self.noise_tolerance:
            logging.info("--------------------Reached the Goal Angle!-----------------")
            reward = reward + 100
            done = True
        return reward, done


    def find_marker_pose(self, marker_ids_vector):
        while True:
            logging.debug(f"Attempting to detect markers ")
            frame        = self.camera.get_frame()
            marker_poses = self.aruco_detector.get_marker_poses(frame, self.camera.camera_matrix, self.camera.camera_distortion)

            # this check if all the seven marker are detected and return all the poses and double check for false detections
            if all(ids in marker_poses for ids in marker_ids_vector) and len(marker_poses) == len(marker_ids_vector):
                break
        return marker_poses

    def find_joint_coordinates(self, markers_pose):
        # the ids detected may have a different order of detection
        # i.e. sometimes the markers_pose index maybe [0, 2, 3] and other [0, 3, 2]
        # so getting x and y coordinates in the right order
        markers_xy_coordinates = []
        for id_index, id_detected in enumerate(markers_pose):
            markers_xy_coordinates.append(markers_pose[id_index][0][0][:-1])
        return markers_xy_coordinates

    def define_state_space(self, servos_position, marker_coordinates, frame_stack, object_marker_yaw):

        # if self.train_mode == 'servos':
        #     servos_position.append(object_marker_yaw)
        #     #servos_position.append(self.target_angle)
        #     state_space_vector = servos_position
        # elif self.train_mode == 'aruco_servos':
        #     coordinate_vector = [element for state_space_list in marker_coordinates for element in state_space_list]
        #     coordinate_vector.append(object_marker_yaw)
        #     for i in servos_position:
        #         coordinate_vector.append(i)
        #     #coordinate_vector.append(self.target_angle)
        #     state_space_vector = coordinate_vector

        state_space_vector = []
        if self.train_mode == 'vector':
            coordinate_vector = [element for state_space_list in marker_coordinates for element in state_space_list]
            coordinate_vector.append(object_marker_yaw)
            #coordinate_vector.append(self.target_angle)
            state_space_vector = coordinate_vector

        elif self.train_mode == 'autoencoder':
            return frame_stack

        return state_space_vector


    def step(self, action):
        start_marker_pose_all   = self.find_marker_pose(marker_ids_vector=self.marker_ids_vector)
        start_object_marker_yaw = start_marker_pose_all[self.object_marker_id][1][2]

        try:
            action_in_steps         = self.gripper.action_to_steps(action)
            current_servo_positions = self.gripper.move(steps=action_in_steps)

        except GripperError as error:
            # handle what to do if the gripper is unrecoverably gone wrong - i.e. save data and fail gracefully
            logging.error(error)
            exit()

        final_marker_pose_all   = self.find_marker_pose(marker_ids_vector=self.marker_ids_vector)
        final_object_marker_yaw = final_marker_pose_all[self.object_marker_id][1][2]

        if self.train_mode == 'autoencoder':
            final_marker_coordinates_all = None
            frame = self.camera.get_frame()
            pre_pro_frame = self.frame_stack.pre_pro_image(frame)
            frame_stack   = self.frame_stack.stack_vector(pre_pro_frame)

        elif self.train_mode == 'vector':
            frame_stack = None
            final_marker_coordinates_all = self.find_joint_coordinates(final_marker_pose_all)

        state        = self.define_state_space(current_servo_positions, final_marker_coordinates_all, frame_stack, final_object_marker_yaw)
        reward, done = self.reward_function(self.target_angle, start_object_marker_yaw, final_object_marker_yaw)
        truncated    = False  # never truncate the episode but here for completion sake
        return state, reward, done, truncated

