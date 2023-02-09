import cv2
import math
import numpy as np
import cares_lib.utils as utils


class ArucoDetector:
    def __init__(self, marker_size, dictionary_id=cv2.aruco.DICT_4X4_50):
        self.dictionary   = cv2.aruco.Dictionary_get(dictionary_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_size  = marker_size

    def get_orientation(self, r_vec):
        r_matrix, _ = cv2.Rodrigues(r_vec)
        roll, pitch, yaw = self.rotation_to_euler(r_matrix)

        def validate_angle(degrees):
            if degrees < 0:
                degrees += 360
            elif degrees > 360:
                degrees -= 360
            return degrees

        roll = validate_angle(math.degrees(roll))
        pitch = validate_angle(math.degrees(pitch))
        yaw = validate_angle(math.degrees(yaw))

        return roll, pitch, yaw

    def rotation_to_euler(self, rotation_matrix):
        # math is based on http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
        def is_close(x, y, rtol=1.e-5, atol=1.e-8):
            return abs(x - y) <= atol + rtol * abs(y)

        yaw = 0.0
        if is_close(rotation_matrix[2, 0], -1.0):
            pitch = math.pi / 2.0
            roll = math.atan2(rotation_matrix[0, 1], rotation_matrix[0, 2])
        elif is_close(rotation_matrix[2, 0], 1.0):
            pitch = -math.pi / 2.0
            roll = math.atan2(-rotation_matrix[0, 1], -rotation_matrix[0, 2])
        else:
            pitch = -math.asin(rotation_matrix[2, 0])
            cos_theta = math.cos(pitch)
            roll = math.atan2(rotation_matrix[2, 1] / cos_theta, rotation_matrix[2, 2] / cos_theta)
            yaw = math.atan2(rotation_matrix[1, 0] / cos_theta, rotation_matrix[0, 0] / cos_theta)
        return roll, pitch, yaw


    def get_pose(self, t_vec, r_vec):
        pose = t_vec
        orientation = self.get_orientation(r_vec)
        return pose, orientation

    def get_marker_poses(self, image, camera_matrix, camera_distortion):
        marker_poses = {}
        (corners, ids, rejected_points) = cv2.aruco.detectMarkers(image, self.dictionary, parameters=self.aruco_params)

        if len(corners) > 0:
            r_vecs, t_vecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, camera_matrix, camera_distortion)

            image_copy = image.copy()
            cv2.aruco.drawDetectedMarkers(image_copy, corners, ids, borderColor=(0, 0, 255))

            for i in range(0, len(r_vecs)):
                cv2.drawFrameAxes(image_copy, camera_matrix, camera_distortion, r_vecs[i], t_vecs[i], self.marker_size / 2.0, 3)

            cv2.imshow("Frame", image_copy)
            cv2.waitKey(100)

            for i in range(0, len(r_vecs)):
                id    = ids[i][0]
                r_vec = r_vecs[i]
                t_vec = t_vecs[i]
                # TODO: change this to output something less bulky than two arrays
                marker_poses[id] = self.get_pose(t_vec, r_vec)

        return marker_poses
