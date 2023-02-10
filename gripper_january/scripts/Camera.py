
import cv2
import math
import numpy as np


from pathlib import Path
file_path = Path(__file__).parent.resolve()


class Camera(object):
    def __init__(self, camera_id=0, robot_id='RR'):

        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise Exception("Could not open video device")

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)  # aruco dictionary
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_size = 18  # mm

        if robot_id == 'RR':
            self.camera_matrix     = np.loadtxt(f"{file_path}/config/camera_matrix_RR.txt")
            self.camera_distortion = np.loadtxt(f"{file_path}/config/camera_distortion_RR.txt")
        else:
            #todo add camera calibration Files for Robot Left
            self.camera_matrix     = np.loadtxt(f"{file_path}/config/camera_matrix_RR.txt")
            self.camera_distortion = np.loadtxt(f"{file_path}/config/camera_distortion_RR.txt")

    def get_frame(self):
        returned, frame = self.camera.read()
        if returned:
            return frame
        print("Error: No frame returned")
        return None

