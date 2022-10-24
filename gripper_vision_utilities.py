"""
Description:

Author: David Valencia

"""
import cv2
import math
import numpy as np



class VisionCamera:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)  # open the camera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        self.arucoDict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.markerSize  = 18  # size of the aruco marker millimeters

        self.valve_mark_id = 6
        self.vision_flag_status = False

        full_path_camera_matrix = "/home/david_lab/Repository/low_dimensiona_latent_space_RL/extra_utilities"
        self.matrix = np.loadtxt((full_path_camera_matrix + "/matrix.txt"))
        self.distortion = np.loadtxt((full_path_camera_matrix + "/distortion.txt"))


    def get_camera_image(self):
        ret, frame = self.camera.read()
        if ret:
            return frame
        else:
            print("problem capturing frame")

    def image_function(self):
        while True:
            image = self.get_camera_image()
            cv2.imshow("State visual Completed", image)
            cv2.waitKey(10)

    def pre_pro_image(self, image_array):
        img = cv2.resize(image_array, (128, 128), interpolation=cv2.INTER_AREA)
        # can also crop the image here
        norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #cv2.imshow("Normalized image", norm_image)
        #cv2.waitKey(10)
        return norm_image


    def isclose(self, x, y, rtol=1.e-5, atol=1.e-8):
        return abs(x - y) <= atol + rtol * abs(y)


    def calculate_euler_angles(self, R):
        """
        From a paper by Gregory G. Slabaugh (undated),
        "Computing Euler angles from a rotation matrix
        """
        phi = 0.0
        if self.isclose(R[2, 0], -1.0):
            theta = math.pi / 2.0
            psi = math.atan2(R[0, 1], R[0, 2])
        elif self.isclose(R[2, 0], 1.0):
            theta = -math.pi / 2.0
            psi = math.atan2(-R[0, 1], -R[0, 2])
        else:
            theta = -math.asin(R[2, 0])
            cos_theta = math.cos(theta)
            psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
            phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
        return psi, theta, phi


    def get_angle(self, rot):
        rotation_matrix, _ = cv2.Rodrigues(rot)
        psi, theta, phi = self.calculate_euler_angles(rotation_matrix)
        phi = math.degrees(phi)
        #if phi < -0:
            #phi = phi  + 360
        return phi


    def get_aruco_angle(self):
        image = self.get_camera_image()
        (corners, IDs, rejected) = cv2.aruco.detectMarkers(image, self.arucoDict, parameters=self.arucoParams)
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.markerSize, self.matrix, self.distortion)
        cv2.aruco.drawDetectedMarkers(image, corners, borderColor=(0, 0, 255))
        try:
            if len(IDs) >= 1:
                if IDs[0] == 6:
                    valve_angle = np.array([self.get_angle(rvec[0][0])])
                    self.vision_flag_status = True
                    return valve_angle, self.vision_flag_status

                else:
                    print("valve aruco marker no detected")
                    self.vision_flag_status = False
                    return 0.0, self.vision_flag_status
        except:
            print("valve aruco marker no detected")
            self.vision_flag_status = False
            return 0.0, self.vision_flag_status

