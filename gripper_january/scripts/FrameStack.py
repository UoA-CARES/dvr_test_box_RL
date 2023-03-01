
import numpy as np
import cv2
from collections import deque



class FrameStack:
    def __init__(self, k=3):
        self.k  = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

    def stack_reset(self, frame):
        for _ in range(self.k):
            self.frames_stacked.append(frame)
        return np.array(list(self.frames_stacked))


    def pre_pro_image(self, frame):
        #frame = frame[240:820, 70:1020]
        frame  = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        img_gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        norm_image  = cv2.normalize(img_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #cv2.imshow("Normalized image", norm_image)
        #cv2.waitKey(0)
        return norm_image


    def stack_vector(self, frame):
        self.frames_stacked.append(frame)
        return np.array(list(self.frames_stacked))
