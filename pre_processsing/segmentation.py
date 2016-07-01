import cv2
import numpy as np
from preprocessing import Preprocessing

class Segmentation:

    def __init__(self):
        self.pre_processing = Preprocessing()

    def segment(self, image):
        mask, image_no_background = self.pre_processing.cut_out_backgound(image)
        mat_points = self.map_out(mask, image_no_background)
        return mat_points, image_no_background

    def map_out(self, img_bin, image):
        mat_points = []
        contours, inheriters = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            moments = cv2.moments(c)
            if moments['m00'] > 200:
                x = []
                y = []
                for i in c:
                    for j in i:
                        x.append(j[0])
                        y.append(j[1])
                max_x, min_x, max_y, min_y = np.argmax(x), np.argmin(x), np.argmax(y), np.argmin(y)
                mat_points.append((x[min_x], y[min_y], x[max_x], y[max_y]))
        return mat_points


