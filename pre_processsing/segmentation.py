import cv2
import numpy as np
from preprocessing import Preprocessing

class Segmentation:

    def __init__(self):
        self.pre_processing = Preprocessing()
        self.LOWER_GRAY_2 = np.array([0, 0, 100])
        self.UPPER_GRAY_2 = np.array([255, 80, 175])

    def segment(self, image):
        mask, image_no_background = self.pre_processing.cut_out_backgound(image)
        image_hue = self.pre_processing.get_mask_brightness(image_no_background)
        cv2.imshow("img_hue", image_no_background)
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

    def get_points_min_max(self, array):
        x, y = cv2.split(array)
        min_x = min(x)[0]
        x = list(x)
        pos_min_x = x.index(min_x)
        min_y = y[pos_min_x][0]
        max_x = max(x)[0]
        pos_max_x = x.index(max_x)
        max_y = y[pos_max_x][0]
        return (min_x, min_y),(max_x, max_y)

    def highlight_smoke_contours(self, image):
        mask = self.pre_processing.enhance_color(self.LOWER_GRAY_2, self.UPPER_GRAY_2, image)
        contours, cany = self.pre_processing.border_image(mask, image)
        i = 0
        sub_mats = []
        for c in contours:
            extension = cv2.contourArea(c)
            if extension > 600:
                contour = contours[i]
                (x_min, y_min), (x_max, y_max) = self.get_points_min_max(contour)
                sub_mats.append((x_min, y_min, x_max, y_max))
                self.pre_processing.draw_image(image, 1, [(x_min, y_min), (x_max, y_max)], (255,0,0))
                self.pre_processing.draw_image(image, 2, contour, (0,255,0))
                self.pre_processing.draw_image(cany, 2, contour,(0,255,0))
            i = i + 1
        return image, cany, sub_mats
