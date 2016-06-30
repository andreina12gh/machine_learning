import cv2
import numpy as np
from pre_processsing.hog_descriptor import HogDescriptor
from pre_processsing.segmentation import Segmentation

class Detector:

    def __init__(self):
        self.hog = HogDescriptor()
        self.segmentation = Segmentation()
        self.model = cv2.SVM()
        self.path_train_fire = "../resources/training/fire/train_2.77777777778%.xml"
        self.label_fire = 1
        self.COLOR_FIRE = (0,0,255)


    def load_train(self, path_train):
        self.model.load(path_train)


    def detect_fire(self, image, load_train=True):
        mat_points, image_no_background = self.segmentation.segment(image)
        if load_train:
            self.load_train(self.path_train_fire)
        detected_image = self.get_submats(mat_points,image_no_background, image, self.label_fire, self.COLOR_FIRE)
        return  detected_image


    def get_submats(self, mat_points, image_no_background, image, label, color):
        for (x, y, w, h) in mat_points:
            subMat = image_no_background[y:h, x:w]
            subMat = cv2.resize(subMat, (64, 64))
            [descriptors] = self.hog.get_list_hog_descriptors([subMat])
            result = self.model.predict(descriptors)
            if result==label:
                cv2.rectangle(image,(x,y),(w,h),color,1)
        return image


if __name__ == '__main__':
    detector = Detector()
