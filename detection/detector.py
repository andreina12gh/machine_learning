import cv2
import numpy as np
from pre_processsing.hog_descriptor import HogDescriptor
from pre_processsing.segmentation import Segmentation

class Detector:

    def __init__(self):
        self.hog = HogDescriptor()
        self.segmentation = Segmentation()
        self.model = cv2.SVM()
        self.path_train_fire_segmented = "../resources/training/fire/train_neq_pos_460_neg_51__16.339869281%_C512_gamma_4.xml"
        self.path_train_fire_no_segmented = "../resources/training/fire/train_neq_n_spos_505_neg_57__2.63157894737%_C0.0009765625_gamma_0.0625.xml"
        #self.path_train_fire_segmented = "../resources/training/fire/train_9.5652173913%.xml"
        self.path_train_smoke_no_segmented = "../resources/training/smoke/train_6.94444444444%.xml"
        self.LABEL_FIRE = 1
        self.LABEL_SMOKE = 2
        self.COLOR_FIRE = (0,0,255)
        self.COLOR_SMOKE = (255, 0, 0)
        self.j = 0

    def load_train(self, path_train):
        self.model.load(path_train)

    def sliding_window(self, image, preprocessed_image, label, color, step = 80):
        (height, width, _) = preprocessed_image.shape
        if(step < height and step < width):
            for h in range(0, height - step, step):
                for w in range(0, width - step, step):
                    sub_image = preprocessed_image[h:h + step, w:w + step]
                    sub_image = cv2.resize(sub_image, (64, 64))
                    [descriptors] = self.hog.get_list_hog_descriptors([sub_image])
                    result = self.model.predict(descriptors)
                    if result == label:
                        cv2.rectangle(image, (h, w), (h + step, w + step), color, 2)
        return image

    def detect_fire_segment(self, image, load_train=True):
        mat_points, image_no_background = self.segmentation.segment(image)
        if load_train:
            self.load_train(self.path_train_fire_segmented)
        detected_image = self.get_submats(mat_points, image_no_background, image, self.LABEL_FIRE, self.COLOR_FIRE)
        return detected_image

    def detect_fire(self, image, load_train=True):
        _, image_no_background = self.segmentation.segment(image)
        if load_train:
            self.load_train(self.path_train_fire_no_segmented)
        #image = self.sliding_window(image, image_no_background, self.LABEL_FIRE, self.COLOR_FIRE)
        [descriptors] = self.hog.get_list_hog_descriptors([image_no_background])
        result = self.model.predict(descriptors)
        if result == self.LABEL_FIRE:
            #cv2.rectangle(image, (10,10),(50,50),self.COLOR_FIRE, 1)
            cv2.putText(image,"FUEGO", (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.5, self.COLOR_FIRE)
        return image

    def get_submats(self, mat_points, image_no_background, image, label, color):
        i = 0
        for (x, y, w, h) in mat_points:
            (x, w) = self.swap_points(x, w)
            (y, h) = self.swap_points(y, h)
            if (x is not None and y is not None):
                subMat = image_no_background[y:h, x:w]
                subMat = cv2.resize(subMat, (64, 64))
                [descriptors] = self.hog.get_list_hog_descriptors([subMat])
                result = self.model.predict(descriptors)
                if result == label:
                    if self.j % 24 == 0:
                        pass
                        #cv2.imwrite("../Fuego/img0_"+str(i)+"_"+str(self.j)+".png", subMat)
                    cv2.rectangle(image, (x,y), (w,h), color,1)
                else:
                    if self.j % 24 == 0:
                        pass
                        #cv2.imwrite("../FalsosPositivos/img0_"+str(i)+"_"+str(self.j)+".png", subMat)
                i +=1
                self.j += 1
        return image

    def swap_points(self, x1, x2):
        if x2 < x1:
            aux_x1 = x1
            x1 = x2
            x2 = aux_x1
            res = (x1, x2)
        elif x2 == x1:
            res = (None, None)
        else:
            res = (x1, x2)
        return res

    def detect_smoke_segment(self, image, load_train = True):
        image_d, cany, mat_points = self.segmentation.highlight_smoke_contours(image)
        if load_train:
            self.load_train(self.path_train_smoke_no_segmented)
        image_res = self.get_submats(mat_points, image_d, image, self.LABEL_SMOKE, self.COLOR_SMOKE)
        return image_res

    def detect_smoke(self):
        pass


if __name__ == '__main__':
    detector = Detector()
