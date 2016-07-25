import cv2
import numpy as np
import os
from pre_processsing.preprocessing import Preprocessing
from pre_processsing.filter_gabor import FilterGabor
from pre_processsing.hog_descriptor import HogDescriptor
from pre_processsing.segmentation import Segmentation


class Training:
    def __init__(self):
        self.path_dir_image_fire = "../resources/images/fire/"
        self.path_dir_image_fire_1 = "../resources/images/fire_1/"
        self.path_dir_image_smoke = "../resources/images/smoke/"
        self.path_dir_image_smoke_1 = "../resources/images/smoke_1/"
        self.path_dir_image_false_positive = "../resources/images/false_positive/"
        self.path_dir_image_false_positive_1 = "../resources/images/false_positive_1/"
        self.path_dir_image_false_positive_fire = "../resources/images/false_positive_2/"
        self.preprocessing = Preprocessing()
        self.filter_gabor = FilterGabor()
        self.hog_descriptor = HogDescriptor()
        self.segmentation = Segmentation()
        self.label_fire = 1
        self.label_smoke = 2
        self.label_false_positive = -2


    def load_data(self, path_dir, label, apply_preprocessing_fire, segment):
        list_image = []
        list_label = []
        i = 0
        for file in os.listdir(path_dir):
            image = cv2.imread(path_dir + file)
            if apply_preprocessing_fire:
                mask, image = self.preprocessing.cut_out_backgound(image)
            else:
                image = self.preprocessing.highlight_smoke_features(image)
            if segment:
                mat_points = self.segmentation.map_out(mask, image)
                list = self.load_segment_image(mat_points, image)
                for j in range(0, len(list)):
                    if i <255:
                        image_64 = cv2.resize(list[j], (64, 64))
                        #path = "/home/evelyn/Desktop/imgs/img_"+str(j)+"_"+str(label)+".png"
                        #cv2.imwrite(path, image_64)
                        list[j]=image_64
                        list_image.append(list[j])
                        list_label.append(label)
                    else:
                        return list_image, list_label
                    i+=1

            else:
                list_image.append(image)
                list_label.append(label)
                ''''else:
                    image = self.preprocessing.highlight_smoke_features(image)
                    list_image.append(image)
                    list_label.append(label)'''
        cv2.destroyAllWindows()
        return list_image, list_label


    def load_segment_image(self, mat_points, image):
        list =[]
        for (x, y, w, h) in mat_points:
            subMat = image[y:h, x:w]
            subMat = cv2.resize(subMat, (64, 64))
            list.append(subMat)
        return list


    def split_data_by_train(self, list_descriptor, list_label):
        size_list = len(list_descriptor)
        list_by_train, list_by_test = np.split(list_descriptor, [int(size_list * 0.8)])
        labels_by_train, labels_by_test = np.split(list_label, [int(size_list * 0.8)])
        size_list_test = len(list_by_test)
        list_by_test, list_cross_validation = np.split(list_by_test, [int(size_list_test * 0.5)])
        labels_by_test, labels_cross_validation = np.split(labels_by_test, [int(size_list_test * 0.5)])

        list_by_train = np.concatenate([list_by_train, list_cross_validation])
        list_by_test = np.concatenate([list_by_test, list_cross_validation])

        labels_by_train = np.concatenate([labels_by_train, labels_cross_validation])
        labels_by_test = np.concatenate([labels_by_test, labels_cross_validation])
        return (list_by_train, list_by_test, labels_by_train, labels_by_test)


    def train(self, list_descriptor, list_label, list_descriptor_test, list_label_test, gamma, C, path_train, minimum):
        svm_params = dict(kernel_type=cv2.SVM_RBF,
                          svm_type=cv2.SVM_C_SVC,
                          degree=3,
                          coef0=0.0,
                          term_crit=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3),
                          gamma=gamma,
                          nu=0.5,
                          p=0.1,
                          Cvalue=C)
        svm = cv2.SVM()
        #responses = np.array(list_label)
        svm.train(list_descriptor, list_label, None, None, params=svm_params)
        svm.save(path_train)
        svm.load(path_train)
        test_hog = list_descriptor_test
        resp = svm.predict_all(test_hog).ravel()
        err = (list_label_test != resp).mean()
        if (err < minimum):
            minimum = err
        return gamma, C, minimum


    def generate_data_descriptor_training(self, path_dir, label, state, segment):
        print "genr", path_dir
        list_image, list_label = self.load_data(path_dir, label, state, segment)
        list_descriptor = self.hog_descriptor.get_list_hog_descriptors(list_image)
        (list_by_train, list_by_test, labels_by_train, labels_by_test) = self.split_data_by_train(list_descriptor, list_label)
        return (list_by_train, list_by_test, labels_by_train, labels_by_test)


    def generate_data_training(self, list_path_to_train, label, type_train, segment):
        (list_training, list_testing, labels_training, labels_testing) = self.get_list_training(list_path_to_train, label, type_train, segment)
        (list_training_fp, list_testing_fp, labels_training_fp, labels_testing_fp) = self.get_list_training([self.path_dir_image_false_positive, self.path_dir_image_false_positive_1], self.label_false_positive, type_train, segment)
        list_training = np.concatenate([list_training, list_training_fp])
        labels_training = np.concatenate([labels_training, labels_training_fp])
        list_testing = np.concatenate([list_testing, list_testing_fp])
        labels_testing =np.concatenate([labels_testing, labels_testing_fp])
        return (list_training, list_testing, labels_training, labels_testing)

    def get_list_training(self, list_path_to_train, label, type_train, segment):
        first = True
        for path_train in list_path_to_train:
            (list_by_train, list_by_test, labels_by_train, labels_by_test) = self.generate_data_descriptor_training(path_train, label, type_train, segment)
            if(first):
                list_training = list_by_train
                list_testing = list_by_test
                labels_training = labels_by_train
                labels_testing = labels_by_test
                first = False
            else:
                list_training = np.concatenate([list_training, list_by_train])
                list_testing = np.concatenate([list_testing, list_by_test])
                labels_training = np.concatenate([labels_training, labels_by_train])
                labels_testing = np.concatenate([labels_testing, labels_by_test])

        return (list_training, list_testing, labels_training, labels_testing)


    def generate_training(self, type_train_fire, segment):
        if(type_train_fire):
            path_train = "../resources/training/fire/train"
            list_path_dir = [self.path_dir_image_fire, self.path_dir_image_fire_1]
            (list_by_train, list_by_test, labels_by_train, labels_by_test) = self.generate_data_training(list_path_dir, self.label_fire, type_train_fire, segment)
        else:
            path_train = "../resources/training/smoke/train"
            list_path_dir = [self.path_dir_image_smoke]
            (list_by_train, list_by_test, labels_by_train, labels_by_test) = self.generate_data_training(list_path_dir, self.label_smoke, type_train_fire, segment)
        self.save_training(list_by_train, labels_by_train, list_by_test, labels_by_test, (-15, 3), (-10, 10), path_train)


    def save_training(self, list_descriptor, list_label, list_descriptor_test, list_label_test, range_gamma, range_C, path_train, minimum_error = 1):
        (min_gamma, max_gamma) = range_gamma
        (min_C, max_C) = range_C
        path_default = path_train+"_prueba.xml"
        for i in range(min_gamma, max_gamma, 1):
            for j in range(min_C, max_C, 1):
                gamma, C, error = self.train(list_descriptor, list_label, list_descriptor_test, list_label_test, 2**(i), 2**(j), path_default, minimum_error)
                if(error < minimum_error):
                    minimum_error = error
                    print 'gamma = ' + str(gamma) + "   C = " + str(C)
                    print 'error: %.2f %%' % (error * 100)
                    val_gamma = gamma
                    val_C = C
        path_train = path_train+"_"+str(minimum_error*100)+"%.xml"
        self.train(list_descriptor, list_label, list_descriptor_test, list_label_test, val_gamma, val_C, path_train, minimum_error)
        os.remove(path_default)


if __name__ == '__main__':
    training = Training()
    # If type train is True, it will generate a train of FIRE, otherwise, of SMOKE
    type_train_fire = True
    segment = False

    training.generate_training(type_train_fire, segment)