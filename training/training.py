import cv2
import numpy as np
import os
from pre_processsing.preprocessing import Preprocessing
from pre_processsing.filter_gabor import FilterGabor
from pre_processsing.hog_descriptor import HogDescriptor

class Training:
    def __init__(self):
        self.path_dir_image_fire = "../images/fire/"
        self.path_dir_image_smoke = "../images/smoke/"
        self.path_dir_image_false_positive = "../images/false_positive/"
        self.preprocessing = Preprocessing()
        self.filter_gabor = FilterGabor()
        self.hog_descriptor = HogDescriptor()
        self.label_fire = 1
        self.label_smoke = 2
        self.label_false_positive = -2


    def load_data(self, path_dir, label, apply_preprocessing_fire):
        list_image = []
        list_label = []
        for file in os.listdir(path_dir):
            image = cv2.imread(path_dir + file)
            image = cv2.resize(image, (64, 64))
            if apply_preprocessing_fire:
                image = self.preprocessing.cut_out_backgound(image)
            else:
                image = self.preprocessing.highlight_smoke_features(image)
            list_image.append(image)
            list_label.append(label)
        return list_image, list_label

    def split_data_by_train(self, list_descriptor, list_label):
        size_list = len(list_descriptor)
        list_by_train, list_by_test = np.split(list_descriptor, [int(size_list * 0.8)])
        labels_by_train, labels_by_test = np.split(list_label, [int(size_list * 0.8)])
        size_list_test = len(list_by_test)
        list_by_test, list_cross_validation = np.split(list_by_test, [int(size_list_test * 0.5)])
        labels_by_test, labels_cross_validation = np.split(labels_by_test, [int(size_list_test * 0.5)])
        print "image_test 0.1", len(list_by_test)

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
        responses = np.array(list_label)
        svm.train(list_descriptor, responses, None, None, params=svm_params)
        svm.save(path_train)
        svm.load(path_train)
        test_hog = list_descriptor_test
        resp = svm.predict_all(test_hog).ravel()
        err = (list_label_test != resp).mean()
        if (err < minimum):
            minimo = err

            print 'gamma = ' + str(gamma) + "   C = " + str(C)
            print 'error: %.2f %%' % (err * 100)

        return gamma, C, minimo

    def generate_data_training(self, path_dir, label, state):
        list_image, list_label = self.load_data(self.path_dir, label, state)
        list_descriptor = self.hog_descriptor.get_list_hog_descriptors(list_image, winSize=(64, 64), blockSize=(16, 16), blockStride=(8, 8), cellSize=(8, 8), nbins=9)
        (list_by_train, list_by_test, labels_by_train, labels_by_test) = self.split_data_by_train(list_descriptor, list_label)
        return (list_by_train, list_by_test, labels_by_train, labels_by_test)

    def generate_training(self):
        (list_by_train_fire, list_by_test_fire, labels_by_train_fire, labels_by_test_fire) = self.generate_data_training(self.path_dir_image_fire, self.label_fire, True)
        (list_by_train_fp, list_by_test_fp, labels_by_train_fp, labels_by_test_fp) = self.generate_data_training(self.path_dir_image_false_positive, self.label_false_positive)


    def save_training(self, list_descriptor, list_label, list_descriptor_test, list_label_test, range_gamma, range_C, path_train, minimum_error = 1):
        (min_gamma, max_gamma) = range_gamma
        (min_C, max_C) = range_C
        path_default = ""
        for i in range(min_gamma, max_gamma, 1):
            for j in range(min_C, max_C, 1):
                gamma, C, error = self.train(list_descriptor, list_label, 2**(i),2**(j), path_default, minimum_error)
                if(error < minimum_error):
                    minimum_error = error
                    print 'gamma = ' + str(gamma) + "   C = " + str(C)
                    print 'error: %.2f %%' % (error * 100)
                    val_gamma = gamma
                    val_C = C
        self.train(list_descriptor, list_label, list_descriptor_test, list_label_test, val_gamma, val_C, path_train, minimum_error)



    i = 0

    # ambos ciclos es para dos directorios diferentes
    for f in os.listdir(humo):
        image = cv2.imread(humo + f)
        image = cv2.resize(image, (64, 64))

        image = process_image(image)
        # cv2.imwrite("/home/evelyn/Desktop/entrenamiento/img_humo_"+str(i)+".png", image)
        image_list.append(image)
        labels.append(2)
        i += 1

    for f in os.listdir(humo_2):
        image = cv2.imread(humo_2 + f)
        image = cv2.resize(image, (64, 64))

        image = process_image(image)
        # cv2.imwrite("/home/evelyn/Desktop/entrenamiento/img_humo_"+str(i)+".png", image)
        image_list.append(image)
        labels.append(2)
        i += 1

    image_train, image_test = np.split(image_list, [int(i * 0.8)])
    labels_train, labels_test = np.split(labels, [int(i * 0.8)])
    print "image_train 0.8", len(image_train)

    i = len(labels_test)
    labels_test, labels_cross_validation_h = np.split(labels_test, [int(i * 0.5)])
    image_test, image_cross_validation_h = np.split(image_test, [int(i * 0.5)])
    print "image_test 0.1", len(image_test)

    image_train = np.concatenate([image_train, image_cross_validation_h])
    image_test = np.concatenate([image_test, image_cross_validation_h])

    labels_train = np.concatenate([labels_train, labels_cross_validation_h])
    labels_test = np.concatenate([labels_test, labels_cross_validation_h])

    image_humo = []
    labels_humo = []
    i = 0
    '''
    for f in os.listdir(vela):
        image = cv2.imread(vela + f)
        image = cv2.resize(image, (64,64))
        #image = procesar_humo(image)

        #image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        image_humo.append(image)
        labels_humo.append(2)
        i += 1

    '''
    # print image_humo
    image_train_humo, image_test_humo = np.array_split(image_humo, [int(i * 0.9)])
    labels_train_humo, labels_test_humo = np.split(labels_humo, [int(i * 0.9)])

    i = 0
    image_list_neg = []
    labels_neg = []
    # lbp = LocalBinaryPatterns(8,3)
    for f in os.listdir(neg):
        image = cv2.imread(neg + f)
        image = cv2.resize(image, (64, 64))

        # image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        image = process_image(image)

        # cv2.imwrite("/home/evelyn/Desktop/entrenamiento/neg/img_neg_"+str(i)+".png", image)
        image_list_neg.append(image)
        labels_neg.append(-2)
        i += 1
    image_train_neg, image_test_neg = np.split(image_list_neg, [int(i * 0.8)])
    labels_train_neg, labels_test_neg = np.split(labels_neg, [int(i * 0.8)])

    i = len(labels_test_neg)

    labels_test_neg, labels_cross_validation = np.split(labels_test_neg, [int(i * 0.5)])
    image_test_neg, image_cross_validation = np.split(image_test_neg, [int(i * 0.5)])

    image_train_neg = np.concatenate([image_train_neg, image_cross_validation])
    image_test_neg = np.concatenate([image_test_neg, image_cross_validation])

    labels_train_neg = np.concatenate([labels_train_neg, labels_cross_validation])
    labels_test_neg = np.concatenate([labels_test_neg, labels_cross_validation])

    image_train = np.concatenate([image_train, image_train_neg])
    image_test = np.concatenate([image_test, image_test_neg])
    labels_train = np.concatenate([labels_train, labels_train_neg])
    labels_test = np.concatenate([labels_test, labels_test_neg])

    # Revisar el codigo para unirlo los entrenamientos con humo
    '''
    labels_train = np.concatenate([labels_train, labels_train_humo])
    labels_test = np.concatenate([labels_test,labels_test_humo])
    image_test = np.concatenate([image_test, image_test_humo])
    image_train = np.concatenate([image_train, image_train_humo])
    #print image_train.shape'''
    # image_test = np.float32(image_test)
    # image_train = np.float32(image_train)

    image_test = hog_list(image_test)
    # entrenar
    descriptors = hog_list(image_train)
    # image_test = np.float32(image_test)
    # image_train = np.float32(image_train)


    minimo = 1
    '''
    for i in range(-15, 3, 1):
        for j in range(-10, 10, 1):
            gamma, C, minimo = train(descriptors, labels_train,2**(i),2**(j), minimo)
    '''
    train(descriptors, labels_train, 0.0078125, 0.0009765625, minimo)
    # train(descriptors, labels_train)

if __name__ == '__main__':
    training = Training