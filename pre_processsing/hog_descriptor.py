import numpy as np
import cv2

class HogDescriptor():

    def get_list_hog_descriptors(self, list, winSize=(64, 64), blockSize=(16, 16), blockStride=(8, 8), cellSize=(8, 8), nbins=9):
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        descriptors = []
        #list_mat_descriptors = []
        if(len(list) > 0):
            first_image = cv2.resize(list[0], winSize)
            des = hog.compute(first_image, cellSize)
            #list_mat_descriptors = np.resize(des, (7,7,2,2,9))
            descriptors = np.transpose(des)
            copyList = list
            copyList = copyList[1:]
            for image in copyList:
                im = cv2.resize(image, winSize)
                des = hog.compute(im, (8, 8))
                #mat_des = np.resize(des, (7,7,2,2,9))
                #list_mat_descriptors = np.vstack((list_mat_descriptors, mat_des))
                des = np.transpose(des)
                descriptors = np.vstack((descriptors, des))
        #return list_mat_descriptors, descriptors
        return descriptors
