import numpy as np
import cv2

class HogDescriptor():

    def get_list_hog_descriptors(self, list, winSize=(64, 64), blockSize=(16, 16), blockStride=(8, 8), cellSize=(8, 8), nbins=9):
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        if(len(list) > 0):
            first_image = cv2.resize(list[0], winSize)
            descriptors = np.transpose(hog.compute(first_image, cellSize))
            copyList = list
            copyList = copyList[1:]
            for image in copyList:
                im = cv2.resize(image, winSize)
                des = hog.compute(im, (8, 8))
                des = np.transpose(des)
                descriptors = np.vstack((descriptors, des))
        return descriptors
