import pywt
import numpy as np
import cv2

def w2d(img, mode='haar', level=1):
    #imArray = cv2.imread(img)
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray)
    imArray /=255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0]*=0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    cv2.imshow("image", imArray_H)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
#w2d("/home/evelyn/Desktop/organizar/imgEduardo.png", 'db1', 7)

capture = cv2.VideoCapture("/home/evelyn/Documents/videosss/puente.mp4")
while(1):
    _, frame = capture.read()
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("original", frame)
    frame = w2d(frame, 'db1', 7)
    k = cv2.waitKey(1)
    if k == 27:
        #video.release()
        break
capture.release()
#video.release()
cv2.destroyAllWindows()

