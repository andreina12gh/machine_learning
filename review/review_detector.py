import cv2
import numpy as np
from detection.detector import Detector
from pre_processsing.preprocessing import Preprocessing
from pre_processsing.segmentation import Segmentation
from numpy.linalg import inv
from matplotlib import pyplot as plt

def testing_detected(path_video):
    capture = cv2.VideoCapture(path_video)
    detector = Detector()
    preprocessing = Preprocessing()
    load_train = True
    detect_segment = True
    scalar = 255
    #video = cv2.VideoWriter("../resources/videos/video_escena0.avi",fourcc=cv2.cv.CV_FOURCC('m','p','4','v'),fps=10,frameSize=(640,480))

    while(1):
        _, frame = capture.read()
        frame = cv2.resize(frame, (640, 640))
        res_1 = detector.detect_fire_segment(frame,load_train)
        cv2.imshow("original", res_1)
        frame = quitar_blanco_clahe(frame)
        #comment the line of down if the image is readed automatically in the model of color BGR
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if detect_segment:
            frame_detected = detector.detect_fire_segment(frame, load_train)
        else:
            frame_detected = detector.detect_fire(frame, load_train)

        cv2.imshow("Test", frame_detected)
        #video.write(frame_detected)
        '''frame_mul = frame * scalar
        b, g, r = cv2.split(frame_mul)
        r_max = np.max(r)
        g_max = np.max(g)
        b_max = np.max(b)
        print r_max, g_max, b_max
        r_1 = r*(scalar/r_max)
        g_1 = g*(scalar/g_max)
        b_1 = b*(scalar/b_max)
        suma = r +g+b
        r = r*inv(suma)
        g = g *inv(suma)
        b = b*inv(suma)
        print "r",r
        print "g",g
        print "b",b
        frame_mul = cv2.merge((r_1,g_1,b_1))
        frame_new = cv2.merge((r, g, b))
        cv2.imshow("mul", frame_mul)
        cv2.imshow("res", frame_new)
        load_train = False
        image_Res = preprocessing.get_image_brightness(frame)'''
        '''
        b, g, r = cv2.split(frame_detected)
        frame_detected_hsv = cv2.cvtColor(frame_detected, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(frame_detected_hsv)
        print "h", h
        b = b -h
        g = g -h
        r = r -h
        frame_detected = cv2.merge((b,g,r))'''
        #cv2.imshow("Res", image_Res)
        k = cv2.waitKey(1)
        if k == 27:
            #video.release()
            break
    capture.release()
    #video.release()
    cv2.destroyAllWindows()

def testing_detected_smoke(path_video):
    capture = cv2.VideoCapture(path_video)
    detector = Detector()
    preprocessing = Preprocessing()
    load_train = True
    detect_segment = True
    segmentation = Segmentation()
    while(1):
        _, frame = capture.read()
        frame = cv2.resize(frame, (640, 480))
        #comment the line of down if the image is readed automatically in the model of color BGR
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        '''if detect_segment:
            frame_detected = detector.detect_fire_segment(frame, load_train)
        else:
            frame_detected = detector.detect_fire(frame, load_train)
        cv2.imshow("Test", frame_detected)
        image_Res = preprocessing.get_image_brightness(frame)
        cv2.imshow("Res", image_Res)'''
        #frame, cany, submats = segmentation.highlight_smoke_contours(frame)
        frame = detector.detect_smoke_segment(frame, load_train)
        load_train = False

        cv2.imshow("frame", frame)
        #cv2.imshow("ca", cany)
        #cv2.imshow("cany", cany)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()

def quitar_blanco_video(path_video):
    capture = cv2.VideoCapture(path_video)

    while(1):
        _, frame = capture.read()
        frame = cv2.resize(frame, (640, 480))
        b, g,r = cv2.split(frame)
        eq_b = cv2.equalizeHist(b)
        eq_g = cv2.equalizeHist(g)
        eq_r = cv2.equalizeHist(r)
        res_b = np.hstack((b, eq_b))
        res_g = np.hstack((g, eq_g))
        res_r = np.hstack((r, eq_r))
        res =cv2.merge((res_b, res_g, res_r))

        cv2.imshow("frame",res)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()

def quitar_blanco_equal(frame):
    b, g,r = cv2.split(frame)
    eq_b = cv2.equalizeHist(b)
    eq_g = cv2.equalizeHist(g)
    eq_r = cv2.equalizeHist(r)
    res_b = np.hstack((b, eq_b))
    res_g = np.hstack((g, eq_g))
    res_r = np.hstack((r, eq_r))
    res =cv2.merge((res_b, res_g, res_r))
    return res

def quitar_blanco_clahe(frame):
    b, g,r = cv2.split(frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(b)
    cl2 = clahe.apply(g)
    cl3 = clahe.apply(r)
    res =cv2.merge((cl1,cl2,cl3))
    return  res

#quitar_blanco("/home/evelyn/Documents/videosss/puente.mp4")
testing_detected("/home/evelyn/Documents/videosss/puente.mp4")
#testing_detected("/home/evelyn/Documents/Fabrica/Videos_e_imgenes/Videos_probar/escena6.mp4")
#testing_detected_smoke("/home/evelyn/Documents/videosss/puente.mp4")
#testing_detected("/home/Mauri/Documents/videos/escena0.mp4")
#testing_detected_smoke("/home/Mauri/Documents/videos/escena3.mp4")
'''frame =cv2.imread("/home/evelyn/Desktop/organizar/imgEduardo.png")
detector = Detector()
preprocessing = Preprocessing()
load_train = True
detect_segment = True
scalar = 255
frame = cv2.resize(frame, (640, 480))
cv2.imshow("Original", frame)
frame_detected = detector.detect_fire_segment(frame, load_train)
cv2.imshow("Resultado", frame_detected)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
'''