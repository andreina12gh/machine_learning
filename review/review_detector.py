import cv2
import numpy as np
from detection.detector import Detector
from pre_processsing.preprocessing import Preprocessing

def testing_detected(path_video):
    capture = cv2.VideoCapture(path_video)
    detector = Detector()
    preprocessing = Preprocessing()
    load_train = True
    detect_segment = False
    while(1):
        _, frame = capture.read()
        frame = cv2.resize(frame, (640, 480))
        #comment the line of down if the image is readed automatically in the model of color BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if detect_segment:
            frame_detected = detector.detect_fire_segment(frame, load_train)
        else:
            frame_detected = detector.detect_fire(frame, load_train)
        cv2.imshow("Test", frame_detected)
        load_train = False
        image_Res = preprocessing.get_image_brightness(frame)
        cv2.imshow("Res", image_Res)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()

def testing_detected_smoke(path_video):
    capture = cv2.VideoCapture(path_video)
    detector = Detector()
    preprocessing = Preprocessing()
    load_train = True
    detect_segment = True
    while(1):
        _, frame = capture.read()
        frame = cv2.resize(frame, (640, 480))
        #comment the line of down if the image is readed automatically in the model of color BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        '''if detect_segment:
            frame_detected = detector.detect_fire_segment(frame, load_train)
        else:
            frame_detected = detector.detect_fire(frame, load_train)
        cv2.imshow("Test", frame_detected)
        load_train = False
        image_Res = preprocessing.get_image_brightness(frame)
        cv2.imshow("Res", image_Res)'''
        frame_smoke = preprocessing.highlight_smoke_features(frame)
        cv2.imshow("h_s", frame_smoke)
        cv2.imshow("original", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()

testing_detected("/home/Mauri/Documents/videos/escena6.mp4")
#testing_detected_smoke("/home/Mauri/Documents/videos/escena0.mp4")