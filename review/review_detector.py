import cv2
import numpy as np
from detection.detector import Detector
from pre_processsing.preprocessing import Preprocessing

def testing_detected(path_video):
    capture = cv2.VideoCapture(path_video)
    detector = Detector()
    preprocessing = Preprocessing()
    load_train = True
    detect_segment = True
    while(1):
        _, frame = capture.read()
        frame = cv2.resize(frame, (640, 480))
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

testing_detected("/home/evelyn/Documents/Fabrica/SVMHOG/Videos/escena13.mp4")