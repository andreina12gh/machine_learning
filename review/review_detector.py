import cv2
import numpy as np
from detection.detector import Detector


def testing_detected(path_video):
    capture = cv2.VideoCapture(path_video)
    detector = Detector()
    load_train = True
    while(1):
        _, frame = capture.read()
        frame = cv2.resize(frame, (640,480))
        frame_detected = detector.detect_fire(frame, load_train)
        cv2.imshow("Test", frame_detected)
        load_train=False
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()

testing_detected("/home/evelyn/Documents/Fabrica/SVMHOG/Videos/escena0.mp4")