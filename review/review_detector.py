import cv2
import numpy as np
from detection.detector import Detector
from pre_processsing.preprocessing import Preprocessing
from pre_processsing.segmentation import Segmentation

def testing_detected(path_video):
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
    segmentation = Segmentation()
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

#testing_detected("/home/evelyn/Documents/Fabrica/Videos_e_imgenes/Videos_probar/escena6.mp4")
#testing_detected_smoke("/home/evelyn/Documents/videosss/puente.mp4")
testing_detected("/home/Mauri/Documents/videos/escena0.mp4")
#testing_detected_smoke("/home/Mauri/Documents/videos/escena3.mp4")