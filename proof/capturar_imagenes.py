import cv2

capture = cv2.VideoCapture("/media/evelyn/D4CB-805B/Videos/video4.crdownload")

i = 0
while(1):

    _, frame = capture.read()
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Video", frame)

    if i % 26 == 0:
        ruta = "/home/evelyn/Desktop/Imagenes/img_bariloche_"+str(i)+"_4.png"
        cv2.imwrite(ruta, frame)

    i+= 1
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()
