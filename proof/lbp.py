'''import cv2
import numpy as np
from skimage import feature

def calc_lbp(frame):
    valor = frame[1][1]
    mat = [[7,6,5],[0,-1,4],[1,2,3]]
    for i in range(0,3):
        for j in range (0,3):
            if frame[i][j]<=valor:
                pow_value = mat[i][j]
                if pow>=0:
                    frame[i][j]=(2**pow_value)
            else:
                frame[i][j] = 0
    frame[1][1]=0
    promedio = np.resize(frame, (1,9))
    promedio = np.sum(promedio, 1)
    return promedio

frame = cv2.imread("../resources/images/smoke/img_0_7.5.png")
frame = cv2.resize(frame, (200,100))
cv2.imshow("res", frame)

frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

frame_res = np.zeros((200,100))
print len(frame)
print len(frame[0])
for j in range (1, len(frame)-1, 1):
    for k in range (1, len(frame[0])-1, 1):
        submat = frame[j-1:j+2,k-1:k+2]
        #vector = np.resize(submat, (1))
        value = calc_lbp(submat)
        frame_res[k-1][j-1] = value
        #print frame_res
        print str(j-1), str(k-1)
print frame_res

cv2.imshow("result", frame_res)
cv2.waitKey(0)
'''
from skimage import feature
import numpy as np
import cv2

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)


		# return the histogram of Local Binary Patterns
		return hist, lbp

lbp = LocalBinaryPatterns(24, 8)
frame = cv2.imread("/home/evelyn/Desktop/download.jpg")
frame =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = cv2.resize(frame, (600,480))


hist, lbp_img = lbp.describe(frame)
cv2.imshow("res", lbp_img)
cv2.waitKey(0)
print hist