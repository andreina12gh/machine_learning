import numpy as np
import cv2
from filter_gabor import FilterGabor

class Preprocessing:

    def __init__(self):
        self.filter_gabor = FilterGabor()
        self.LOWER_RED = np.array([0, 105, 155])
        self.UPPER_RED = np.array([32, 237, 255])
        self.LOWER_GRAY = np.array([100, 15, 0])
        self.UPPER_GRAY = np.array([255, 255, 255])

    def cut_out_backgound(self, image):
        mask = self.enhance_color(self.LOWER_RED, self.UPPER_RED, image)
        image_no_backgrund = cv2.bitwise_and(image, image, mask=mask)
        return mask, image_no_backgrund

    def enhance_color(self, lower, upper, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image_hsv, lower, upper)
        return mask

    def convert_image_nbits(self, image, x, y, z):
        image = np.array(image)
        b, g, r = cv2.split(image)
        layer_b = self.get_layer(b, x)
        layer_g = self.get_layer(g, y)
        layer_r = self.get_layer(r, z)
        image_nbits = cv2.merge((layer_b, layer_g, layer_r))
        return image_nbits

    def get_layer_nbit(self, layer, nbit):
        pow2 = 2**nbit
        value = 256 / pow2
        layer /= value
        layer *= value
        return layer

    def get_layer(self, image, nlayer):
        b, g, r = cv2.split(image)
        layers = [b, g, r]
        layer = None
        if nlayer >= 0 and nlayer < 3:
            layer = layers[nlayer]
        return layer

    def combine_layers(self, image, image_nbits, nlayer_img, nlayer_img_nbits):
        layer_res = None
        layer_o = self.get_layer(image, nlayer_img)
        layer_nbits = self.get_layer(image_nbits, nlayer_img_nbits)
        if(layer_o and layer_nbits):
            layer_res = layer_o + layer_nbits
        return layer_res

    def highlight_smoke_features(self, image):
        mask = cv2.inRange(image, self.LOWER_GRAY, self.UPPER_GRAY)
        masked_image = cv2.bitwise_or(image, image, mask=mask)
        gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        gray_image = np.array(gray_image, dtype=np.float32)
        norm_gray_image = gray_image / 255.
        image_gabor = self.filter_gabor.apply_filter(norm_gray_image, ks=21, sig=5, lm=1.3, th=90, ps=88)
        image_gabor = image_gabor * 255
        image_gabor = np.array(image_gabor, dtype=np.uint8)
        return image_gabor

    def border_image(self, mask):
        kernel = np.ones((6,6),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        blur = cv2.GaussianBlur(mask,(5,5),0)
        cany = cv2.Canny(blur,1,1)
        cany = cv2.dilate(cany, np.ones((3,3)), iterations=2)
        cany = cv2.erode(cany, np.ones((3,3)), iterations= 1)
        contours, hier = cv2.findContours(blur, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours, cany

    def apply_threshold(self, layer, thresh_min, thresh_max):
        _, image_binary = cv2.threshold(layer, thresh_min, thresh_max, cv2.THRESH_BINARY)
        return image_binary


    def get_mask_brightness(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        image_bin_v = self.apply_threshold(v, 100, 255)
        return image_bin_v

    def get_image_brightness(self, image):
        mask_layer_v = self.get_mask_brightness(image)
        image_no_background = cv2.bitwise_and(image, image, mask=mask_layer_v)
        _, image_no_background = self.cut_out_backgound(image_no_background)
        return image_no_background

    def draw_image(self, image, type, data, color):
        #if type = 1 is an rectangle
        if(type == 1):
            [(x_min, y_min), (x_max, y_max)] = data
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        #if type = 2 is an polygon
        elif(type == 2):
            approx = cv2.approxPolyDP(data, 0.05 * cv2.arcLength(data, True), True)
            if len(approx) == 3:
                cv2.drawContours(image, [data], 0, color, 2)
        #if type = 3 is an text
        else:
            cv2.putText(image, data,(20,20), cv2.FONT_HERSHEY_COMPLEX, 1.5, color)
        return image