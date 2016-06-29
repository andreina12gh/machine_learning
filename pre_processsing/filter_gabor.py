import numpy as np
import cv2

class FilterGabor:
    def nothing(x):
        pass

    def get_kernel(self, ks, sig, th, lm, ps):
        if not ks % 2:
            exit(1)
        theta = th * np.pi / 180.
        psi = ps * np.pi / 180.
        xs = np.linspace(-1., 1., ks)
        ys = np.linspace(-1., 1., ks)
        lmbd = np.float(lm)
        x, y = np.meshgrid(xs, ys)
        sigma = np.float(sig) / ks
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        return np.array(
            np.exp(-0.5 * (x_theta ** 2 + y_theta ** 2) / sigma ** 2) * np.cos(2. * np.pi * x_theta / lmbd + psi),
            dtype=np.float32)

    def apply_filter(self, image, ks, sig, lm, th, ps):
        kernel = self.get_kernel(ks, sig, th, lm, ps)
        image_with_filter = cv2.filter2D(image, cv2.CV_32F, kernel)
        image_with_filter = cv2.resize(image_with_filter, (640, 480))
        return image_with_filter