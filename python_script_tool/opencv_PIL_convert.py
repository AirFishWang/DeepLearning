# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np


def opencv_to_pil():
    img = cv2.imread("../data/cat.jpeg")
    cv2.imshow("OpenCV", img)
    cv2.waitKey()

    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    image.show()


def pil_to_opencv():
    image = Image.open("../data/cat.jpeg")
    image.show()
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    cv2.imshow("OpenCV", img)
    cv2.waitKey()


if __name__ == "__main__":
    opencv_to_pil()
    pil_to_opencv()



