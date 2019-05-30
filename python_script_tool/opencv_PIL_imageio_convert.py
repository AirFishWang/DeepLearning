# -*- coding: utf-8 -*-
import cv2
import imageio
from PIL import Image
import numpy as np
from skimage import io


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


def imageio_to_opencv():
    # image = imageio.imread("../data/cat.jpeg")
    image = imageio.imread("../data/rgba.png")
    image = np.asarray(image, np.uint8)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        pass

    cv2.imshow("OpenCV", image)
    cv2.waitKey()


def skimage_to_opencv():
    skimage_image = io.imread("../data/cat.jpeg")  # the type is numpy(uint8) mode = rgb
    opencv_image = cv2.cvtColor(skimage_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("opencv", opencv_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    # opencv_to_pil()
    pil_to_opencv()
    # imageio_to_opencv()
    # skimage_to_opencv()




