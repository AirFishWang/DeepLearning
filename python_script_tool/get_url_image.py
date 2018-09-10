# -*- coding: utf-8 -*-
import cv2
import numpy as np
import requests


def get_url_image():
    url = "https://developer.nvidia.com/sites/default/files/akamai/cuda/images/deeplearning/TRT3_Benchmark1.PNG"
    url_image = requests.get(url)
    image = np.asarray(bytearray(url_image.content), np.uint8)
    image = cv2.imdecode(image, 0)
    cv2.imshow("image", image)
    cv2.waitKey()

if __name__ == "__main__":
    get_url_image()