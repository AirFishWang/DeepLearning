# -*- coding: utf-8 -*-
import cv2
import numpy as np

image_path = "../data/cat.jpeg"


def roi_image():
    src_image = cv2.imread(image_path)
    cv2.imshow("src_image", src_image)
    # cv2.waitKey()
    h, w = src_image.shape[:2]
    print "src_image: h = {}   w = {}".format(h, w)
    roi = src_image[:h, :w, :]  # h, w, channel
    h, w = roi.shape[:2]
    print "src_image: h = {}   w = {}".format(h, w)
    cv2.imshow("roi", roi)
    cv2.waitKey()


def resize_image():
    src_image = cv2.imread(image_path)
    dst_image = cv2.resize(src_image, (400, 300))   # (w, h)
    cv2.imshow("dst_image", dst_image)
    cv2.waitKey(0)


def read_image_test():
    image_path = "../data/rgba.png"
    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    print "cv2.IMREAD_ANYDEPTH shape = {}".format(image.shape)
    cv2.imshow("image", image)
    cv2.waitKey()

    image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    print "cv2.IMREAD_ANYCOLOR shape = {}".format(image.shape)
    cv2.imshow("image", image)
    cv2.waitKey()

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print "cv2.IMREAD_UNCHANGED shape = {}".format(image.shape)
    cv2.imshow("image", image)
    cv2.waitKey()

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print "cv2.IMREAD_GRAYSCALE shape = {}".format(image.shape)
    cv2.imshow("image", image)
    cv2.waitKey()


def concatenate_image():
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    # dst_image = np.concatenate((image, image), axis=1)
    padding_image = np.ones([h, 50, 3], dtype=np.uint8) * 255
    dst_image = np.concatenate((padding_image, image, padding_image), axis=1)
    cv2.imshow("padding_image", padding_image)
    cv2.imshow("concatenate", dst_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    # roi_image()
    # read_image_test()
    # resize_image()
    concatenate_image()