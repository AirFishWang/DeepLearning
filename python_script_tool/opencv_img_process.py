# -*- coding: utf-8 -*-
import cv2

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


if __name__ == "__main__":
    # roi_image()
    read_image_test()