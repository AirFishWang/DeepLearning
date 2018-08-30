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


if __name__ == "__main__":
    roi_image()