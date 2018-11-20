# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     mser.py
   Description :
   Author :        wangchun
   date：          18-11-19
-------------------------------------------------
   Change Activity:
                   18-11-19:
-------------------------------------------------
   reference: https://www.jianshu.com/p/1b9c275698c9
"""
import cv2


def mser_test():
    image_file = "/home/wangchun/Desktop/图片筛选/广告_select/meitu_ps_illegal/6d5a2a9b5e02b6b4cdb5790d5da99411.jpg"
    src_image = cv2.imread(image_file)
    gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(_delta=3, _min_area=10, _max_area=1200)
    regions, boxes = mser.detectRegions(gray_image)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(src_image, hulls, 1, (0, 255, 0))
    cv2.imshow('src_image', src_image)
    cv2.waitKey(0)
    pass


if __name__ == "__main__":
    mser_test()