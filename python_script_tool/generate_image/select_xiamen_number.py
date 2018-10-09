# -*- coding: utf-8 -*-
"""
    由于厦门房地产图片中的数字具有两种字体(数字是arial, 逗号又是宋体， 所以不好直接生成图片)
    此脚本用于扣0到9图片，然后将这些图片拼接出文本行(拼接部分的代码在gen_image.py里面)

    据统计:
    对于0到9的字符宽度为5，高度为9， 左右留白为1， 逗号左不留白，右留4， 因为有逗号的存在， 文本行上留白2， 下留白4
"""
import cv2, os
import numpy as np

roi_dicts = {'0': [83, 5, 89, 19],
             '1': [90, 5, 96, 19],
             '2': [76, 5, 82, 19],
             '3': [160, 5, 166, 19],
             '4': [96, 93, 102, 107],
             '5': [146, 153, 152, 167],
             '6': [110, 93, 116, 107],
             '7': [124, 93, 130, 107],
             '8': [117, 93, 123, 107],
             '9': [132, 153, 138, 167],
             ',': [103, 93,  109, 107]}


def gen_char_roi():
    image_file = "./binary.png"
    image = cv2.imread(image_file)
    out_dir = "./xiamen_char"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for key, value in roi_dicts.items():
        x1, y1, x2, y2 = value
        roi = image[y1:y2+1, x1:x2+1, :]
        cv2.imwrite(os.path.join(out_dir, key+".png"), roi)


def binary_image():
    image_file = "./xiamen.png"
    gray_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    gray_image = 255 - gray_image
    ret, binary = cv2.threshold(gray_image, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    print binary.shape
    cv2.imwrite("binary.png", binary)
    cv2.imshow("binary", binary)
    cv2.waitKey(0)


def textline_seg():
    output_dir = "./xiamen_textline"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def find_right_boundary(img):
        height, width = img.shape[:2]
        v_sum = np.sum(img, axis=0)              # 垂直投影，竖直方向上累加
        right = width - 1
        for i in range(width-1, -1, -1):
            if v_sum[i] > 0:
                right = i
                break

        return right

    input_image = "./xiamen.png"
    gray_image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    ret, binary = cv2.threshold(gray_image, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)  # 黑底白字
    cv2.imshow("binary", binary)
    cv2.waitKey(0)

    rects = [[89, 35, 196, 49],
             [89, 63, 196, 77],
             [89, 93, 196, 107],
             [118, 125, 196, 139],
             [118, 153, 196, 167],
             [89, 180, 196, 194],
             [118, 207, 196, 221]]
    index = 0
    for rect in rects:
        x1, y1, x2, y2 = rect
        roi = binary[y1:y2+1, x1:x2+1]
        right = find_right_boundary(roi)
        text_roi = roi[:, 0:right+2]
        text_roi = 255 - text_roi
        index += 1
        cv2.imwrite(os.path.join(output_dir, "{}.png".format(index)), text_roi)

if __name__ == "__main__":
    # binary_image()
    # gen_char_roi()
    textline_seg()
