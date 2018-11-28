# -*- coding: utf-8 -*-
import cv2
import os
import shutil
import numpy as np
from files_walk import get_image_list

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


def threshold_image():
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold的第二个参数是阈值，第三个参数是二值化后设置的max value，如果指定了cv2.THRESH_TRIANGLE或者cv2.THRESH_OTSU，
    # 算法将不采用提供的阈值，而是使用计算出来的阈值，返回值ret为最终选取的阈值
    ret, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    # ret, binary = cv2.threshold(gray_image, 125, 255, cv2.THRESH_BINARY)
    print "ret = {}".format(ret)
    cv2.imshow("binary", binary)
    cv2.waitKey(0)


def need_convert_backgroud(gray_image):
    """
    convert a "black-groud" image to "white-ground" image according to peripheral pixel if necessary
    :param gray_image:
    :return:
    """

    ret, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    d = 3
    h, w = gray_image.shape[:2]
    boundary = np.concatenate((binary[0:d, :].flatten(),
                               binary[h-d:h, :].flatten(),
                               binary[:, 0:d].flatten(),
                               binary[:, w-d:w].flatten()), axis=0)
    white_pixel_count = np.where(boundary == 255)[0].shape[0]
    black_pixel_count = np.where(boundary == 0)[0].shape[0]
    if white_pixel_count >= black_pixel_count:
        return False
    else:
        return True


def backgroud_classify():
    input_dir = "/home/wangchun/Desktop/图片筛选/广告_select/meitu_ps_illegal_crnn"
    output_dir = input_dir + "_classify"
    white_dir = os.path.join(output_dir, "white_ground")
    black_dir = os.path.join(output_dir, "black_ground")
    for x in [white_dir, black_dir]:
        if not os.path.exists(x):
            os.makedirs(x)
    image_lists = get_image_list(input_dir)
    count = len(image_lists)
    for index, image_file in enumerate(image_lists):
        image_name = os.path.split(image_file)[1]
        image = cv2.imread(image_file)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if need_convert_backgroud(gray_image):
            shutil.copyfile(image_file, os.path.join(black_dir, image_name))
            cv2.imwrite(os.path.join(black_dir, image_name+"_gray.png"), 255-gray_image)
            print "{}/{} copy {} to black dir".format(index + 1, count, image_file)
        else:
            shutil.copyfile(image_file, os.path.join(white_dir, image_name))
            cv2.imwrite(os.path.join(white_dir, image_name + "_gray.png"), gray_image)
            print "{}/{} copy {} to white dir".format(index + 1, count, image_file)

    print "backgroud_classify finished"


def rotate_image_test():
    src_image = cv2.imread(image_path)
    h, w = src_image.shape[:2]
    center_x = w / 2
    center_y = h / 2
    angle = 10
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - center_x
    M[1, 2] += (nH / 2) - center_y
    dst_image = cv2.warpAffine(src_image, M, (nW, nH), borderValue=(255, 255, 255))
    cv2.imshow("rotate", dst_image)
    cv2.waitKey(0)


def affine_image_test():
    src_image = cv2.imread(image_path)
    h, w = src_image.shape[:2]
    pts1 = np.float32([[100, 100], [200, 100], [100, 200]])
    pts2 = np.float32([[200, 100], [300, 100], [100, 200]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst_image = cv2.warpAffine(src_image, M, (w, h), borderValue=(255, 255, 255))
    cv2.imshow("affine", dst_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    # roi_image()
    # read_image_test()
    # resize_image()
    # concatenate_image()
    # threshold_image()
    # backgroud_classify()
    # rotate_image_test()
    affine_image_test()
    pass
