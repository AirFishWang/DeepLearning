# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:    visual_mnist
   Description:
   Author:       wangchun
   date:         2020/4/27
-------------------------------------------------
"""
import os
import cv2
import keras
from keras.datasets import mnist

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    for i in range(y_train.shape[0]):
        label = y_train[i]
        output_dir = os.path.join("train", str(label))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img = 255 - x_train[i]
        cv2.imwrite(os.path.join(output_dir, "{:05d}.png".format(i+1)), img)

    for i in range(y_test.shape[0]):
        label = y_test[i]
        output_dir = os.path.join("test", str(label))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img = 255 - x_test[i]
        cv2.imwrite(os.path.join(output_dir, "{:05d}.png".format(i + 1 + 60000)), img)


    pass