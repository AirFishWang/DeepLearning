# -*- coding: utf-8 -*-
import numpy as np


def compare(x, y):          # 如果 x 应该排在 y 的前面，返回 -1，如果 x 应该排在 y 的后面，返回 1。如果 x 和 y 相等，返回 0。
    x = int(x.split('.')[0])
    y = int(y.split('.')[0])
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0


def sort_test():
    file_list = ["1.png", "2.png", "4.jpg", "3.bmp"]
    print sorted(file_list, compare)    # print ['1.png', '2.png', '3.bmp', '4.jpg']


def argsort_test():
    l = [0.9, 0.4, 0.2, 0.5, 0.7]
    print np.argsort(l)               # print [2 1 3 4 0]

    l = np.array(l)
    print l[np.argsort(l)]            # print [0.2 0.4 0.5 0.7 0.9]


def sort_lambda():
    x = [[3, 2], [4, 1], [2, 5], [0, 4]]
    print sorted(x, key=lambda x: x[0])
    print sorted(x, key=lambda x: x[1])
    print sorted(x, key=lambda x: x[1]*x[0])

if __name__ == "__main__":
    # sort_test()
    # argsort_test()
    sort_lambda()
