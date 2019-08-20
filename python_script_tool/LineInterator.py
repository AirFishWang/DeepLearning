# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     LineInterator
   Description :   a python implementation of cv::LineInterator(c++)
   Author :        wangchun
   date：          2019/8/16
-------------------------------------------------
   Change Activity:
                   2019/8/16
-------------------------------------------------
"""
import cv2
import time
import random
import numpy as np


class LineInterator(object):
    def __init__(self, point_1, point_2, connectivity=8, left_to_right=True):
        assert connectivity == 8 or connectivity == 4

        x1, y1 = self.point_1 = point_1
        x2, y2 = self.point_2 = point_2
        bt_pix = 1
        width = max(x1, x2) + 1
        istep = width
        self.step = width

        dx = x2-x1
        dy = y2-y1
        s = -1 if dx < 0 else 0
        if left_to_right:
            dx = (dx ^ s) - s            # 异或-1 等价于相反数减1, 这里异或s 再减去s, 等价于取反
            dy = (dy ^ s) - s

            x1 ^= (x1 ^ x2) & s          # 与-1, 值保持不变, 与0等于0, 当s==0时, x1和y1值不变, 当s==0时， x1 = x2, y1 = y2
            y1 ^= (y1 ^ y2) & s
        else:
            dx = (dx ^ s) - s

        self.ptr = y1 * istep + x1 * bt_pix

        s = -1 if dy < 0 else 0
        dy = (dy ^ s) - s
        istep = (istep ^ s) - s          # 依据dy确定istep的符号, dy>0, istep>0,否则istep<0

        s = -1 if dy > dx else 0

        # conditional swaps              # 交换值
        dx ^= dy & s
        dy ^= dx & s
        dx ^= dy & s

        bt_pix ^= istep & s
        istep ^= bt_pix & s
        bt_pix ^= istep & s

        if connectivity == 8:
            assert (dx >= 0 and dy >= 0)
            self.err = dx - (dy + dy)
            self.plus_delta = dx + dx
            self.minus_delta = -(dy + dy)
            self.plus_step = int(istep)
            self.minus_step = int(bt_pix)
            self.count = dx + 1
        else:  # connectivity == 4
            assert (dx >= 0 and dy >= 0)
            self.err = 0
            self.plus_delta = (dx + dx) + (dy + dy)
            self.minus_delta = -(dy + dy)
            self.plus_step = int(istep - bt_pix)
            self.minus_step = int(bt_pix)
            self.count = dx + dy + 1

    def get_all_pointes(self):
        all_points = []
        for i in range(self.count):
            py = int(self.ptr / self.step)
            px = int(self.ptr - py * self.step)
            all_points.append((px, py))

            mask = -1 if self.err < 0 else 0
            self.err += self.minus_delta + (self.plus_delta & mask)
            self.ptr += self.minus_step + (self.plus_step & mask)

        if all_points[0] == self.point_2:
            all_points = all_points[::-1]
        return all_points


def get_points(pt1, pt2, connectivity=8):
    it = LineInterator(pt1, pt2, connectivity=connectivity)
    points = it.get_all_pointes()
    return points


def line_test():
    h = w = 100

    for i in range(1):
        img_1 = np.ones((h, w), dtype=np.uint8) * 255
        img_2 = img_1.copy()
        pt1 = (random.randint(0, w - 1), random.randint(0, h - 1))
        pt2 = (random.randint(0, w - 1), random.randint(0, h - 1))

        pt1 = (0, 0)
        pt2 = (3, 10)

        cv2.line(img_1, pt1, pt2, 0, 1)

        points = np.array(get_points(pt1, pt2, connectivity=4))
        index = (points[:, 1], points[:, 0])
        img_2[index] = 0

        print (img_1 == img_2).all()
        cv2.imshow("img_1", img_1)
        cv2.imshow("img_2", img_2)
        cv2.imwrite("img_1.png", img_1)
        cv2.imwrite("img_2.png", img_2)
        cv2.waitKey(0)


if __name__ == "__main__":
    pt1 = (1, 0)
    pt2 = (10, 0)
    start = time.time()
    for i in range(10000):
        it = LineInterator(pt1, pt2, connectivity=4)
        points = it.get_all_pointes()
    end = time.time()
    print "cost time = {}".format(end - start)
    print points

    it = LineInterator(pt2, pt1)
    points2 = it.get_all_pointes()
    print points2

    # line_test()



