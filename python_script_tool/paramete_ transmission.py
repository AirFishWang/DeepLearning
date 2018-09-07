# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     paramete_ transmission
   Description :
   Author :        wangchun
   date：          18-9-6
-------------------------------------------------
   Change Activity:
                   18-9-6:
-------------------------------------------------
"""


class Test():
    def __init__(self):
        self.x = 3


def add_list(src_list):
    for i in range(len(src_list)):
        src_list[i] += 1


def change_obj_variable(obj):
    obj.x = 4


if __name__ == "__main__":
    test_list = [1, 2, 3, 4]
    add_list(test_list)
    print test_list

    test_obj = Test()
    change_obj_variable(test_obj)
    print test_obj.x
