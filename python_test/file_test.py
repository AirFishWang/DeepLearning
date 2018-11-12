# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     file_test.py
   Description :
   Author :        wangchun
   date：          18-11-8
-------------------------------------------------
   Change Activity:
                   18-11-8:
-------------------------------------------------
"""

if __name__ == "__main__":
    # with open("test.txt", "w") as f:
    #     f.write("hello world")
    #     f.write("hello world")
    #     # f.flush()
    #     f.write("hello world")
    #     f.write("hello world")
    f = open("test.txt", "w")
    f.write("hello world")
    f.write("hello world")
    # f.flush()
    f.write("hello world")
    f.write("hello world")
    f.close()