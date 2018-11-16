# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     child_fun_test
   Description :
   Author :        wangchun
   date：          18-11-15
-------------------------------------------------
   Change Activity:
                   18-11-15:
-------------------------------------------------
"""

"""
    reference : https://blog.csdn.net/celte/article/details/38982673
"""

def father_fun():
    x= [10]
    def child_fun():

        print "in child_fun x = {}".format(x[0])
        x[0] = 20

    child_fun()
    print "in father_fun x = ", x[0]

if __name__ == "__main__":
    father_fun()