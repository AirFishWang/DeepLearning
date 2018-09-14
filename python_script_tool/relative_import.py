# -*- coding: utf-8 -*-

# https://laike9m.com/blog/pythonxiang-dui-dao-ru-ji-zhi-xiang-jie,60/
import os, sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import tf_retinanet.detector
    __package__ = "tf_retinanet.detector"