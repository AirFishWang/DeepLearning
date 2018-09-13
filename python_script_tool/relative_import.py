# -*- coding: utf-8 -*-
import os, sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import tf_retinanet.detector
    __package__ = "tf_retinanet.detector"