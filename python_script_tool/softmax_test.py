# -*- coding: utf-8 -*-
# 参考： https://segmentfault.com/a/1190000010039529

import numpy as np


def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


if __name__ == "__main__":
    x = np.array([1, 2, 3])
    result = softmax(x)
    print result
    print result.sum()