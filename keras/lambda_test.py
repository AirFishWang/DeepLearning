# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     lambda_test
   Description :
   Author :        wangchun
   date：          2019/4/25
-------------------------------------------------
   Change Activity:
                   2019/4/25:
-------------------------------------------------
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers import merge
from keras.utils import plot_model
from keras.layers import Input, Lambda
from keras.models import Model


def slice(x, index):
    return x[:, :, index]


a = Input(shape=(4, 2))
x1 = Lambda(slice, output_shape=(4, 1), arguments={'index': 0})(a)
x2 = Lambda(slice, output_shape=(4, 1), arguments={'index': 1})(a)
x1 = Reshape((4, 1, 1))(x1)
x2 = Reshape((4, 1, 1))(x2)
# output = merge([x1, x2], mode='concat')
# model = Model(a, output)
model = Model(a, [x1, x2])
x_test = np.array([[[1, 2], [2, 3], [3, 4], [4, 5]]])
print model.predict(x_test)
plot_model(model, to_file='lambda.png', show_shapes=True)