# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:    lr_callback
   Description:
   Author:       wangchun
   date:         2020/8/25
-------------------------------------------------
"""

from keras.callbacks import Callback
from keras import backend as K


class LrCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        print("lr = {}".format(K.eval(lr)))

lr_callback = LrCallback()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[loss_callback, lr_callback])
