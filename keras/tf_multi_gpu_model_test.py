# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.python.keras.applications import Xception
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
import numpy as np


class RedirectModel(tf.keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.

    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```

    Args
        callback : callback to wrap.
        model    : model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)


num_samples = 1000
height = 224
width = 224
num_classes = 1000

# Instantiate the base model (or "template" model).
# We recommend doing this with under a CPU device scope,
# so that the model's weights are hosted on CPU memory.
# Otherwise they may end up hosted on a GPU, which would
# complicate weight sharing.
with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3),
                     classes=num_classes)
    model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')
    print "optimizer = {}".format(model.optimizer)            # optimizer = None
    print model.optimizer.get_config()

# Replicates the model on 8 GPUs.
# This assumes that your machine has 8 available GPUs.

parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


print "optimizer = {}".format(hasattr(model, 'optimizer'))
print model.optimizer.get_config()

# Generate dummy data.
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))


checkpointer = ModelCheckpoint(
            "tf_weights.{epoch:02d}.h5",
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            monitor="val_loss",
            mode='min'
        )
checkpointer = RedirectModel(checkpointer, model)

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=3, batch_size=64, callbacks=[checkpointer])

print "optimizer = {}".format(model.optimizer)
print model.optimizer.get_config()
# Save model via the template model (which shares the same weights):
model.save('my_model.h5')