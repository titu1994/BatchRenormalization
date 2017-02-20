import numpy as np
import json

import keras.callbacks as callbacks
from keras.datasets import cifar10
import keras.utils.np_utils as kutils
from keras import backend as K

from wrn_batchnorm import WideResidualNetwork

batch_size = 32
nb_epoch = 100
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
trainX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)

model = WideResidualNetwork(depth=16, width=4, weights=None, classes=10) # ordinary WRN with Batch Normalization

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.load_weights('weights/Batchnorm Weights.h5')

# history = model.fit(trainX, trainY, batch_size, nb_epoch=nb_epoch,
#                     callbacks=[
#                         callbacks.ModelCheckpoint("weights/Batchnorm Weights.h5", monitor="val_acc", save_best_only=True,
#                                                   save_weights_only=True)],
#                     validation_data=(testX, testY))
#
# with open('history/batchnorm_history.txt', 'w') as f:
#     json.dump(history.history, f)

scores = model.evaluate(testX, testY, batch_size)
print("Test loss : %0.5f" % (scores[0]))
print("Test accuracy = %0.5f" % (scores[1]))


