import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

print(f"=======GPU======\n {gpus}")
# More explicit GPU configuration

from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np

from ard_cnn import ARDCNN
from data_ros import DataSet

BATCH = 8
EPS = 1e-8


def weighted_cross_entropy(mask):
    def loss(mask, pred):
        weight = mask * 19 + 1
        cross_entropy = (mask * K.log(pred + EPS) + \
            (1 - mask) * K.log(1 - pred + EPS)) * weight
        return -K.mean(cross_entropy)
    return loss


if __name__ == '__main__':
    x_path = '/workspace/dataset/rain_train/'
    y_path = '/workspace/dataset/rain_train/'
    
    # Get the dataset
    dataset = DataSet(x_path, y_path, batch_size=BATCH, mode='ard-cnn')()
    
    # Create model with proper input shape
    # Use a fixed input shape based on your cropped images (200x200 from _map function)
    rain_input = keras.Input(shape=(200, 200, 3), name='rain')
    
    ard_cnn = ARDCNN(rain_input)
    model = keras.Model(rain_input, ard_cnn.outputs)

    optimizer = tf.keras.optimizers.Adam(0.0001)
    loss = 'binary_crossentropy'
    #loss = weighted_cross_entropy(labels)
    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

    callbacks = [keras.callbacks.TensorBoard(log_dir='../log/bezier_adv/'),
                 keras.callbacks.ModelCheckpoint('../model/ard.{epoch:02d}_{loss:.5f}.hdf5',
                                                 'acc',
                                                 save_best_only=False,
                                                 mode='max')]

    # Use the dataset directly with model.fit()
    model.fit(dataset, epochs=40, callbacks=callbacks)
