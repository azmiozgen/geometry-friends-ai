import json

import h5py
import keras
from keras import backend as K
from keras.layers import Conv2D, Dense, Flatten
from keras.losses import MeanSquaredError
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l2

import main
import params

###################
## Hyper-parameters
###################

FEATURE_MAP1 = 32
FEATURE_MAP2 = 64
FEATURE_MAP3 = 64
FEATURE_MAP4 = 32
FEATURE_MAP5 = 16
KERNEL_SIZE1 = 8
STRIDE1 = 4
KERNEL_SIZE2 = 4
STRIDE2 = 2
KERNEL_SIZE3 = 3
STRIDE3 = 1
KERNEL_SIZE4 = 4
STRIDE4 = 1
KERNEL_SIZE5 = 3
STRIDE5 = 1
FC_SIZE1 = 512
FC_SIZE2 = 256
FC_SIZE3 = 128
FC_SIZE4 = 64
# DROPOUT1 = 0.25
# DROPOUT2 = 0.50
MOMENTUM = 0.9
# DECAY = 1e-4       ## lr = self.lr * (1. / (1. + self.decay * self.iterations))
L2_REG = 0.001

## Reshape images
HEIGHT = int(params.HEIGHT * params.SHAPE_REDUCE_RATE)
WIDTH = int(params.WIDTH * params.SHAPE_REDUCE_RATE)
if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (params.INPUT_STACK, HEIGHT, WIDTH)
else:
    INPUT_SHAPE = (HEIGHT, WIDTH, params.INPUT_STACK)


def load_legacy_model(path):
    """
    Rebuild a Sequential model saved by Keras 2 (e.g. models/one.h5) and load its weights.
    Keras 3 cannot parse the old Sequential config, which is a plain list of layers.
    """
    with h5py.File(path, 'r') as f:
        layer_configs = json.loads(f.attrs['model_config'])['config']
    if isinstance(layer_configs, dict):
        layer_configs = layer_configs['layers']
    input_shape = layer_configs[0]['config']['batch_input_shape'][1:]
    legacy_model = Sequential([keras.Input(shape=input_shape)])
    for layer in layer_configs:
        config = dict(layer['config'])
        config.pop('batch_input_shape', None)
        legacy_model.add(getattr(keras.layers, layer['class_name']).from_config(config))
    legacy_model.load_weights(path)
    return legacy_model


def build_model():
    def conv(filters, kernel_size, stride, **kwargs):
        return Conv2D(filters, (kernel_size, kernel_size), strides=(stride, stride),
                      activation='relu', kernel_initializer='glorot_uniform', padding='same',
                      kernel_regularizer=l2(L2_REG), **kwargs)

    def dense(units, activation='relu'):
        return Dense(units, activation=activation, kernel_initializer='glorot_uniform',
                     kernel_regularizer=l2(L2_REG))

    new_model = Sequential()
    new_model.add(keras.Input(shape=INPUT_SHAPE))
    new_model.add(conv(FEATURE_MAP1, KERNEL_SIZE1, STRIDE1))
    new_model.add(conv(FEATURE_MAP2, KERNEL_SIZE2, STRIDE2))
    # new_model.add(Dropout(DROPOUT1))
    new_model.add(conv(FEATURE_MAP3, KERNEL_SIZE3, STRIDE3))
    new_model.add(conv(FEATURE_MAP4, KERNEL_SIZE4, STRIDE4))
    # new_model.add(conv(FEATURE_MAP5, KERNEL_SIZE5, STRIDE5))

    new_model.add(Flatten())
    new_model.add(dense(FC_SIZE1))
    new_model.add(dense(FC_SIZE2))
    new_model.add(dense(FC_SIZE3))
    new_model.add(dense(FC_SIZE4))
    new_model.add(dense(params.N_ACTIONS, activation='linear'))
    return new_model


if main.LOAD is not None:
    try:
        model = load_model(main.LOAD, compile=False)
    except (KeyError, TypeError, ValueError):
        model = load_legacy_model(main.LOAD)
    print('Model {} restored.'.format(main.LOAD.split('/')[1].rstrip('.h5')))
else:
    model = build_model()

## Keras 3 cannot restore a trainable optimizer from .h5 files, so always compile a fresh one.
# sgd = SGD(learning_rate=LR, momentum=MOMENTUM, nesterov=True)
adam = Adam() if main.LR is None else Adam(learning_rate=main.LR)
model.compile(loss=MeanSquaredError(), optimizer=adam)
