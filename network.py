import numpy as np
from keras import backend as K
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_accuracy
from keras.initializers import glorot_uniform
from keras.regularizers import l1_l2, l1, l2
from keras.preprocessing.image import ImageDataGenerator
import os
import main, params

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
# DECAY = 1e-4		 ## lr = self.lr * (1. / (1. + self.decay * self.iterations))
L2_REG = 0.001

## Reshape images
if K.image_data_format() == 'channels_first':
	INPUT_SHAPE = (INPUT_STACK, int(params.HEIGHT * params.SHAPE_REDUCE_RATE), int(params.WIDTH * params.SHAPE_REDUCE_RATE))
else:
	INPUT_SHAPE = (int(params.HEIGHT * params.SHAPE_REDUCE_RATE), int(params.WIDTH * params.SHAPE_REDUCE_RATE), params.INPUT_STACK)

if main.LOAD is not None:
	model = load_model(main.LOAD)
	print("Model {} restored.".format(main.LOAD.split("/")[1].rstrip(".h5")))
else:
	model = Sequential()
	model.add(Conv2D(FEATURE_MAP1, (KERNEL_SIZE1, KERNEL_SIZE1), subsample=(STRIDE1, STRIDE1), activation='relu', input_shape=INPUT_SHAPE,
					kernel_initializer='glorot_uniform', padding='same', kernel_regularizer=l2(L2_REG)))
	model.add(Conv2D(FEATURE_MAP2, (KERNEL_SIZE2, KERNEL_SIZE2), subsample=(STRIDE2, STRIDE2), activation='relu', kernel_initializer='glorot_uniform', \
					padding='same', kernel_regularizer=l2(L2_REG)))
	# model.add(Dropout(DROPOUT1))
	model.add(Conv2D(FEATURE_MAP3, (KERNEL_SIZE3, KERNEL_SIZE3), subsample=(STRIDE3, STRIDE3), activation='relu', kernel_initializer='glorot_uniform', \
					padding='same', kernel_regularizer=l2(L2_REG)))
	model.add(Conv2D(FEATURE_MAP4, (KERNEL_SIZE4, KERNEL_SIZE4), subsample=(STRIDE4, STRIDE4), activation='relu', kernel_initializer='glorot_uniform', \
					padding='same', kernel_regularizer=l2(L2_REG)))
	#model.add(Conv2D(FEATURE_MAP5, (KERNEL_SIZE5, KERNEL_SIZE5), subsample=(STRIDE5, STRIDE5), activation='relu', kernel_initializer='glorot_uniform', \
	#				padding='same', kernel_regularizer=l2(L2_REG)))

	model.add(Flatten())
	model.add(Dense(FC_SIZE1, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=l2(L2_REG)))
	model.add(Dense(FC_SIZE2, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=l2(L2_REG)))
	model.add(Dense(FC_SIZE3, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=l2(L2_REG)))
	model.add(Dense(FC_SIZE4, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=l2(L2_REG)))
	model.add(Dense(params.N_ACTIONS, activation='linear', kernel_initializer='glorot_uniform', kernel_regularizer=l2(L2_REG)))

	# sgd = SGD(lr=LR, momentum=MOMENTUM, decay=DECAY, nesterov=True)
	adam = Adam(lr=main.LR)
	model.compile(loss='mse', optimizer=adam)
