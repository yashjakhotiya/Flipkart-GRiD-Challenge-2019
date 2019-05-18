import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, ZeroPadding2D, DepthwiseConv2D, Dense, Flatten, LeakyReLU, MaxPooling2D
from tensorflow.keras import regularizers
from hyperparams import Hyperparams
H = Hyperparams()

class Model(object):

	def __init__(self):
		return None

	def get_model(self):
		
		model = Sequential()
		model.add(Conv2D(32, kernel_size=3, 
					 input_shape=(192, 256, 3), data_format='channels_last'))
		model.add(LeakyReLU(0.2))
		
		model.add(Conv2D(64, kernel_size=5,
					  data_format='channels_last'))
		model.add(LeakyReLU(0.2))
		model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1),
						   data_format='channels_last'))
		
		model.add(Conv2D(64, kernel_size=7, strides=(2, 2),
					  data_format='channels_last'))
		model.add(LeakyReLU(0.2))
		model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
		
		model.add(Conv2D(128, kernel_size=9, strides=(2, 2),
					  data_format='channels_last'))
		model.add(LeakyReLU(0.2))
		model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
		
		model.add(Conv2D(256, kernel_size=7, strides=(3, 3), 
					 data_format='channels_last',kernel_regularizer=regularizers.l2(0.001)))
		model.add(LeakyReLU(0.2))
		
		model.add(Flatten())
		
		model.add(Dense(512, kernel_regularizer=regularizers.l2(0.001)))
		model.add(LeakyReLU(0.2))	
		
		model.add(Dense(256, kernel_regularizer=regularizers.l2(0.001)))
		model.add(LeakyReLU(0.1))
		
		model.add(Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
		#model.add(LeakyReLU(0.1))
		
		model.add(Dense(64,  kernel_regularizer=regularizers.l2(0.001), activation='relu'))
		#model.add(LeakyReLU(0.2))
		
		model.add(Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
		#model.add(LeakyReLU(0.2))
		
		model.add(Dense(4))
		
		return model