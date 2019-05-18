import pandas, logging, imageio, math
import tensorflow as tf
import numpy as np
from skimage.transform import rescale	
from hyperparams import Hyperparams

#logger configuration
FORMAT = "[%(filename)s: %(lineno)3s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

H = Hyperparams()

class Train_data_loader(tf.keras.utils.Sequence):

	def __init__(self, batch_size, num_train):
		self.num_train = num_train
		self.batch_size = batch_size
		self.training_csv = pandas.read_csv("../shuffled_training.csv")
		self.labels = np.transpose(np.vstack((self.training_csv["x1"], self.training_csv["x2"], 
			self.training_csv["y1"], self.training_csv["y2"])))[:num_train]
		self.labels = self.labels.astype(np.float64)
		logger.info("Training labels loaded of shape : {}".format(self.labels.shape))

	def __len__(self):
		return self.labels.shape[0] // self.batch_size

	def __getitem__(self, idx):
		
		img_batch = np.array([rescale(imageio.imread("../train_images/{}.png".format(i)), scale=H.img_scale_factor, 
			mode='constant', anti_aliasing='True', multichannel=True) 
			for i in range(idx*self.batch_size, (idx+1)*self.batch_size)])
		
		labels_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
		
		#converting absolute values to width and height
		labels_batch[:, 1] = labels_batch[:, 1] - labels_batch[:, 0]
		labels_batch[:, 3] = labels_batch[:, 3] - labels_batch[:, 2]

		#Normalization
		if H.normalization:
			labels_batch[:, 0] = labels_batch[:, 0] / 640
			labels_batch[:, 1] = labels_batch[:, 1] / 640
			labels_batch[:, 2] = labels_batch[:, 2] / 480
			labels_batch[:, 3] = labels_batch[:, 3] / 480

		labels_batch[:, [1, 2]] = labels_batch[:, [2, 1]]
		# logger.info("Loaded train batch number : {} with train shape : {} and label shape : {}".format(idx, img_batch.shape, labels_batch.shape))
		
		return img_batch, labels_batch

class Val_data_loader(tf.keras.utils.Sequence):

	def __init__(self, batch_size, num_train):
		self.num_train = num_train
		self.batch_size = batch_size
		self.training_csv = pandas.read_csv("../shuffled_training.csv")
		self.labels = np.transpose(np.vstack((self.training_csv["x1"], self.training_csv["x2"], 
			self.training_csv["y1"], self.training_csv["y2"])))[num_train:]
		self.labels = self.labels.astype(np.float64)
		logger.info("Val labels loaded of shape : {}".format(self.labels.shape))

	def __len__(self):
		return self.labels.shape[0] // self.batch_size

	def __getitem__(self, idx):

		img_batch = np.array([rescale(imageio.imread("../train_images/{}.png".format(i+self.num_train)), scale=H.img_scale_factor, 
			mode='constant', anti_aliasing='True', multichannel=True) 
			for i in range(idx*self.batch_size, (idx+1)*self.batch_size)])

		labels_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
		
		#converting absolute values to width and height and normalization
		labels_batch[:, 1] = labels_batch[:, 1] - labels_batch[:, 0]
		labels_batch[:, 3] = labels_batch[:, 3] - labels_batch[:, 2]

		#Normalization
		if H.normalization:
			labels_batch[:, 0] = labels_batch[:, 0] / 640
			labels_batch[:, 1] = labels_batch[:, 1] / 640
			labels_batch[:, 2] = labels_batch[:, 2] / 480
			labels_batch[:, 3] = labels_batch[:, 3] / 480
		
		labels_batch[:, [1, 2]] = labels_batch[:, [2, 1]]
		# logger.info("Loaded val batch number : {} with val shape : {} and label shape : {}".format(idx, img_batch.shape, labels_batch.shape))

		return img_batch, labels_batch

class Test_data_loader(tf.keras.utils.Sequence):

	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.test_csv = pandas.read_csv("../test.csv")
		logger.info("Test csv loaded")
		self.num_test = self.test_csv["image_name"].shape[0] // self.batch_size

	def __len__(self):
		return self.num_test

	def __getitem__(self, idx):

		img_batch = np.array([rescale(imageio.imread("../test_images/{}.png".format(i)), scale=H.img_scale_factor, 
			mode='constant', anti_aliasing='True', multichannel=True) 
			for i in range(idx*self.batch_size, min((idx+1)*self.batch_size, self.test_csv.shape[0]))])

		return img_batch
