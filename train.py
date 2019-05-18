import tensorflow as tf 
from hyperparams import Hyperparams
from data_loader import Train_data_loader, Val_data_loader, Test_data_loader
from model import Model
import logging, pandas
import numpy as np

#logger configuration
FORMAT = "[%(filename)s: %(lineno)3s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

H = Hyperparams()

train_batch_generator = Train_data_loader(H.train_batch_size, H.num_train)
val_batch_generator = Val_data_loader(H.val_batch_size, H.num_train)
test_batch_generator = Test_data_loader(H.test_batch_size)
logger.info("Generators instantiated")

model = Model().get_model()
logger.info("Model loaded")

adam = tf.keras.optimizers.Adam(lr=H.learning_rate)
model.compile(optimizer=adam, loss='mean_squared_error')
logger.info("Model compiled")

baseline_loss = H.baseline
logger.info("Beginning training")
num_batch = H.num_train//H.train_batch_size
shuffled_batch = np.array([np.random.choice(num_batch, size=(num_batch), replace=False) for _ in range(H.num_epochs)])
loss = np.zeros(shape=(num_batch))
for epoch in range(H.num_epochs):
	for batch_idx in shuffled_batch[epoch]:
		img_batch, labels_batch = train_batch_generator[batch_idx]
		loss[batch_idx] = model.train_on_batch(img_batch, labels_batch, class_weight={0:H.x1, 1:H.x2_x1, 2:H.y1, 3:H.y2_y1})
		logger.info("Epoch : {}, Step : {}, Loss : {}".format(epoch, batch_idx, loss[batch_idx]))
		if loss[batch_idx] < baseline_loss:
			model.save_weights("saved_weights/model_{}.h5".format(np.rint(loss[batch_idx])))
			baseline_loss = loss[batch_idx]
			logger.info("New best selected - {}".format(loss))
	model.save_weights("saved_weights/model_epoch_{}.h5".format(epoch))
	logger.info("Model weights - model_epoch_{} saved".format(epoch))
	val_loss = model.evaluate_generator(generator=val_batch_generator, steps=H.num_val//H.val_batch_size, 
		max_queue_size=3, use_multiprocessing=False, verbose=2)
	avg_train_loss = np.mean(loss)
	logger.info("Avg Train Loss for Epoch : {} is {}".format(epoch, avg_train_loss))
	logger.info("Validation - Epoch : {}, Val_Loss : {}".format(epoch, val_loss))
