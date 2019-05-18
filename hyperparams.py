class Hyperparams:

	train_batch_size = 16
	val_batch_size = 16
	test_batch_size = 16

	num_train = 10000
	num_val = 4000

	num_epochs = 10
	leaky_relu_alpha = 0.1
	learning_rate = 0.0001

	img_scale_factor = 0.4

	first_layer = 256
	second_layer = 64
	third_layer = 16

	baseline = 0.01

	normalization = False

	x1 = 1.
	x2_x1 = 1.
	y1 = 1.
	y2_y1 = 1.