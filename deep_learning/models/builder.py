# Third-Party Modules
import tensorflow as tf

# Local Modules - Utils
import utils.file_io_utils as file_io_utils

# Initialize Logger
logger = file_io_utils.load_logger(script_name=__name__)

class MobileNetV2Model:
	def __init__(self, img_shape=(160,160,3)):
			self.img_shape = img_shape

			# Data Augmentation
			self.data_augmentation = data_augmentation()

			# Rescale Pixel Values (Normalization)
			self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

	def build_model(self, image_batch):
		# Create the fine-tuning model from the pre-trained model MobileNet V2
		self.base_model = tf.keras.applications.MobileNetV2(
			input_shape=self.img_shape,
			include_top=False,
			weights='imagenet')

		# Feature Extractor
		feature_batch = self.base_model(image_batch)
		print(feature_batch.shape)

		# Freeze the convolutional base
		self.base_model.trainable = False

		# Print base model architecture
		self.base_model.summary()

		# Add a Classification Head
		global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
		feature_batch_average = global_average_layer(feature_batch)

		logger.info(feature_batch_average.shape)

		# Apply a Dense layer to conver these features into a single prediction per image
		prediction_layer = tf.keras.layers.Dense(1)
		prediction_batch = prediction_layer(feature_batch_average)

		logger.info(prediction_batch.shape)

		# Build the model by chaining together the different parts (from above)
		inputs = tf.keras.Input(shape=self.img_shape)
		x = self.data_augmentation(inputs)
		x = self.preprocess_input(x)
		x = self.base_model(x, training=False)
		x = global_average_layer(x)
		x = tf.keras.layers.Dropout(0.2)(x)
		outputs = prediction_layer(x)
		self.model = tf.keras.Model(inputs, outputs)

		# Now setup a model for fine-tuning
		# Fine-tune from this layer onwards
		fine_tune_at=100
		
		# Unfreezing the top layers of the model
		self.base_model.trainable = True

		# Let's take a look to see how many layers are in the base model
		logger.info(f'Number of layers in the base model: {len(self.base_model.layers)}')

		# Freeze all the layers before the `fine_tune_at` layer
		for layer in self.base_model.layers[:fine_tune_at]:
			layer.trainable = False



	def compile_model(self, base_learning_rate=0.0001, metrics=['accuracy']):
		self.model.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=metrics)

		# Number of Trainable Parameters
		len(self.model.trainable_variables)

	
	def report(self):
		self.model.summary()

		# Number of Trainable Parameters
		len(self.model.trainable_variables)

# Helper Functions
def data_augmentation(flip_dir='horizontal', rotation=0.2):
	return tf.keras.Sequential([tf.keras.layers.RandomFlip(flip_dir),
								tf.keras.layers.RandomRotation(rotation),
								])


def load_distribute_strategy():
	print('Loading distribute strategy...')
	# Uncomment once TPUs are available
	# try:
	# 	# TPUStrategy for distributed model_building
	# 	tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
	# 	tf.config.experimental_connect_to_cluster(tpu)
	# 	tf.tpu.experimental.initialize_tpu_system(tpu)
	# 	strategy = tf.distribute.experimental.TPUStrategy(tpu)
	# 	print('Using TPU')
	# except ValueError:

	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

	if len(physical_devices) > 0:
		for device in physical_devices:
			tf.config.experimental.set_memory_growth(device, True)

		# For GPU or multi-GPU machines
		strategy = tf.distribute.MirroredStrategy()
		print('Using GPU')
	else:
		strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU

	# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines

	print("Number of devices: {}".format(strategy.num_replicas_in_sync))
	return strategy