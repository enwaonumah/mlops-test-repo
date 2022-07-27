# Standard Library Modules
import os

# Third-Party Modules
import tensorflow as tf

# Configure Dataset for Performance
AUTOTUNE = tf.data.AUTOTUNE
class DataSplits:
	def __init__(self, data_path, batch_size=32, image_size=(160,160)):
		self.data_path = data_path
		self.batch_size = batch_size
		self.image_size = image_size

	def create_splits(self):
		self.create_training_set()
		self.create_validation_set()
		self.create_test_set()
		self.prefetch()

	def create_training_set(self):
		train_dir = os.path.join(self.data_path, 'train')
		self.train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=self.batch_size,
                                                            image_size=self.image_size)

		self.class_names = self.train_dataset.class_names

	def create_validation_set(self):
		validation_dir = os.path.join(self.data_path, 'validation')
		self.validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=self.batch_size,
                                                            	 image_size=self.image_size)

	def create_test_set(self):
		# Determine how many batches of data are asvailable in the validation set and then move 20% to the test set
		val_batches = tf.data.experimental.cardinality(self.validation_dataset)
		self.test_dataset = self.validation_dataset.take(val_batches // 5)
		self.validation_dataset = self.validation_dataset.skip(val_batches // 5)

		print(f'Number of validation batches: {tf.data.experimental.cardinality(self.validation_dataset)}')
		print(f'Number of test batches: {tf.data.experimental.cardinality(self.test_dataset)}')


	def prefetch(self):
		self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)
		self.validation_dataset = self.validation_dataset.prefetch(buffer_size=AUTOTUNE)
		self.test_dataset = self.test_dataset.prefetch(buffer_size=AUTOTUNE)

# Datasets
def load_cats_dogs_dataset():
	_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
	path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
	PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

	return PATH

