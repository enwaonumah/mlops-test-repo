# Standard Library Modules
import os
import sys

# Third-Party Modules
import numpy as np
import tensorflow as tf

# Local Modules
from utils import file_io_utils

# Initialize Logger
logger = file_io_utils.load_logger(script_name=__name__)

class ModelInference:
	def __init__(self, model, labels):
		logger.info('Loading Model...')
		self.model = model
		self.labels = labels
	
	def preprocess_image(self, image):
		# Normalization is not needed. Occurs within model.
		img = image.resize(self.model.input_shape[1:3])
		img = np.expand_dims(img, axis=0)

		return img

	def run_prediction(self, image):
		predictions = self.model.predict(image)
	
		# Apply a sigmoid since our model returns logits
		predictions = tf.nn.sigmoid(predictions)
		predictions = tf.where(predictions < 0.5, 0, 1)
		predicted_class = int(predictions)

		# Get Label
		predicted_label = self.labels[predicted_class]
		
		return predicted_class, predicted_label