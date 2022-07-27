# Third-Party Modules
import tensorflow as tf

# Local Modules - Utils
import utils.file_io_utils as file_io_utils

# Initialize Logger
logger = file_io_utils.load_logger(script_name=__name__)

def test(builder, data_splits, visualize=True):
	# Evaluation and Prediction
	loss, accuracy = builder.model.evaluate(data_splits.test_dataset)
	logger.info(f'Test accuracy: {accuracy}')
	
	# Make Predictions
	# Retrieve a batch of images from the test set
	image_batch, label_batch = data_splits.test_dataset.as_numpy_iterator().next()
	predictions = builder.model.predict_on_batch(image_batch).flatten()

	# Apply a sigmoid since our model returns logits
	predictions = tf.nn.sigmoid(predictions)
	predictions = tf.where(predictions < 0.5, 0, 1)

	logger.info(f'Predictions: {predictions.numpy()}')
	logger.info(f'Labels: {label_batch}')