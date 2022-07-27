# Standard library Modules
import os

# Third-Party Modules
import matplotlib.pyplot as plt

# Local Modules - Utils
import utils.file_io_utils as file_io_utils

# Initialize Logger
logger = file_io_utils.load_logger(script_name=__name__)

def display_images_with_predictions(image_batch, predictions, class_names, output_filepath):
	plt.figure(figsize=(10, 10))
	for i in range(9):
		ax = plt.subplot(3, 3, i + 1)
		plt.imshow(image_batch[i].astype("uint8"))
		plt.title(class_names[predictions[i]])
		plt.axis("off")

	logger.info('Saving image...')
	plt.savefig(output_filepath)
	logger.info('Image Saved!')

