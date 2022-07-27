# Standard library Modules
import json
import os
import sys

#Third-Party Modules
import tensorflow as tf
import valohai
from sklearn.metrics import confusion_matrix

# Set Path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Local Modules
from deep_learning.inferencing import inference
from utils import file_io_utils

# Initialize Logger
logger = file_io_utils.load_logger(script_name=__name__)

def main():
	# Load Parameters
	batch_size = valohai.parameters('batch_size').value
	image_width = valohai.parameters('image_width').value
	image_height = valohai.parameters('image_height').value
	image_channels = valohai.parameters('image_channels').value
	learning_rate = valohai.parameters('learning_rate').value
	metrics = valohai.parameters('metrics').value
	epochs = valohai.parameters('epochs').value

	print('------------------')
	print(batch_size)
	print(image_width)
	print(image_height)
	print(image_channels)
	print(learning_rate)
	print(metrics)
	print(epochs)


	lol=5

if __name__ == '__main__':
	main()