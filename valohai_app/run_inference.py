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
	# Download Files from Valohai/S3
	# Get and Extract Data Zip from S3
	model_zip_path = valohai.inputs('model').path(process_archives=False)
	label_file_path = valohai.inputs('labels').path()
	
	# Unzip and load model
	unzipped_dir = file_io_utils.unzip_archive(zip_path=model_zip_path)
	model = tf.keras.models.load_model(unzipped_dir)
	
	# Load labels
	labels = file_io_utils.load_pickle(label_file_path)

	# Create model inference object
	inf_obj = inference.ModelInference(model, labels)

	# Perform inferences
	run_inference_custom(inf_obj)


def run_inference_custom(inf_obj):
	# Load image and run inference
	results_dict = {'label_ids': [0,1], 'label_names': inf_obj.labels, 'predictions': [], 'actuals': []}

	for image_path in valohai.inputs("images").paths():
		image = file_io_utils.load_image(image_path)
		img = inf_obj.preprocess_image(image)
		pred_class, pred_label = inf_obj.run_prediction(img)

		results_dict['predictions'].append(pred_class)
		results_dict['actuals'].append(results_dict['label_names'].index(os.path.basename(image_path).split('_')[0].lower() +'s'))

		# Write to Log
		logger.info(f'Inference - Filename: {os.path.basename(image_path)}, Predicted Class: {pred_class}, Predicted Label: {pred_label}')

	
	# Confusion matrix
	matrix = confusion_matrix(results_dict['actuals'], results_dict['predictions'], labels=results_dict['label_ids'])
	
	results = []
	results.append(results_dict['label_names'])
	[results.append(x) for x in matrix.tolist()] 
	
	print(json.dumps({"inference_results": results}))
	
	TP, FN, FP, TN = confusion_matrix(results_dict['actuals'], results_dict['predictions'], labels=results_dict['label_ids']).reshape(-1)
	logger.info(f'Outcome Values: TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}')
	
	print('Inference Completed!')



if __name__ == '__main__':
	main()