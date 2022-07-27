# Standard library Modules
import json
import os
import pickle
import sys 

#Third-Party Modules
import tensorflow as tf
import valohai

# Set Path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Local Modules
from deep_learning.ingestion import data_processing
from deep_learning.models import builder
from deep_learning.training import train
from deep_learning.testing import test
from utils import file_io_utils

def main():
	# Get Data and Create Splits
	print('Loading Dataset...')
	dataset_zip_path = valohai.inputs('dataset').path(process_archives=False)
	unzipped_dir = file_io_utils.unzip_archive(zip_path=dataset_zip_path)

	# Load Parameters
	batch_size = valohai.parameters('batch_size').value
	image_width = valohai.parameters('image_width').value
	image_height = valohai.parameters('image_height').value
	image_channels = valohai.parameters('image_channels').value
	learning_rate = valohai.parameters('learning_rate').value
	metrics = valohai.parameters('metrics').value
	epochs = valohai.parameters('epochs').value

	# Utilize GPUs
	strategy = builder.load_distribute_strategy()
	print(json.dumps({"Num_GPUs": strategy.num_replicas_in_sync}))

	with strategy.scope():
		# Initialize DataSplits and create the training, validation, and test splits
		data_splits = data_processing.DataSplits(data_path=unzipped_dir, 
												batch_size=batch_size, 
												image_size=(image_width, image_height))
		data_splits.create_splits()

		print('Data Creation Completed')

		# Build Model Architecture
		print('Building Model Architecture...')
		image_batch, _ = next(iter(data_splits.train_dataset))
		img_shape = (image_width, image_height, image_channels)
		model_builder = builder.MobileNetV2Model(img_shape=img_shape)
		model_builder.build_model(image_batch)

		# Compile the Model
		print('Compiling the Model...')	
		model_builder.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
					optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
					metrics=metrics)

		model_builder.report()

		# Train the fine-tune Model and Display Results
		print('Training Model...')
		print(f"epochs: {epochs}")

		history = train.train(model_builder.model,
						data_splits.train_dataset,
						data_splits.validation_dataset,
						epochs)

		# Evaluation and Prediction
		print('Testing Model...')
		test.test(model_builder, data_splits, visualize=True)

		# Save Model
		output_dir = valohai.outputs().path('model_output')
		output_zip = valohai.outputs().path('model_output.zip')
		labels = valohai.outputs().path('labels.pkl')
		training_history = valohai.outputs().path('training_history.pkl')

		print(f'Saving Model...')
		tf.keras.models.save_model(
			model=model_builder.model,
			filepath=output_dir,
			overwrite=True,
			include_optimizer=True,
			save_format=None,
			signatures=None,
			options=None,
			save_traces=True)	

		# Zip and write folder to Valohai
		file_io_utils.make_archive(source=output_dir, destination=output_zip)

		# Write out class labels
		with open(labels, 'wb') as file_pkl:
			pickle.dump(data_splits.class_names, file_pkl)

		# Write out Training metrics
		with open(training_history, 'wb') as file_pkl:
			pickle.dump(history.history, file_pkl)

		# Log to Valohai Metadata
		for idx in range(len(history.history['loss'])):
			with valohai.logger() as logger:
				print('idx', idx)
				logger.log('epoch', idx+1)
				logger.log('loss', history.history['loss'][idx])
				logger.log('accuracy', history.history['accuracy'][idx])
				logger.log('val_loss', history.history['val_loss'][idx])
				logger.log('val_accuracy', history.history['val_accuracy'][idx])

	print('Training and Testing Completed!')


if __name__ == '__main__':
	main()