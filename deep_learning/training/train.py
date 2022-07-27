# Local Modules - Utils
import utils.file_io_utils as file_io_utils

# Initialize Logger
logger = file_io_utils.load_logger(script_name=__name__)


def train(model, train_dataset, validation_dataset, epochs=10):
	loss0, accuracy0 = model.evaluate(validation_dataset)

	logger.info(f'initial loss: {loss0:.2f}')
	logger.info(f'initial accuracy: {accuracy0:.2f}')

	history = model.fit(train_dataset,
						epochs=epochs,
						validation_data=validation_dataset)

	return history