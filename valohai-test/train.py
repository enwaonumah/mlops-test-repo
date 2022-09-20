import numpy as np
import tensorflow as tf
import valohai

input_path = {
    'default': 's3://valohai-demo-badbc3cb/data/mnist.npz'
}

valohai.prepare(
    step="train-model",
    image="tensorflow/tensorflow:2.6.0",
    default_inputs= input_path,
    default_parameters={
        'learning_rate': 0.001,
        'epoch': 5,
    },
)

def log_metadata(epoch, logs):
    with valohai.logger() as logger:
        logger.log('epoch', epoch)
        logger.log('accuracy', logs['accuracy'])
        logger.log('loss', logs['loss'])


with np.load(input_path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
 
x_train, x_test = x_train / 255.0, x_test / 255.0
 
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=valohai.parameters('learning_rate').value)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])
 
model.fit(x_train, y_train, epochs= valohai.parameters('epoch').value)

callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metadata)
model.fit(x_train, y_train, epochs=valohai.parameters('epoch').value, callbacks=[callback])

test_loss, test_accuracy = model.evaluate(x_test,  y_test, verbose=2)
 
with valohai.logger() as logger:
    logger.log('test_accuracy', test_accuracy)
    logger.log('test_loss', test_loss)

model.evaluate(x_test,  y_test, verbose=2)
 
output_path = valohai.outputs().path('model.h5')
model.save(output_path)
