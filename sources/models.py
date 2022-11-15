import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import load_data

short_cut = {'Formaldehyde_0_1_ppm': 'Form_01',
			 'Formaldehyde_0_5_ppm': 'Form_05',
			 'Normal': 'Normal',
			 'Benzen_0_1_ppm': 'Benzen_01',
			 'Benzen_0_5_ppm': 'Benzen_05',
			 'Toluen_0_1_ppm': 'Toluen_01',
			 'Toluen_0_5_ppm': 'Toluen_05'}

target = {'Normal': 0,
		  'Benzen_0_1_ppm': 1, 'Benzen_0_5_ppm': 2,
		  'Formaldehyde_0_1_ppm': 3, 'Formaldehyde_0_5_ppm': 4,
		  'Toluen_0_1_ppm': 5, 'Toluen_0_5_ppm': 6}


def densely_connected(sequences, features, ncategory=2):
	inputs = tf.keras.Input(shape=(sequences, features))
	x = tf.keras.layers.Flatten()(inputs)
	x = tf.keras.layers.Dense(200, activation='relu')(x)
	if ncategory == 2:
		outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	else:
		outputs = tf.keras.layers.Dense(ncategory, activation='softmax')(x)
	model = tf.keras.Model(inputs, outputs)

	return model


def conv_1d(sequences, features, ncategory=2):
	inputs = tf.keras.Input(shape=(sequences, features))
	x = tf.keras.layers.Conv1D(10, int(sequences * 0.2), activation='relu')(inputs)
	x = tf.keras.layers.MaxPooling1D(2)(x)
	x = tf.keras.layers.Conv1D(10, int(sequences * 0.1), activation='relu')(x)
	x = tf.keras.layers.MaxPooling1D(2)(x)
	x = tf.keras.layers.Conv1D(10, int(sequences * 0.05), activation='relu')(x)
	x = tf.keras.layers.GlobalAveragePooling1D()(x)
	if ncategory == 2:
		outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	else:
		outputs = tf.keras.layers.Dense(ncategory, activation='softmax')(x)
	model = tf.keras.Model(inputs, outputs)

	return model


def simple_lstm(sequences, features, ncategory=2):
	inputs = tf.keras.Input(shape=(sequences, features))
	x = tf.keras.layers.LSTM(27)(inputs)
	if ncategory == 2:
		outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	else:
		outputs = tf.keras.layers.Dense(ncategory, activation='softmax')(x)
	model = tf.keras.Model(inputs, outputs)

	return model


def lstm(sequences, features, ncategory=2):
	inputs = tf.keras.Input(shape=(sequences, features))
	x = tf.keras.layers.LSTM(27, return_sequences=True)(inputs)
	x = tf.keras.layers.LSTM(27)(x)
	if ncategory == 2:
		outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	else:
		outputs = tf.keras.layers.Dense(ncategory, activation='softmax')(x)
	model = tf.keras.Model(inputs, outputs)

	return model


def gru(sequences, features, ncategory=2):
	inputs = tf.keras.Input(shape=(sequences, features))
	x = tf.keras.layers.GRU(27, return_sequences=True)(inputs)
	x = tf.keras.layers.GRU(27)(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	if ncategory == 2:
		outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	else:
		outputs = tf.keras.layers.Dense(ncategory, activation='softmax')(x)
	model = tf.keras.Model(inputs, outputs)

	return model


def bidirectional_lstm(sequences, features, ncategory=2):
	inputs = tf.keras.Input(shape=(sequences, features))
	x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(27, return_sequences=True))(inputs)
	x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(27))(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	if ncategory == 2:
		outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	else:
		outputs = tf.keras.layers.Dense(ncategory, activation='softmax')(x)
	model = tf.keras.Model(inputs, outputs)

	return model



def model_train(model, train_x, train_y, valid_x, valid_y, fname, epochs=30, ncategory=2):
	# remove existing model
	if os.path.exists(os.path.join(fname)):
		print(f'remove previous model: {fname}')
		os.remove(os.path.join(fname))

	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(fname, save_best_only=True)
	]

	if ncategory == 2:
		model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics='accuracy')
	else:
		model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics='accuracy')
	history = model.fit(train_x, train_y,
						batch_size=128,
						epochs=epochs,
						validation_data=(valid_x, valid_y),
						callbacks=callbacks)

	return history


def model_evaluate(test_x, test_y, fname):
	model = tf.keras.models.load_model(fname)
	results = model.evaluate(test_x, test_y)

	fbody = fname.split('.')[0]
	with open('../results/' + fbody + '.txt', 'w') as fd:
		fd.write(f"{str(results)}\n")

	return results


def display(history, fname):
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(loss) + 1)
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training')
	plt.plot(epochs, val_loss, 'rx-', label='Validation')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Loss')
	plt.legend()

	accuracy = history.history['accuracy']
	val_accuracy = history.history['val_accuracy']
	fbody = fname.split('.')[0]
	print(accuracy, type(accuracy))
	print(val_accuracy, type(val_accuracy))
	with open('../results/' + fbody + '.txt', 'a') as fd:
		fd.write(f"{str(accuracy)}\n")
		fd.write(f"{str(val_accuracy)}")

	plt.savefig('../results/' + fbody + '.png')
	plt.show()


def do_experiment(data_gen, models, duration, epochs, ncategory, scaling=True):

	# Data set Selection
	train_x, train_y, valid_x, valid_y, test_x, test_y = data_gen(data_set, duration=duration, scaling=scaling)

	# Model Selection
	model = models(train_x.shape[1], train_x.shape[-1], ncategory=ncategory)

	d_names = [short_cut[name] for name in data_set]
	scale = 'scale_1' if scaling else 'scale_0'
	fname = '_'.join(d_names) + f'_du_{duration}_ep_{epochs}_{models.__name__}_{data_gen.__name__.split("_")[0]}_{scale}.h5'
	print(fname)

	# Model training
	history = model_train(model, train_x, train_y, valid_x, valid_y, fname, epochs=epochs, ncategory=ncategory)
	results = model_evaluate(test_x, test_y, fname)

	print(results)
	display(history, fname)


if __name__ == '__main__':
	tf.keras.backend.clear_session()
	if tf.test.is_gpu_available('gpu'):
		print('GPU is available')

	data_set = ['Normal', 'Formaldehyde_0_1_ppm']
	duration = 10

	models = {0: simple_lstm,
			  1: densely_connected,
			  2: conv_1d,
			  3: lstm,
			  4: gru,
			  5: bidirectional_lstm}

	data_gen = {0: load_data.som_timeseries_dataset,
				1: load_data.profile_timeseries_dataset}

	do_experiment(data_gen[1], models[1], duration=duration, epochs=2, ncategory=len(data_set), scaling=True)