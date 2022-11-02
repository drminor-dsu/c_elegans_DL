import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import load_data

tf.keras.backend.clear_session()
if tf.test.is_gpu_available('gpu'):
    print('GPU is available')


def simple_lstm(sequences, features):
    inputs = keras.Input(shape=(sequences, features))
    x = layers.LSTM(27)(inputs)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    return model


def densely_connected(sequences, features):
    inputs = keras.Input(shape=(sequences, features))
    x = layers.Flatten()(inputs)
    x = layers.Dense(200, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    return model


def binary_model_train(model, train_x, train_y, valid_x, valid_y, fname):
    callbacks = [
        keras.callbacks.ModelCheckpoint(fname, save_best_only=True)
    ]
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', 'F1'])
    history = model.fit(train_x, train_y,
                        batch_size=128,
                        epochs=20,
                        validation_data=(valid_x, valid_y),
                        callbacks=callbacks)

    return history


def model_evaluate(test_x, test_y, fname):
    model = keras.models.load_model(fname)
    results = model.evaluate(test_x, test_y)

    return results


def display(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training')
    plt.plot(epochs, val_loss, 'rx', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('../results/image.png')
    plt.show()


if __name__ == '__main__':
    target = {'Normal': 0,
              'Benzen_0_1_ppm': 1, 'Benzen_0_5_ppm': 2,
              'Formaldehyde_0_1_ppm': 3, 'Formaldehyde_0_5_ppm': 4,
              'Toluen_0_1_ppm': 5, 'Toluen_0_5_ppm': 6}

    data_set = ['Formaldehyde_0_1_ppm', 'Benzen_0_1_ppm']
    duration = 60
    fname = '_'.join(data_set) + f'_duration_{duration}.keras'
    print(fname)
    # fname = f'{fname}_duration_{duration}.keras'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_data.timeseries_dataset(data_set, duration=duration)

    model = simple_lstm(train_x.shape[1], train_x.shape[-1])
    # model = densely_connected(train_x.shape[1], train_y.shape[-1])
    history = binary_model_train(model, train_x, train_y, valid_x, valid_y, fname)
    results = model_evaluate(test_x, test_y, fname)
    print(results)
    display(history)