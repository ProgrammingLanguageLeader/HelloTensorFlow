import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


def generate_centered_samples(samples_number, center_coord):
    return np.random.normal(size=[samples_number, 2]) + [*center_coord]


def get_samples_data(
        samples_number=1000,
        point_0=(-1., -1.),
        point_1=(1., 1.)
):
    zeros, ones = np.zeros((samples_number, 1)), np.ones((samples_number, 1))
    labels = np.vstack([zeros, ones])
    z_sample = generate_centered_samples(
        samples_number,
        point_0
    )
    o_sample = generate_centered_samples(
        samples_number,
        point_1
    )
    return np.vstack([z_sample, o_sample]), labels


if __name__ == '__main__':
    tf_old_logger = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)

    logr = Sequential()
    logr.add(
        Dense(1, input_dim=2, activation='sigmoid')
    )
    logr.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    x_train, y_train = get_samples_data()
    x_test, y_test = get_samples_data(100)

    logr.fit(
        x_train,
        y_train,
        batch_size=16,
        epochs=100,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    tf.logging.set_verbosity(tf_old_logger)
