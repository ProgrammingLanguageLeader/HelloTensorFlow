from argparse import ArgumentParser

import tensorflow as tf
import numpy as np


def parse_arguments():
    default = {
        'samples_number': 1000,
        'batch_size': 100,
        'steps_number': 20000,
        'display_step': 100
    }
    parser = ArgumentParser()
    parser.add_argument(
        '-t',
        '--samples',
        type=int,
        default=default['samples_number'],
        help='Samples number'
    )
    parser.add_argument(
        '-b',
        '--batch',
        type=int,
        default=default['batch_size'],
        help='Batch size'
    )
    parser.add_argument(
        '-s',
        '--steps',
        type=int,
        default=default['steps_number'],
        help='Steps number'
    )
    parser.add_argument(
        '-d',
        '--display',
        type=int,
        default=default['display_step'],
        help='Display step'
    )
    args = parser.parse_args()
    return {
        'samples_number': args.samples,
        'batch_size': args.batch,
        'steps_number': args.steps,
        'display_step': args.display
    }


if __name__ == '__main__':
    args = parse_arguments()
    samples_number = args['samples_number']
    batch_size = args['batch_size']
    steps_number = args['steps_number']
    display_step = args['display_step']
    learning_rate = 0.0001

    x_data = np.random.uniform(1, 10, (samples_number, 1))
    y_data = 2 * x_data + 1 + np.random.normal(0, 2, (samples_number, 1))

    x = tf.placeholder(
        tf.float32,
        shape=(batch_size, 1)
    )
    y = tf.placeholder(
        tf.float32,
        shape=(batch_size, 1)
    )

    with tf.variable_scope('linear-regression') as scope:
        k = tf.Variable(
            tf.random_normal((1, 1)),
            name='slope'
        )
        b = tf.Variable(
            tf.zeros((1, )),
            name='bias'
        )

    y_predicted = tf.matmul(x, k) + b
    loss = tf.reduce_sum((y - y_predicted) ** 2)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session() as session:
        session.run(
            tf.global_variables_initializer()
        )
        for step_number in range(steps_number):
            indices = np.random.choice(samples_number, batch_size)
            x_batch, y_batch = x_data[indices], y_data[indices]
            _, loss_val, k_val, b_val = session.run(
                [optimizer, loss, k, b],
                feed_dict={
                    x: x_batch,
                    y: y_batch
                }
            )
            if (step_number + 1) % display_step == 0:
                print(
                    'Epoch %d: loss=%.8f, k=%.4f, b=%.4f' % (
                        step_number + 1,
                        loss_val,
                        k_val,
                        b_val
                    )
                )
