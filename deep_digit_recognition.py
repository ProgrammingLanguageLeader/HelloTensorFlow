import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    image_size = 784
    digits_number = 10
    neuron_number = 100
    batch_size = 100
    steps_number = 15000

    tf_old_logger = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, [None, image_size])
    w_relu = tf.Variable(
        tf.truncated_normal([image_size, neuron_number], stddev=0.1)
    )
    b_relu = tf.Variable(
        tf.truncated_normal([neuron_number], stddev=0.1)
    )
    h = tf.nn.relu(tf.matmul(x, w_relu) + b_relu)
    keep_probability = tf.placeholder(tf.float32)
    h_drop = tf.nn.dropout(h, keep_probability)
    w = tf.Variable(tf.zeros([neuron_number, digits_number]))
    b = tf.Variable(tf.zeros([digits_number]))
    y = tf.nn.softmax(tf.matmul(h_drop, w) + b)

    y_ = tf.placeholder(tf.float32, [None, digits_number])
    logit = tf.matmul(h_drop, w) + b
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=[logit], labels=[y_])
    )
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    session = tf.Session()
    session.run(
        tf.global_variables_initializer()
    )
    for step in range(steps_number):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        session.run(
            train_step,
            feed_dict={
                x: batch_xs,
                y_: batch_ys,
                keep_probability: 0.5
            }
        )
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(
        'Accuracy: %s' % session.run(
            accuracy,
            feed_dict={
                x: mnist.test.images,
                y_: mnist.test.labels,
                keep_probability: 1.
            }
        )
    )

    tf.logging.set_verbosity(tf_old_logger)
