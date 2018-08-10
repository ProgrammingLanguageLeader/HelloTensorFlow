import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    tf_old_logger = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)

    image_size = 784
    digits_number = 10
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, [None, image_size])
    W = tf.Variable(tf.zeros([image_size, digits_number]))
    b = tf.Variable(tf.zeros([digits_number]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, digits_number])
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
    )
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    session = tf.Session()
    session.run(
        tf.global_variables_initializer()
    )
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        session.run(
            train_step,
            feed_dict={
                x: batch_xs,
                y_: batch_ys
            }
        )
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(
        'Accuracy: %s' % session.run(
            accuracy,
            feed_dict={
                x: mnist.test.images,
                y_: mnist.test.labels
            }
        )
    )

    tf.logging.set_verbosity(tf_old_logger)
