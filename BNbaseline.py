import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def bn_norm(x, out_size, phase_train):
    beta = tf.Variable(tf.constant(0.0, shape=[out_size]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[out_size]), name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')


tf.set_random_seed(0)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
phase_train = tf.placeholder(tf.bool)
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 20])
b_conv1 = bias_variable([20])
h_conv1 = bn_norm(conv2d(x_image, W_conv1) + b_conv1, 20, phase_train)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 20, 50])
b_conv2 = bias_variable([50])
h_conv2 = bn_norm(conv2d(h_pool1, W_conv2) + b_conv2, 50, phase_train)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([4, 4, 50, 500])
b_conv3 = bias_variable([500])
h_conv3 = tf.nn.relu(bn_norm(conv2d(h_pool2, W_conv3) + b_conv3, 500, phase_train))

W_fc2 = weight_variable([500, 10])
b_fc2 = bias_variable([10])
h_pool2_flat = tf.reshape(h_conv3, [-1, 500])
y_conv = tf.matmul(h_pool2_flat, W_fc2) + b_fc2

# loss = tf.reduce_mean(tf.square(y_conv - y_))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

mnist = input_data.read_data_sets('MNIST_data', one_hot=True, validation_size=50000)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epochs in range(20):
    for i in range(200):
        batch = mnist.train.next_batch(50)
        train_data = batch[0]
        train_label = batch[1]
        if i % 500 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: train_data, y_: train_label, phase_train: False})
            print("epochs %d, step %d, training accuracy %g" % (epochs, i, train_accuracy))
        train_step.run(feed_dict={x: train_data, y_: train_label, phase_train: True})

acc = 0
for i in range(10):
    batch = mnist.test.next_batch(1000)
    test_data = batch[0]
    test_label = batch[1]
    acc += accuracy.eval(feed_dict={x: test_data, y_: test_label, phase_train: False})
acc = acc / 10
print("test accuracy %g" % acc)
