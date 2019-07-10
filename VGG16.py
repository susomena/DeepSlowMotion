import numpy as np
import tensorflow as tf


class VGG16:

    """

    VGG16 neural network class.

    """

    def __init__(self):
        """
        Constructor of the VGG16 class. Pretrained weights are loaded from a
        numpy file. The mean of the ImageNet dataset used to train VGG16 is
        also loaded, so images can be centered around zero on inference.
        """
        weights = np.load('vgg16_weights_no_fc.npz')

        self._conv1_1_kernel = tf.constant(weights['conv1_1_W'])
        self._conv1_1_biases = tf.constant(weights['conv1_1_b'])
        self._conv1_2_kernel = tf.constant(weights['conv1_2_W'])
        self._conv1_2_biases = tf.constant(weights['conv1_2_b'])
        self._conv2_1_kernel = tf.constant(weights['conv2_1_W'])
        self._conv2_1_biases = tf.constant(weights['conv2_1_b'])
        self._conv2_2_kernel = tf.constant(weights['conv2_2_W'])
        self._conv2_2_biases = tf.constant(weights['conv2_2_b'])
        self._conv3_1_kernel = tf.constant(weights['conv3_1_W'])
        self._conv3_1_biases = tf.constant(weights['conv3_1_b'])
        self._conv3_2_kernel = tf.constant(weights['conv3_2_W'])
        self._conv3_2_biases = tf.constant(weights['conv3_2_b'])
        self._conv3_3_kernel = tf.constant(weights['conv3_3_W'])
        self._conv3_3_biases = tf.constant(weights['conv3_3_b'])
        self._conv4_1_kernel = tf.constant(weights['conv4_1_W'])
        self._conv4_1_biases = tf.constant(weights['conv4_1_b'])
        self._conv4_2_kernel = tf.constant(weights['conv4_2_W'])
        self._conv4_2_biases = tf.constant(weights['conv4_2_b'])
        self._conv4_3_kernel = tf.constant(weights['conv4_3_W'])
        self._conv4_3_biases = tf.constant(weights['conv4_3_b'])

        self._mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32,
                                 shape=[1, 1, 1, 3])

    def __call__(self, x):
        """
        This function performs inference on an image x (except for the fully
        connected layers). This function returns the feature maps of the
        conv_4_3 layer that will be used for the loss computation as in Jiang
        et al.
        :param x: input image
        :return: feature maps of the conv_4_3 layer
        """
        # Centering input around zero
        net = x - self._mean

        # CONV 1_1
        net = tf.nn.conv2d(net, self._conv1_1_kernel, [1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, self._conv1_1_biases))

        # CONV 1_2
        net = tf.nn.conv2d(net, self._conv1_2_kernel, [1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, self._conv1_2_biases))

        net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')

        # CONV 2_1
        net = tf.nn.conv2d(net, self._conv2_1_kernel, [1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, self._conv2_1_biases))

        # CONV 2_2
        net = tf.nn.conv2d(net, self._conv2_2_kernel, [1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, self._conv2_2_biases))

        net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')

        # CONV 3_1
        net = tf.nn.conv2d(net, self._conv3_1_kernel, [1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, self._conv3_1_biases))

        # CONV 3_2
        net = tf.nn.conv2d(net, self._conv3_2_kernel, [1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, self._conv3_2_biases))

        # CONV 3_3
        net = tf.nn.conv2d(net, self._conv3_3_kernel, [1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, self._conv3_3_biases))

        net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')

        # CONV 4_1
        net = tf.nn.conv2d(net, self._conv4_1_kernel, [1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, self._conv4_1_biases))

        # CONV 4_2
        net = tf.nn.conv2d(net, self._conv4_2_kernel, [1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, self._conv4_2_biases))

        # CONV 4_3
        net = tf.nn.conv2d(net, self._conv4_3_kernel, [1, 1, 1, 1],
                           padding='SAME')
        conv_4_3 = tf.nn.relu(tf.nn.bias_add(net, self._conv4_3_biases))

        return conv_4_3
