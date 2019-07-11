import tensorflow as tf
import tensorflow.contrib.slim as slim


def unet_model(x, output_feature_maps, output_activation_fn, scope_prefix):
    """
    Generic U-net model for both, the flow computation network and the
    arbitrary-time flow interpolation network.
    :param x: input tensor of the network
    :param output_feature_maps: number of feature maps in the output of the
    network
    :param output_activation_fn: string indicating what activation function to
    use for the output of the network. Only 'lrelu' and 'tanh' values are
    allowed
    :param scope_prefix: prefix to differentiate the scope of the flow
    computation network and the scope of the arbitrary-time flow interpolation
    network
    :return: the output of the network
    """
    with slim.arg_scope([slim.conv2d], padding='SAME'):
        # ENCODER HIERARCHY 1
        net = slim.conv2d(x, 32, [7, 7], activation_fn=None,
                          scope=scope_prefix + '_enc_conv1_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 32, [7, 7], activation_fn=None,
                          scope=scope_prefix + '_enc_conv_1_2')
        enc_1 = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.avg_pool2d(enc_1, [2, 2])

        # ENCODER HIERARCHY 2
        net = slim.conv2d(net, 64, [5, 5], activation_fn=None,
                          scope=scope_prefix + '_enc_conv2_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 64, [5, 5], activation_fn=None,
                          scope=scope_prefix + '_enc_conv_2_2')
        enc_2 = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.avg_pool2d(enc_2, [2, 2])

        # ENCODER HIERARCHY 3
        net = slim.conv2d(net, 128, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_enc_conv3_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 128, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_enc_conv_3_2')
        enc_3 = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.avg_pool2d(enc_3, [2, 2])

        # ENCODER HIERARCHY 4
        net = slim.conv2d(net, 256, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_enc_conv4_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 256, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_enc_conv_4_2')
        enc_4 = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.avg_pool2d(enc_4, [2, 2])

        # ENCODER HIERARCHY 5
        net = slim.conv2d(net, 512, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_enc_conv5_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 512, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_enc_conv_5_2')
        enc_5 = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.avg_pool2d(enc_5, [2, 2])

        # ENCODER HIERARCHY 6
        net = slim.conv2d(net, 512, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_enc_conv6_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 512, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_enc_conv_6_2')
        net = tf.nn.leaky_relu(net, alpha=0.1)

        # DECODER HIERARCHY 1
        shape = tf.slice(tf.shape(enc_5), [1], [2])
        net = tf.image.resize_bilinear(net, shape)
        net = tf.concat([enc_5, net], axis=3)
        net = slim.conv2d(net, 512, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv1_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 512, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv1_2')
        net = tf.nn.leaky_relu(net, alpha=0.1)

        # DECODER HIERARCHY 2
        shape = tf.slice(tf.shape(enc_4), [1], [2])
        net = tf.image.resize_bilinear(net, shape)
        net = tf.concat([enc_4, net], axis=3)
        net = slim.conv2d(net, 256, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv2_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 256, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv2_2')
        net = tf.nn.leaky_relu(net, alpha=0.1)

        # DECODER HIERARCHY 3
        shape = tf.slice(tf.shape(enc_3), [1], [2])
        net = tf.image.resize_bilinear(net, shape)
        net = tf.concat([enc_3, net], axis=3)
        net = slim.conv2d(net, 128, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv3_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 128, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv3_2')
        net = tf.nn.leaky_relu(net, alpha=0.1)

        # DECODER HIERARCHY 4
        shape = tf.slice(tf.shape(enc_2), [1], [2])
        net = tf.image.resize_bilinear(net, shape)
        net = tf.concat([enc_2, net], axis=3)
        net = slim.conv2d(net, 64, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv4_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 64, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv4_2')
        net = tf.nn.leaky_relu(net, alpha=0.1)

        # DECODER HIERARCHY 5
        shape = tf.slice(tf.shape(enc_1), [1], [2])
        net = tf.image.resize_bilinear(net, shape)
        net = tf.concat([enc_1, net], axis=3)
        net = slim.conv2d(net, 32, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv5_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, output_feature_maps, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv5_2')

        if output_activation_fn == 'lrelu':
            y = tf.nn.leaky_relu(net, alpha=0.1)
        elif output_activation_fn == 'tanh':
            y = 127.5 * (tf.nn.tanh(net) + 1)  # Scaling back to [0, 255]
        else:
            raise ValueError('The output activation function can only be leaky'
                             'ReLU or hyperbolic tangent')

    return y
