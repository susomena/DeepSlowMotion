from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim

from vgg16 import VGG16


def clip(x, min_value, max_value):
    """
    This function clips a value x to keep it between a minimum value and a
    maximum value.
    :param x: the value to be clipped
    :param min_value: the minimum value for x
    :param max_value: tha maximum value for x
    :return: the value of x clipped to the minimum and maximum values
    """
    return tf.math.maximum(min_value, tf.math.minimum(max_value, x))


def bilinear_warping(img, f):
    """
    This function computes the frame result of warping an initial frame img
    with an optical flow.
    :param img: the image to be warped
    :param f: optical flow
    :return: the warped image
    """
    batch_size, height, width, num_feature_maps = img.get_shape().as_list()

    f_x = f[:, :, :, 0:1]
    f_y = f[:, :, :, 1:2]

    v_x = (f_x + 1) / 2 * (width - 1)
    v_y = (f_y + 1) / 2 * (height - 1)

    v_x0 = clip(tf.floor(v_x), 0., width - 1.)
    v_x1 = clip(v_x0 + 1., 0., width - 1.)
    v_y0 = clip(tf.floor(v_y), 0., height - 1.)
    v_y1 = clip(v_y0 + 1., 0., height - 1.)

    v_00 = tf.cast(tf.concat([v_y0, v_x0], axis=3), dtype=tf.int32)
    v_01 = tf.cast(tf.concat([v_y1, v_x0], axis=3), dtype=tf.int32)
    v_10 = tf.cast(tf.concat([v_y0, v_x1], axis=3), dtype=tf.int32)
    v_11 = tf.cast(tf.concat([v_y1, v_x1], axis=3), dtype=tf.int32)

    img_00 = tf.cast(tf.gather_nd(img, v_00, batch_dims=1), dtype=tf.float32)
    img_01 = tf.cast(tf.gather_nd(img, v_01, batch_dims=1), dtype=tf.float32)
    img_10 = tf.cast(tf.gather_nd(img, v_10, batch_dims=1), dtype=tf.float32)
    img_11 = tf.cast(tf.gather_nd(img, v_11, batch_dims=1), dtype=tf.float32)

    w_00 = (v_x1 - v_x) * (v_y1 - v_y)
    w_01 = (v_x1 - v_x) * (1 - (v_y1 - v_y))
    w_10 = (1 - (v_x1 - v_x)) * (v_y1 - v_y)
    w_11 = (1 - (v_x1 - v_x)) * (1 - (v_y1 - v_y))

    o = tf.add_n([w_00 * img_00, w_01 * img_01, w_10 * img_10, w_11 * img_11])

    return o,


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
        shape = tf.shape(enc_5)[1:3]
        net = tf.image.resize_bilinear(net, shape)
        net = tf.concat([enc_5, net], axis=3)
        net = slim.conv2d(net, 512, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv1_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 512, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv1_2')
        net = tf.nn.leaky_relu(net, alpha=0.1)

        # DECODER HIERARCHY 2
        shape = tf.shape(enc_4)[1:3]
        net = tf.image.resize_bilinear(net, shape)
        net = tf.concat([enc_4, net], axis=3)
        net = slim.conv2d(net, 256, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv2_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 256, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv2_2')
        net = tf.nn.leaky_relu(net, alpha=0.1)

        # DECODER HIERARCHY 3
        shape = tf.shape(enc_3)[1:3]
        net = tf.image.resize_bilinear(net, shape)
        net = tf.concat([enc_3, net], axis=3)
        net = slim.conv2d(net, 128, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv3_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 128, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv3_2')
        net = tf.nn.leaky_relu(net, alpha=0.1)

        # DECODER HIERARCHY 4
        shape = tf.shape(enc_2)[1:3]
        net = tf.image.resize_bilinear(net, shape)
        net = tf.concat([enc_2, net], axis=3)
        net = slim.conv2d(net, 64, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv4_1')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = slim.conv2d(net, 64, [3, 3], activation_fn=None,
                          scope=scope_prefix + '_dec_conv4_2')
        net = tf.nn.leaky_relu(net, alpha=0.1)

        # DECODER HIERARCHY 5
        shape = tf.shape(enc_1)[1:3]
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


def flow_computation_model(x):
    """
    This function returns the model of the flow computation network.
    :param x: the input of the flow computation network
    :return: the output of the flow computation network
    """
    return unet_model(x, 4, 'lrelu', 'flow_comp')


def arbitrary_time_flow_interpolation_model(x):
    """
    This function returns the model of the arbitrary-time flow interpolation
    network.
    :param x: the input of the arbitrary-time flow interpolation network
    :return: the output of the arbitrary-time flow interpolation network
    """
    return unet_model(x, 5, 'tanh', 'flow_interp')


def charbonnier_loss(ground_truth, prediction, e=0.01):
    """
    This function computes the Charbonnier loss of a predicted tensor.
    :param ground_truth: the tensor to be predicted
    :param prediction: the predicted tensor
    :param e: small quantity added to the squared difference between the ground
    truth and the prediction used to smooth the loss function
    :return: the value of the Charbonnier loss
    """
    return tf.reduce_sum(tf.sqrt(
        tf.square(ground_truth - prediction) + e ** 2), axis=3)


def l2_loss(ground_truth, prediction):
    """
    This function computes the L2 loss of a predicted tensor.
    :param ground_truth: the tensor to be predicted
    :param prediction: the predicted tensor
    :return: the value of the L2 loss
    """
    return tf.reduce_sum(tf.square(ground_truth - prediction), axis=3)


def reconstruction_loss(ground_truth, prediction):
    """
    This function computes the reconstruction loss of a list of predicted
    intermediate frames. Instead of using L1 loss like in the original work of
    Jiang at al., Charbonnier loss is used in order to have a smooth loss
    function at zero.
    :param ground_truth: list of real intermediate frames to be predicted
    :param prediction: list of predicted intermediate frames
    :return: the value of the reconstruction loss
    """
    l_r = 0.

    for i in range(len(ground_truth)):
        l_r += charbonnier_loss(ground_truth[i], prediction[i])

    l_r /= len(ground_truth)

    return l_r


def perceptual_loss(ground_truth, prediction):
    """
    This function computes the perceptual loss of a list of predicted
    intermediate frames. According to Jiang et al., this loss function is equal
    to the mean squared L2 norm of the difference between the activation of the
    conv4_3 layer of the VGG16 layer for the intermediate frame to be predicted
    and the predicted intermediate frame.
    :param ground_truth: list of real intermediate frames to be predicted
    :param prediction: list of predicted intermediate frames
    :return: the value of the perceptual loss
    """
    vgg_net = VGG16()
    l_p = 0.

    for i in range(len(ground_truth)):
        ground_truth_feat = vgg_net(ground_truth[i])
        prediction_feat = vgg_net(prediction[i])

        l_p += l2_loss(ground_truth_feat, prediction_feat)

    l_p /= len(ground_truth)

    return l_p


def warping_loss(i_0, i_1, i_t, f_01, f_10, f_t0, f_t1):
    """
    This function computes the warping loss of the reconstruction of all the
    frames using the predicted optical flow.
    :param i_0: first frame of the sample (ground truth)
    :param i_1: last frame of the sample (ground truth)
    :param i_t: list of intermediate frames (ground truth)
    :param f_01: predicted optical flow from the first frame to the last frame
    :param f_10: predicted optical flow from the last frame to the first frame
    :param f_t0: list of predicted optical flows from the intermediate frames
    to the first frame
    :param f_t1: list of predicted optical flows from the intermediate frames
    to the last frame
    :return: the value of the warping loss
    """
    l_w = charbonnier_loss(i_0, bilinear_warping(i_1, f_01))
    l_w += charbonnier_loss(i_1, bilinear_warping(i_0, f_10))

    for i in range(len(i_t)):
        l_w += (1. / len(i_t)) * charbonnier_loss(
            i_t[i], bilinear_warping(i_0, f_t0[i]))

        l_w += (1. / len(i_t)) * charbonnier_loss(
            i_t[i], bilinear_warping(i_1, f_t1[i]))

    return l_w


def smoothness_loss(f_01, f_10):
    """
    This function computes the smoothness loss of the predicted optical flows
    between the first and the last frames. According to Jiang et al., the
    smoothness loss is computed as the total variation loss of the optical
    flows.
    :param f_01: optical flow from the first to the last frame
    :param f_10: optical flow from the last to the first frame
    :return: the value of the smoothness loss
    """
    l_s = charbonnier_loss(f_01[:, 1:, :, :], f_01[:, :-1, :, :])
    l_s += charbonnier_loss(f_01[:, :, 1:, :], f_01[:, :, :-1, :])

    l_s += charbonnier_loss(f_10[:, 1:, :, :], f_10[:, :-1, :, :])
    l_s += charbonnier_loss(f_10[:, :, 1:, :], f_10[:, :, :-1, :])

    return l_s
