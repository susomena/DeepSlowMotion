from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf

from dataset import Data
import model

parser = argparse.ArgumentParser(description='Trains a neural network for'
                                             'video frame interpolation.')

parser.add_argument('-d', '--datasets', type=str,
                    help='List of directories where data is located',
                    nargs='+', metavar=('DIR 1', 'DIR 2'))
parser.add_argument('-tp', '--train-percentage', default=50, type=int,
                    help='Percentage of samples for the training set')
parser.add_argument('-vp', '--val-percentage', default=25, type=int,
                    help='Percentage of samples for the validation set')
parser.add_argument('-da', '--data-augmentation', action='store_true',
                    help='Whether data augmentation should be used or not')
parser.add_argument('-f', '--frames', default=9, type=int,
                    help='Number of frames to be interpolated')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    help='Size of the training mini-batches')
parser.add_argument('-lr', '--lambda-r', default=0.8, type=float,
                    help='Weight for the reconstruction loss')
parser.add_argument('-lp', '--lambda-p', default=0.005, type=float,
                    help='Weight for the perceptual loss')
parser.add_argument('-lw', '--lambda-w', default=0.4, type=float,
                    help='Weight for the warping loss')
parser.add_argument('-ls', '--lambda-s', default=1., type=float,
                    help='Weight for the smoothness loss')
parser.add_argument('-l', '--learning-rate', default=0.0001, type=float,
                    help='Initial value for the learning rate')
parser.add_argument('-de', '--decay-epochs', default=200, type=int,
                    help='Number of epochs for decreasing learning rate')
parser.add_argument('-df', '--decay-factor', default=0.1, type=float,
                    help='Factor for decreasing learning rate')
parser.add_argument('-e', '--epochs', default=500, type=int,
                    help='Number of training epochs')

args = parser.parse_args()

data = Data(args.datasets, args.frames + 2, args.train_percentage,
            args.val_percentage, args.data_augmentation)

graph = tf.Graph()
with graph.as_default():
    tf_i0 = tf.placeholder(dtype=tf.float32,
                           shape=(args.batch_size, 720, 1280, 3))
    tf_i1 = tf.placeholder(dtype=tf.float32,
                           shape=(args.batch_size, 720, 1280, 3))

    tf_frames = []
    for i in range(args.frames):
        tf_frames.append(tf.placeholder(
            dtype=tf.float32, shape=(args.batch_size, 720, 1280, 3)))

    flow_comp_input = tf.concat([tf_i0, tf_i1], axis=3)
    flow_comp_output = model.flow_computation_model(flow_comp_input)

    f_01 = flow_comp_output[:, :, :, :2]
    f_10 = flow_comp_output[:, :, :, 2:]

    interpolated_frames = []
    flow_t0 = []
    flow_t1 = []
    reuse = None
    for t in np.arange(1 / (args.frames + 1), 1., 1 / (args.frames + 1)):
        f_t0 = -(1 - t) * t * f_01 + t ** 2 * f_10
        f_t1 = (1 - t) ** 2 * f_01 - t * (1 - t) * f_10

        flow_t0.append(f_t0)
        flow_t1.append(f_t1)

        g_0 = model.bilinear_warping(tf_i0, f_t0)
        g_1 = model.bilinear_warping(tf_i1, f_t1)

        concat_list = [tf_i0, tf_i1, f_01, f_10, f_t0, f_t1, g_0, g_1]

        flow_interp_input = tf.concat(concat_list, axis=3)
        o = model.arbitrary_time_flow_interpolation_model(flow_interp_input,
                                                          reuse=reuse)

        v_t0 = o[:, :, :, :1]
        v_t1 = o[:, :, :, 1:2]
        delta_f_t0 = o[:, :, :, 2:4]
        delta_f_t1 = o[:, :, :, 4:6]

        g_0 = model.bilinear_warping(tf_i0, f_t0 + delta_f_t0)
        g_1 = model.bilinear_warping(tf_i1, f_t1 + delta_f_t1)

        z = (1 - t) * v_t0 + t * v_t1
        i_t = 1 / z * ((1 - t) * v_t0 * g_0 + t * v_t1 * g_1)

        interpolated_frames.append(i_t)
        reuse = True

    l_r = model.reconstruction_loss(tf_frames, interpolated_frames)
    l_p = model.perceptual_loss(tf_frames, interpolated_frames)
    l_w = model.warping_loss(tf_i0, tf_i1, tf_frames,
                             f_01, f_10, flow_t0, flow_t1)
    l_s = model.smoothness_loss(f_01, f_10)

    loss = args.lambda_r * l_r
    loss += args.lambda_p * l_p
    loss += args.lambda_w * l_w
    loss += args.lambda_s * l_s

    global_step = tf.Variable(0)
    decay_iter = args.decay_epochs * data.num_train_batches(args.batch_size)
    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step,
                                               decay_iter, args.decay_factor,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                               global_step)

    saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    print('Initialized')

    start_epoch = 0
    min_validation_loss = -1
    if os.path.exists('model/model.index'):
        saver.restore(session, 'model/model')
        start_epoch = session.run(global_step)
        start_epoch /= data.num_train_batches(args.batch_size)

        vf = open('model/val')
        min_validation_loss = float(vf.read())
        vf.close()

    if args.data_augmentation:
        data_retrieval_functions = [data.get_training_batch,
                                    data.get_training_flipped_batch]
    else:
        data_retrieval_functions = [data.get_training_batch]

    start_epoch /= len(data_retrieval_functions)
    for epoch in range(int(start_epoch), args.epochs):
        data.shuffle()

        epoch_loss = 0.
        for batch in range(data.num_train_batches(args.batch_size)):
            for f in data_retrieval_functions:
                i0, i1, it = f(batch, args.batch_size)
                d = {tf_i0: i0, tf_i1: i1}

                for i in range(args.frames):
                    d[tf_frames[i]] = it[i]

                _, step_loss = session.run([optimizer, loss], feed_dict=d)
                epoch_loss += step_loss

        epoch_loss /= data.num_train_batches(args.batch_size)
        epoch_loss /= len(data_retrieval_functions)
        print('Mean loss at epoch %d: %f' % (epoch, epoch_loss))

        validation_loss = 0.
        for batch in range(data.num_valid_batches(args.batch_size)):
            i0, i1, it = data.get_validation_batch(batch, args.batch_size)
            d = {tf_i0: i0, tf_i1: i1}

            for i in range(args.frames):
                d[tf_frames[i]] = it[i]

            step_loss = session.run([loss], feed_dict=d)[0]
            validation_loss += step_loss

        validation_loss /= data.num_valid_batches(args.batch_size)
        print('Validation loss: %f' % validation_loss)

        if min_validation_loss == -1 or min_validation_loss > validation_loss:
            min_validation_loss = validation_loss
            print('Saving model...')

            if not os.path.exists('model'):
                os.mkdir('model')

            saver.save(session, 'model/model')

            vf = open('model/val', 'w')
            vf.write('%f' % min_validation_loss)
            vf.close()

    test_loss = 0.
    for batch in range(data.num_test_batches(args.batch_size)):
        i0, i1, it = data.get_test_batch(batch, args.batch_size)
        d = {tf_i0: i0, tf_i1: i1}

        for i in range(args.frames):
            d[tf_frames[i]] = it[i]

        step_loss = session.run([loss], feed_dict=d)[0]
        test_loss += step_loss

    test_loss /= data.num_test_batches(args.batch_size)
    print('Test loss: %f' % test_loss)
