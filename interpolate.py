from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
from scipy.misc import imread, imsave
import tensorflow as tf

from dataset import get_frames
import model

parser = argparse.ArgumentParser(description='Interpolates intermediate frames'
                                             ' for a video using a neural '
                                             'network.')

parser.add_argument('-id', '--input-dir', type=str,
                    help='Directory where original frames are located')
parser.add_argument('-od', '--output-dir', type=str,
                    help='Directory where interpolated frames should be saved')
parser.add_argument('-f', '--frames', default=9, type=int,
                    help='Number of frames to be interpolated')

args = parser.parse_args()

original_frames = get_frames([args.input_dir])[0]
height, width, _ = imread(original_frames[0]).shape

graph = tf.Graph()
with graph.as_default():
    tf_i0 = tf.placeholder(dtype=tf.float32, shape=(1, height, width, 3))
    tf_i1 = tf.placeholder(dtype=tf.float32, shape=(1, height, width, 3))

    flow_comp_input = tf.concat([tf_i0, tf_i1], axis=3)
    flow_comp_output = model.flow_computation_model(flow_comp_input)

    f_01 = flow_comp_output[:, :, :, :2]
    f_10 = flow_comp_output[:, :, :, 2:]

    interpolated_frames = []
    reuse = None
    for t in np.arange(1 / (args.frames + 1), 1., 1 / (args.frames + 1)):
        f_t0 = -(1 - t) * t * f_01 + t ** 2 * f_10
        f_t1 = (1 - t) ** 2 * f_01 - t * (1 - t) * f_10

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

    saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    print('Initialized')

    if os.path.exists('model/model.index'):
        saver.restore(session, 'model/model')
    else:
        print('No weights to restore. Maybe you forgot to train the model?')
        exit(1)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    imsave(args.output_dir + '00001.png', imread(original_frames[0]))
    n = 2

    for i in range(1, len(original_frames)):
        i0 = [imread(original_frames[i - 1])]
        i1 = [imread(original_frames[i])]
        d = {tf_i0: i0, tf_i1: i1}

        frames = session.run(interpolated_frames, feed_dict=d)

        for frame in frames:
            imsave(args.output_dir + '%05d.png' % n, frame[0])
            n += 1

        imsave(args.output_dir + '%05d.png' % n, i1[0])
        n += 1

        print('%.2f%% completed' % (i * 100 / len(original_frames)))

print('100.00% completed')
