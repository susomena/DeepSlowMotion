from __future__ import print_function

import argparse

from dataset import Data

parser = argparse.ArgumentParser(description='Trains a neural network for video'
                                             'frame interpolation.')

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

args = parser.parse_args()

data = Data(args.datasets, args.frames + 2, args.train_percentage,
            args.val_percentage, args.data_augmentation)
