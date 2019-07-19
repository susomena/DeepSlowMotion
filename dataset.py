from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
from scipy.misc import imread


class Data:
    """

    A class for managing the dataset.

    """

    def __init__(self, path_list, sample_size, train_percentage,
                 val_percentage, data_augmentation):
        """
        Constructor of the Data class. This constructor generates the training,
        validation and test sets and performs data augmentation on the training
        set if required.
        :param path_list: list of directories where data is located
        :param sample_size: number of frames that compose a sample (number of
        frames to be interpolated plus two frames at the beginning and the end
        of the frame sequence.
        :param train_percentage: percentage of the data used as the training
        set
        :param val_percentage: percentage of the data used as the validation
        set
        :param data_augmentation: whether data augmentation should be used or
        not
        :type path_list: list[str]
        :type sample_size: int
        :type train_percentage: int
        :type val_percentage: int
        :type data_augmentation: bool
        """
        frames = get_frames(path_list)
        samples = get_samples(frames, sample_size)

        train_limit = (train_percentage / 100) * len(samples)
        val_limit = ((train_percentage + val_percentage) / 100) * len(samples)
        self._train_set = samples[0:int(train_limit)]
        self._validation_set = samples[int(train_limit):int(val_limit)]
        self._test_set = samples[int(val_limit):]

        if data_augmentation:
            self._augment_train_set()

    def _augment_train_set(self):
        """
        This function performs data augmentation on the training set by adding
        the same frame sequences in inverse order.
        """
        reversed_samples = []

        for s in self._train_set:
            reversed_samples.append(list(reversed(s)))

        self._train_set += reversed_samples

    def get_num_batches(self, batch_size):
        """
        This function returns the number of batches in the training set for a
        given batch size.
        :param batch_size: size of the batches
        :type batch_size: int
        :return: number of batches in the training set for the given batch size
        :rtype: int
        """
        return len(self._train_set) // batch_size

    def shuffle(self):
        """
        This function randomly permutes the training set, so next time batches
        won't be the same.
        """
        random.shuffle(self._train_set)

    def get_batch(self, i, batch_size):
        """
        This function returns the batch in position i for batches with a given
        batch size.
        :param i: index of the batch in the training set
        :param batch_size: size of the batches
        :type i: int
        :type batch_size: int
        :return: training batch in position i for the given batch size
        :rtype: (list, list, list[list])
        """
        batch_files = self._train_set[i * batch_size:(i + 1) * batch_size]

        i0 = []
        i1 = []
        for s in batch_files:
            i0.append((imread(s[0]) - 127.5) / 255.)
            i1.append((imread(s[-1]) - 127.5) / 255.)

        it = [[] for _ in range(len(batch_files[0]) - 2)]
        for i in range(len(it)):
            for j in range(batch_size):
                it[i].append((imread(batch_files[j][i + 1]) - 127.5) / 255.)

        return i0, i1, it

    def get_flipped_batch(self, i, batch_size):
        """
        This function returns the batch in position i for batches with a given
        batch size. The pictures in this batch are horizontally flipped for
        data augmentation.
        :param i: index of the batch in the training set
        :param batch_size: size of the batches
        :type i: int
        :type batch_size: int
        :return: training batch in position i for the given batch size with
        horizontally flipped pictures
        :rtype: (list, list, list[list])
        """
        batch_files = self._train_set[i * batch_size:(i + 1) * batch_size]

        i0 = []
        i1 = []
        for s in batch_files:
            i0.append(np.fliplr((imread(s[0]) - 127.5) / 255.))
            i1.append(np.fliplr((imread(s[-1]) - 127.5) / 255.))

        it = [[] for _ in range(len(batch_files[0]) - 2)]
        for i in range(len(it)):
            for j in range(batch_size):
                it[i].append(np.fliplr(
                    (imread(batch_files[j][i + 1]) - 127.5) / 255.))

        return i0, i1, it


def get_frames(path_list):
    """
    This function retrieves all the frames from the dataset directories.
    :param path_list: list of directories of the dataset.
    :type path_list: list[str]
    :return: list of frames separated per video.
    :rtype: list[list[str]]
    """
    frames = []

    for d in path_list:
        for sd in os.listdir(d):
            if os.path.isdir(os.path.join(d, sd)):
                frames.append([])
                for f in sorted(os.listdir(os.path.join(d, sd))):
                    if os.path.isfile(os.path.join(d, sd, f)):
                        file_path = os.path.abspath(os.path.join(d, sd, f))
                        frames[-1].append(file_path)

    return frames


def get_samples(frames, sample_size):
    """
    This function separates a list of frames into a list of samples, i.e.,
    sequences of frames.
    :param frames: list of frames separated per video.
    :param sample_size: number of frames per sample.
    :type frames: list[list[str]]
    :type sample_size: int
    :return: list of samples.
    :rtype: list[list[str]]
    """
    samples = []

    for v in frames:
        for i in range(len(v) // sample_size):
            samples.append(v[i * sample_size:(i + 1) * sample_size])

    return samples
