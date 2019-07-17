from __future__ import division
from __future__ import print_function

import os


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
            reversed_samples.append(reversed(s))

        self._train_set += reversed_samples

    def get_epoch_iterations(self, batch_size):
        """
        This function returns the number of iterations for an epoch with a
        given batch size.
        :param batch_size: size of the batches
        :return: number of iterations for an epoch with the given batch size
        """
        return len(self._train_set) / batch_size


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
