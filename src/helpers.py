import numpy as np
import os
import re
import torch

from AvailableRooftopDataset import AvailableRooftopDataset
from torch.utils.data import DataLoader, random_split


def get_label_file(filename_image):
    """Get the name of the corresponding label file"""
    filename, file_extension = os.path.splitext(filename_image)
    filename_label = filename + "_label"
    return filename_label + file_extension


def get_image_file(filename_label):
    """Get the name of the corresponding image file"""
    filename, file_extension = os.path.splitext(filename_label)
    file = filename + file_extension
    filename_image = re.sub("_label" + file_extension, file_extension, file)
    return filename_image


def has_label(filename_image, label_folder):
    """Check that the image whether a corresponding label file"""
    filename_label = get_label_file(filename_image)
    path_label = os.path.join(label_folder, filename_label)
    return path_label.is_file()


def summary_stats(array, axis=0, type="median"):
    """Summary statistics of array of given type.

    Inputs:
    =========
    array : ndarray
        Array to summarize.
    axis : int, optional
        Axis along which to summarize.
    type : str
        Type of summary to produce.
        For mean and standard deviation, give one of ["mean", "average", "avg"].
        For order statistics, give one of ["median", "order", "quantiles"].

    Raises:
    =========
    NotImplementedError
        When the type is not recognized.

    Returns:
    =========
    summary : ndarray
        Contains summary statistics of array for each column:
        First row is mid-point (e.g. mean or median)
        Second row is lower bound (e.g. mean - std or first quartile)
        Third row is upper bound (e.g. mean + std or third quartile)
    """
    if type in ["mean", "average", "avg"]:
        mid = np.mean(array, axis=axis)
        std = np.std(array, axis=axis)
        lower = avg - std
        upper = avg + std
    elif type in ["median", "order", "quantiles"]:
        mid = np.median(array, axis=axis)
        lower = np.percentile(array, 25, axis=axis)
        upper = np.percentile(array, 75, axis=axis)
    else:
        raise NotImplementedError
    return np.stack((mid, lower, upper))
