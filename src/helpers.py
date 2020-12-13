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
        lower = mid - std
        upper = mid + std
    elif type in ["median", "order", "quantiles"]:
        mid = np.median(array, axis=axis)
        lower = np.percentile(array, 25, axis=axis)
        upper = np.percentile(array, 75, axis=axis)
    else:
        raise NotImplementedError
    return np.stack((mid, lower, upper))


def get_DataLoaders(
    roof_dataset: AvailableRooftopDataset,
    train_percentage: float,
    validation_percentage: float,
    test_percentage: float,
    batch_size: int,
    seed: int = 42,
):
    """
    Inputs:
    ========
    roof_dataset : AvailableRooftopDataset
        Dataset to be used.
    train_percentage : float
        Percentage of images to be used for training.
    validation_percentage : float
        Percentage of images to be used for validation.
    test_percentage : float
        Percentage of images to be used for testing.
    batch_size : int
        Batch size of the DataLoaders.
    seed : int, optional
        Seed to use durig split. Should not be changed for consistent results.
        The default is 42.

    Raises:
    ========
    ValueError
        If the percentages do not sum to 1.

    Returns:
    ========
    roof_dataloader_train : torch.utils.data.DataLoader
        Training DataLoader.
    roof_dataloader_validation : torch.utils.data.DataLoader
        Validation DataLoader.
    roof_dataloader_test : torch.utils.data.DataLoader
        Test DataLoader.
    """

    sum_percentages = train_percentage + validation_percentage + test_percentage
    if abs(sum_percentages - 1) > 1e-8:
        raise ValueError(
            f"The sum of all 3 percentages should be 1. Got {sum_percentages}."
        )

    # Split the dataset in train_set and test_set
    dataset_length = len(roof_dataset)
    train_dataset_length = int(dataset_length * train_percentage)
    validation_dataset_length = int(dataset_length * validation_percentage)
    test_dataset_length = (
        dataset_length - train_dataset_length - validation_dataset_length
    )

    roof_dataset_train, roof_dataset_validation, roof_dataset_test, = random_split(
        roof_dataset,
        [train_dataset_length, validation_dataset_length, test_dataset_length],
        generator=torch.Generator().manual_seed(seed),
    )

    # Create dataloaders associated to train/test set
    roof_dataloader_train = DataLoader(
        roof_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    roof_dataloader_validation = DataLoader(
        roof_dataset_validation, batch_size=batch_size, shuffle=True, num_workers=0
    )
    roof_dataloader_test = DataLoader(
        roof_dataset_test, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return roof_dataloader_train, roof_dataloader_validation, roof_dataloader_test
