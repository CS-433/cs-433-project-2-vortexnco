import numpy as np


def summary_stats(array, axis=0, type="median", lower_bound=None):
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
    lower_bound : float
        Lower bound to use; upper bound is defined symmetrically.
        If type is "mean", this is how many standard deviations to go away from the mean.
        If type is "median", this is the lower percentile (<50).

    Raises:
    =========
    NotImplementedError
        When the type is not recognized.

    Returns:
    =========
    summary : ndarray
        Contains summary statistics of array for each column:
        First row is lower bound (e.g. mean - std or first quartile)
        Second row is mid-point (e.g. mean or median)
        Third row is upper bound (e.g. mean + std or third quartile)
    """
    if type in ["mean", "average", "avg"]:
        if not lower_bound:
            lower_bound = 1
        else:
            assert(lower_bound >= 0)
        mid = np.mean(array, axis=axis)
        std = np.std(array, axis=axis)
        lower = mid - lower_bound * std
        upper = mid + lower_bound * std
    elif type in ["median", "order", "quantiles"]:
        if not lower_bound:
            lower_bound = 25
        else:
            assert(lower_bound >= 0 and lower_bound <= 50)
        mid = np.median(array, axis=axis)
        lower = np.percentile(array, lower_bound, axis=axis)
        upper = np.percentile(array, 100 - lower_bound, axis=axis)
    else:
        raise NotImplementedError
    return np.stack((lower, mid, upper), axis=axis)
