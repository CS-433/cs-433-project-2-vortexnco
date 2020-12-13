import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import random

from PIL import Image
from itertools import product
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from AvailableRooftopDataset import AvailableRooftopDataset
from model.unet_model import UNet
from pipeline import load_data
from helpers import *


## VISUALISATION
map_rgb = {0: [0, 255, 0], 1: [0, 0, 0], 2: [255, 0, 0], 3: [255, 215, 0]}

# If color values are binary
COMPARE_MAP_01 = {(2, 0): 0, (0, 0): 1, (1, 1): 2, (1, -1): 3}

# If color values are 3 bytes
COMPARE_MAP_uint8 = {(254, 0): 0, (0, 0): 1, (255, 255): 2, (255, 1): 3}


def compare_labels(true_label, predicted_label):
    """Outputs an array annotated as TP, FP, TN or FN as ints,
    according to the chosen COMPARE_MAP"""
    height, width = true_label.shape
    comp_array = np.array([predicted_label + true_label, predicted_label - true_label])
    result = np.empty((height, width), dtype=int)
    for i, j in product(range(height), range(height)):
        result[i, j] = COMPARE_MAP_uint8[tuple(comp_array[:, i, j])]
    return result


def show_label_comparison(true_label, predicted_label):
    """
    Plots an array annotated with TP, FP, TN and FN

    Parameters
    ----------
    true_label : ndarray of 1s and 0s
        True label.
    predicted_label : ndarray of 1s and 0s
        Prediction from the model.

    Returns
    -------
    None
    """
    comparison = compare_labels(true_label, predicted_label)
    height, width = comparison.shape
    # Convert to RGB
    comparison_rgb = np.empty((height, width, 3), dtype=int)
    f = lambda i, j: map_rgb[comparison[i, j]]

    for i, j in product(range(height), range(width)):
        comparison_rgb[i, j, :] = map_rgb[comparison[i, j]]

    plt.imshow(comparison_rgb)
    TP = mpatches.Patch(color="green", label="TP")
    TN = mpatches.Patch(color="black", label="TN")
    FP = mpatches.Patch(color="red", label="FP")
    FN = mpatches.Patch(color=[255 / 255, 215 / 255, 0], label="FN")
    plt.legend(handles=[TP, FP, TN, FN], bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


## TESTING (once threshold is chosen)
def test(test_predictions, test_labels, threshold):
    """Returns evaluation measures over a test set.

    Inputs:
    ========
    test_predictions : ndarray
        Array of ints containing the predictions for each image in the test set.
        Predictions should be raw (not probabilities).
    test_labels : ndarray
        Array of ints containing the true labels for each image in the test set.
    threshold : float
        Threshold over which predictions should be decided as 1.

    Returns:
    ========
    f1, precision, recall, support : ndarray
        Evaluation measures for each prediction
    """
    n = len(test_predictions)
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)
    support = np.zeros(n)

    pred_probas = 1 / (1 + np.exp(-test_predictions))
    test_predictions = np.where(pred_probas > threshold, 1, 0)
    for i, true, pred in enumerate(zip(test_labels, test_predictions)):
        precision[i], recall[i], f1[i], support[i] = precision_recall_fscore_support(
            true.flatten(), pred.flatten()
        )
    return f1, precision, recall, support


## VALIDATION
def validation(predictions, labels, n_thresholds, plot=True):
    """Determine the best threshold given validation set and
    visualise results (precision-recall curve and F1-score against thresholds)

    Inputs:
    ========
    predictions : ndarray
        model predictions on validation set
    labels : ndarray
        true labels corresponding to predictions on validation set
    n_thresholds : int
        number of thresholds to test for
    plot : bool
        whether to plot results or not

    Returns:
    ========
    precision, recall, f1_scores : ndarray
        Evaluation measures for each threshold, for each image
    best_thresh : float
        The threshold which maximizes some value
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    pred_probas = 1 / (1 + np.exp(-predictions))

    precision = np.zeros((len(predictions), n_thresholds))
    recall = np.zeros((len(predictions), n_thresholds))
    f1_scores = np.zeros((len(predictions), n_thresholds))

    for i, (true, pred) in enumerate(zip(labels, pred_probas)):
        pred = pred.flatten()

        # Sort increasingly
        sort = np.argsort(pred)
        true = true.flatten()[sort]
        pred = pred[sort]

        # For each threshold, find the first index
        # for which we start predicting 1
        thresholds_idx = np.searchsorted(pred, thresholds)
        # Find indices of thresholds for which all are predicted 0
        limit_idx = np.where(thresholds_idx == len(pred))[0]
        # Use the indices to make thresholds_idx legal as an index array
        thresholds_idx[limit_idx] = 0
        # print("Threshold indices: {}".format(thresholds_idx))
        # True positives for each threshold
        tps = np.cumsum(true[::-1])[::-1][thresholds_idx]
        # If you never predict 1 you have no true positives
        tps[limit_idx] = 0
        # print("True positives: {}".format(tps))
        predicted_true = len(true) - thresholds_idx
        actually_true = tps[0]

        prec = tps / predicted_true
        # If you never predict 1 your precision is bad
        # But I need the precision-recall curve to make sense
        # and the F1-score to be defined
        prec[limit_idx] = 1
        precision[i] = prec
        with np.errstate(divide='ignore', invalid='ignore'):
            rec = tps / actually_true
        rec = np.nan_to_num(rec, 0)
        recall[i] = rec
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * (prec * rec) / (prec + rec)
        f1_scores[i] = np.nan_to_num(f1, 0)
        # print("Precision: {}".format(prec[-4:]))
        # print("Recall: {}".format(rec[-4:]))
        # print("F1: {}".format(f1_scores[i][-4:]))

        # ax_prec_rec.plot(rec, prec)

    prec_mid, prec_lower, prec_upper = summary_stats(precision, type='median')
    rec_mid, *_ = summary_stats(recall, type='median')
    f1_mid, f1_lower, f1_upper = summary_stats(f1_scores, type='median')

    plt.figure(1)
    ax_f1 = plt.axes()
    ax_f1.fill_between(thresholds, f1_lower, f1_upper, alpha=0.6)
    ax_f1.plot(thresholds, f1_mid)

    plt.figure(2)
    ax_prec_rec = plt.axes()
    ax_prec_rec.fill_between(rec_mid, prec_lower, prec_upper, alpha=0.6)
    ax_prec_rec.plot(rec_mid, prec_mid)
    # Should I use the lower and upper rec??
    # What's the significance of this statistically?
    # ax_prec_rec.fill_between(rec_lower, prec_lower, prec_mid, alpha=0.6)
    # ax_prec_rec.fill_between(rec_upper, prec_mid, prec_upper, alpha=0.6)
    # ax_prec_rec.plot(rec_lower, prec_lower)
    # ax_prec_rec.plot(rec_upper, prec_upper)

    plt.show()


# x = np.linspace(0, 30, 30)
# y = np.sin(x/6*np.pi)
# error = np.random.normal(0.1, 0.02, size=y.shape)
# y += np.random.normal(0, 0.1, size=y.shape)

# plt.plot(x, y, 'k-')
# plt.fill_between(x, y-error, y+error)
# plt.show()

# roc_curve


if __name__ == "__main__":
    # If you don't yet have the model predictions
    # Import model for testing
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = UNet(n_channels=3, n_classes=1, bilinear=False)
    # model = model.to(device)
    # model.load_state_dict(
    #     torch.load(
    #         "../stuff/Adam_e_3_reschedulat100toe4_epochs_200",
    #         map_location=torch.device("cpu"),
    #     )
    # )

    # _, validation_set, _ = load_data(
    #     dir_data="../data/",
    #     prop_noPV=0,
    #     min_rescale_images=0.6,
    #     batch_size=100,
    #     train_percentage=0.7,
    #     validation_percentage=0.15,
    # )

    # model.eval()
    # with torch.no_grad():
    #     images, labels = next(iter(validation_set))
    #     images = images.to(device, dtype=torch.float32)
    #     predictions = model(images).cpu().numpy()
    #     predictions = np.squeeze(predictions, axis=1)
    #     labels = labels.cpu().numpy()
    #     np.savez_compressed("../stuff/validation_set", predictions=predictions, labels=labels)

    arrays = np.load("../stuff/validation_set.npz")
    predictions = arrays["predictions"]
    labels = arrays["labels"]
    threshold_true = 0.5
    labels = np.where(labels > threshold_true, 1, 0)

    # true = np.array([0, 0, 1, 1])
    # p1 = np.round(np.array([0.1, 0.3, 0.7, 0.9]), decimals=1)
    # t1 = np.linspace(0, 1, 11)
    # plot_precision_recall(np.array([p1]), np.array([true]), 11)

    # print(np.cumsum(true)[build_idx(t1, p1)])
    plot_precision_recall(predictions, labels, 100)
    # prec, rec, thresh = precision_recall_curve(
    #     labels[0].flatten(), 1 / (1 + np.exp(predictions[0].flatten()))
    # )
    # print(prec)
    # print(rec)
    # print(thresh)
    # threshold_test(predictions, labels)
