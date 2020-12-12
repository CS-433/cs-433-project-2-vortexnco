import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import random
import torch

from model.unet_model import UNet
from pipeline import load_data
from PIL import Image
from itertools import product
from helpers import *
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    f1_score,
)
from scipy import interpolate

# precision_score, recall_score, f1_score
from AvailableRooftopDataset import AvailableRooftopDataset
from torch.utils.data import DataLoader


dataFolder = "../data"
imageFolder = os.path.join(dataFolder, "PV")
labelFolder = os.path.join(dataFolder, "labels")
# test_filename = "DOP25_LV03_1301_11_2015_1_15_497500.0_119062.5.png"

map_rgb = {0: [0, 255, 0], 1: [0, 0, 0], 2: [255, 0, 0], 3: [255, 215, 0]}


# If color values are binary
COMPARE_MAP_01 = {(2, 0): 0, (0, 0): 1, (1, 1): 2, (1, -1): 3}

# If color values are 3 bytes
COMPARE_MAP_uint8 = {(254, 0): 0, (0, 0): 1, (255, 255): 2, (255, 1): 3}


def compare_labels(true_label, predicted_label):
    """Outputs an array annotated as TP, FP, TN or FN as ints"""
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
    None.
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


def eval_model(model, test_set):
    """Returns evaluation measures over a test set."""
    precision = np.zeros(len(val_set))
    recall = np.zeros(len(val_set))
    f1 = np.zeros(len(val_set))
    support = np.zeros(len(val_set))
    for i, image, true_label in enumerate(test_set):
        precision[i], recall[i], f1[i], support[i] = precision_recall_fscore_support(
            true_label, model(image)
        )
    return f1, precision, recall, support


# def build_idx(thresholds, pred_proba):
#     """Build the index array that will be used to
#     compute the precision and recall for different
#     thresholds.

#     Returns:
#     ========
#     thresholds_idx : ndarray
#         Array the same shape as thresholds that holds
#         the first index of pred_probas after which
#         pred_probas is larger than a given threshold
#     """
#     thresholds_idx = np.searchsorted(pred_proba, thresholds)
    # idx = 0
    # thresh = thresholds[0]
    # # print(pred_proba)
    # for i, proba in enumerate(pred_proba):
    # #     print("I'm at index {} in pred_proba".format(i))
    #     while proba >= thresh:
    # #         print("idx is {}; the proba is bigger than the threshold.".format(idx))
    #         thresholds_idx[idx] = i
    #         idx += 1
    #         thresh = thresholds[idx]
    # #     print("idx is {}; the proba is smaller than the threshold so I'm advancing.".format(idx))
    # # print(thresholds_idx)
    # return np.r_[thresholds_idx[:-1], len(pred_proba) - 1]
def summary_stats(array, axis = 0, type = "median"):
    """Summary statistics of given type"""
    if type in ["mean", "average", "avg"]:
        mid = np.mean(array, axis=axis)
        std = np.std(array, axis=axis)
        lower = avg - std
        upper = avg + std
    elif type in ["median", "order", "quantiles"]:
        mid = np.median(array, axis=axis)
        lower = np.percentile(array, 25, axis=axis)
        upper = np.percentile(array, 75, axis=axis)
    return mid, lower, upper



def plot_precision_recall(predictions, labels, n_thresholds):
    """Plot precision-recall curves over a validation set
    to determine the best threshold.

    Inputs:
    ========
    predictions : array
      model predictions
    labels : array
      true labels corresponding to predictions
    n_thresholds : int
      number of thresholds to test for
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    pred_probas = 1 / (1 + np.exp(-predictions))

    precision = np.zeros((len(predictions), n_thresholds))
    recall = np.zeros((len(predictions), n_thresholds))
    f1_scores = np.zeros((len(predictions), n_thresholds))
    # fig, (ax_prec_rec, ax_f1) = plt.subplots(nrows=2)

    for i, (true, pred) in enumerate(zip(labels, pred_probas)):
        pred = pred.flatten()

        # Sort increasingly
        sort = np.argsort(pred)
        true = true.flatten()[sort]
        # print("True: {}".format(true))
        pred = pred[sort]
        # print("Predicted: {}".format(pred))

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
