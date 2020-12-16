import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from PIL import Image
from itertools import product


map_rgb = {0: [0, 255, 0], 1: [0, 0, 0], 2: [255, 0, 0], 3: [255, 215, 0]}
# If color values are binary
COMPARE_MAP_01 = {(2, 0): 0, (0, 0): 1, (1, 1): 2, (1, -1): 3}
# If color values are 3 bytes
COMPARE_MAP_uint8 = {(254, 0): 0, (0, 0): 1, (255, 255): 2, (255, 1): 3}


def compare_labels(true_label, predicted_label):
    """Outputs an array annotated as TP, FP, TN or FN as ints,
    according to the chosen COMPARE_MAP

    Inputs:
    ========
    true_label : ndarray
        True label for the image; only contains 1s and 0s.
    predicted_label : ndarray
        Prediction from the model for the image; only contains 1s and 0s.

    Returns:
    ========
    result : ndarray
        Annotations indicating whether each pixel in the predicted label is a
        true positive, true negative, false positive or false negative.
    """
    height, width = true_label.shape
    comp_array = np.array([predicted_label + true_label, predicted_label - true_label])
    result = np.empty((height, width), dtype=int)
    for i, j in product(range(height), range(height)):
        result[i, j] = COMPARE_MAP_uint8[tuple(comp_array[:, i, j])]
    return result


def show_label_comparison(true_label, predicted_label):
    """
    Plots an array annotated with TP, FP, TN and FN.

    Inputs:
    ========
    true_label : ndarray
        True label for the image; only contains 1s and 0s.
    predicted_label : ndarray
        Prediction from the model for the image; only contains 1s and 0s.

    Returns:
    ========
    None
    """
    comparison = compare_labels(true_label, predicted_label)
    height, width = comparison.shape
    # Convert to RGB
    comparison_rgb = np.empty((height, width, 3), dtype=int)
    for i, j in product(range(height), range(width)):
        comparison_rgb[i, j, :] = map_rgb[comparison[i, j]]

    # Plot the array
    plt.imshow(comparison_rgb)
    TP = mpatches.Patch(color="green", label="TP")
    TN = mpatches.Patch(color="black", label="TN")
    FP = mpatches.Patch(color="red", label="FP")
    FN = mpatches.Patch(color=[255 / 255, 215 / 255, 0], label="FN")
    plt.legend(handles=[TP, FP, TN, FN], bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


def plot_precision_recall_f1(
    thresholds,
    precision_summary,
    recall_summary,
    f1_summary,
    idx_best=None,
):
    """
    Plot precision-recall curve and F1 score against thresholds.
    Summary arrays should have the form:
        First row is lower bound (e.g. mean - std or first quartile)
        Second row is mid-point (e.g. mean or median)
        Third row is upper bound (e.g. mean + std or third quartile)

    Inputs:
    ========
    thresholds : ndarray
        Thresholds that give rise to precision, recall and F1 values.
    precision_summary : ndarray
        Summary of precision for each threshold.
    recall_summary : ndarray
        Summary of recall for each threshold.
    f1_summary : ndarray
        Summary of F1-score for each threshold.
    idx_best : int, optional
        Index of the best threshold.
        If None then no information is added to the plots.

    Returns:
    ========
    None
    """
    precision_lower, precision_mid, precision_upper = (row for row in precision_summary)
    recall_lower, recall_mid, recall_upper = (row for row in recall_summary)
    f1_lower, f1_mid, f1_upper = (row for row in f1_summary)

    plt.figure(1)
    ax_f1 = plt.axes()
    ax_f1.fill_between(thresholds, f1_lower, f1_upper, alpha=0.6)
    ax_f1.plot(thresholds, f1_mid)
    plt.grid(True)
    plt.xlabel("Thresholds")
    plt.ylabel(r"$F_1$-score")
    if idx_best:
        # Annotate with best estimator
        best_thresh = thresholds[idx_best]
        plt.text(best_thresh, f1_mid[idx_best], "{:.3f}".format(f1_mid[idx_best]))
        plt.text(best_thresh + 0.05, 0, "{:.3f}".format(best_thresh))
        plt.axvline(x=best_thresh, ymin=0, ymax=1, color="black", linestyle="--")
        # plt.axvline(x=best_thresh, ymin=0, ymax=f1_mid[idx_best], color="black", linestyle="--")

    plt.figure(2)
    ax_prec_rec = plt.axes()
    ax_prec_rec.fill_between(recall_mid, precision_lower, precision_upper, alpha=0.6)
    ax_prec_rec.plot(recall_mid, precision_mid)
    # Should I use the lower and upper rec??
    # What's the significance of this statistically?
    # ax_prec_rec.plot(recall_lower, precision_lower)
    # ax_prec_rec.plot(recall_upper, precision_upper)
    # ax_prec_rec.fill_between(recall_lower, precision_lower, precision_mid, alpha=0.6)
    # ax_prec_rec.fill_between(recall_upper, precision_mid, precision_upper, alpha=0.6)
    plt.grid(True)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if idx_best:
        # Annotate with best estimator
        recall_best = recall_mid[idx_best]
        precision_best = precision_mid[idx_best]
        ax_prec_rec.plot([recall_best], [precision_best], "ro")
        plt.text(
            recall_best + 0.02,
            precision_best + 0.02,
            "({:.3f}, {:.3f})".format(recall_best, precision_best),
        )

    plt.show()
