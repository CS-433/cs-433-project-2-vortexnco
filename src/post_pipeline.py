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
        # True positives for each threshold
        tps = np.cumsum(true[::-1])[::-1][thresholds_idx]
        # If you never predict 1 you have no true positives
        tps[limit_idx] = 0
        predicted_true = len(true) - thresholds_idx
        actually_true = tps[0]

        prec = tps / predicted_true
        # If you never predict 1 your precision is bad
        # But I need the precision-recall curve to make sense
        # (i.e. that precision = 1 when recall = 0)
        # and the F1-score to be defined
        # (i.e. that precision and recall aren't both 0)
        prec[limit_idx] = 1
        precision[i] = prec
        with np.errstate(divide="ignore", invalid="ignore"):
            rec = tps / actually_true
        rec = np.nan_to_num(rec, 0)
        recall[i] = rec
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = 2 * (prec * rec) / (prec + rec)
        f1_scores[i] = np.nan_to_num(f1, 0)

    prec_summary = summary_stats(precision, type="median")
    rec_summary = summary_stats(recall, type="median")
    f1_summary = summary_stats(f1_scores, type="median")

    # Estimating what the threshold should be set to
    f1_mid, f1_lower, f1_upper = (row for row in f1_summary)
    # This measure penalises uncertain choices (maximize (mean - spread))
    idx_best = np.argmax(f1_mid - (f1_upper - f1_lower))
    if plot:
        plot_precision_recall_f1(
            thresholds, precision_summary, recall_summary, f1_summary, idx_best=idx_best
        )

    return precision, recall, f1_scores, thresholds[idx_best]


def plot_precision_recall_f1(
    thresholds,
    precision_summary,
    recall_summary,
    f1_summary,
    idx_best=None,
    to_file=False,
):
    """Plot precision-recall curve and F1 score against thresholds.

    Inputs:
    ========
    thresholds : ndarray
        Thresholds that give rise to precision, recall and F1 values.
    precision_summary : ndarray
        Contains summary statistics of precision for each threshold:
        First row is mid-point (e.g. mean or median)
        Second row is lower bound (e.g. mean - std or first quartile)
        Third row is upper bound (e.g. mean + std or third quartile)
    recall_summary : ndarray
        Contains summary statistics of recall for each threshold:
        First row is mid-point (e.g. mean or median)
        Second row is lower bound (e.g. mean - std or first quartile)
        Third row is upper bound (e.g. mean + std or third quartile)
    f1_summary : ndarray
        Contains summary statistics of F1-score for each threshold:
        First row is mid-point (e.g. mean or median)
        Second row is lower bound (e.g. mean - std or first quartile)
        Third row is upper bound (e.g. mean + std or third quartile)
    idx_best : int, optional
        Index of the best threshold.
        If None then no information is added to the plots.

    Returns:
    ========
    None
    """
    precision_mid, precision_lower, precision_upper = (row for row in precision_summary)
    recall_mid, recall_lower, recall_upper = (row for row in recall_summary)
    f1_mid, f1_lower, f1_upper = (row for row in f1_summary)

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
    ax_prec_rec.fill_between(rec_mid, prec_lower, prec_upper, alpha=0.6)
    ax_prec_rec.plot(rec_mid, prec_mid)
    # Should I use the lower and upper rec??
    # What's the significance of this statistically?
    # ax_prec_rec.plot(rec_lower, prec_lower)
    # ax_prec_rec.plot(rec_upper, prec_upper)
    # ax_prec_rec.fill_between(rec_lower, prec_lower, prec_mid, alpha=0.6)
    # ax_prec_rec.fill_between(rec_upper, prec_mid, prec_upper, alpha=0.6)
    plt.grid(True)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if idx_best:
        # Annotate with best estimator
        rec_best = rec_mid[idx_best]
        prec_best = prec_mid[idx_best]
        ax_prec_rec.plot([rec_best], [prec_best], "ro")
        plt.text(
            rec_best + 0.02,
            prec_best + 0.02,
            "({:.3f}, {:.3f})".format(rec_best, prec_best),
        )

    plt.show()


def main(
    model_name, prop_noPV, load_data=True, validation=True, test=True, verbose=False
):
    """
    Inputs:
    ========
    model_name : str
        Which model to do things with. This is assumed to be both the name of the directory in which parameters are stored, and the name of the parameters file.
    prop_noPV : float
        If generating data, what proportion of the noPV data should be included. This is intended to be the same proportion that the model was trained with.
        Should be between 0 and 1.
    load_data : bool
        Whether data should be loaded from a file; otherwise it will be generated.
        The file should:
            - be in the same directory as the model parameters
            - be called "data.npz"
            - contain 4 arrays: "val_predictions", "val_labels", "test_predictions" and "test_labels".
        If it's generated it'll be saved in a file in the model directory, according to the form given previously.
        Labels are expected to be floats and will be thresholded.
        Predictions are expected to be raw (not probabilities).
    validation : bool
        Whether to go through validation steps (to find the best threshold).
    test : bool
        Whether to go through testing steps (evaluate the model with a given threshold).
        The results are stored in a txt file called "test_results.txt" in the model directory.
    verbose : bool
        Whether to give information during run.
    """
    model_dir = os.path.join(dir_models, model_name)
    params_file = os.path.join(model_dir, model_name)
    data_file = os.path.join(model_dir, "data.npz")
    results_file = os.path.join(model_dir, "test_results.txt")

    # Import model parameters
    print("Importing model parameters from {}".format(params_file))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model = model.to(device)
    model.load_state_dict(torch.load(params_file, map_location=torch.device("cpu")))

    if load_data:
        print("=" * 50)
        print("Loading data")
        # arrays = np.load("../stuff/validation_set.npz")
        arrays = np.load(data_file)
        # val_predictions = arrays["val_predictions"]
        # val_labels = arrays["val_labels"]
        # test_predictions = arrays["test_predictions"]
        # test_labels = arrays["test_labels"]
        val_predictions, val_labels, test_predictions, test_labels = arrays.values()
    else:
        # Generate validation and test sets
        print("=" * 50)
        print("Generating data")
        _, validation_set, test_set = load_data(
            dir_data="../data/",
            prop_noPV=prop_noPV,
            min_rescale_images=0.6,
            batch_size=100,
            train_percentage=0.7,
            validation_percentage=0.15,
        )

        model.eval()
        with torch.no_grad():
            # Get images and labels from both sets
            val_images, val_labels = next(iter(validation_set))
            test_images, test_labels = next(iter(test_set))
            val_images = val_images.to(device, dtype=torch.float32)
            test_images = test_images.to(device, dtype=torch.float32)
            # Make predictions (predictions are not probabilities at this stage)
            print("Running model on data")
            val_predictions = model(val_images)
            test_predictions = model(test_images)
            # Save to file as numpy arrays
            print("Saving results to file")
            np.savez_compressed(
                data_file,
                val_predictions=np.squeeze(val_predictions.cpu().numpy(), axis=1),
                val_labels=val_labels.cpu().numpy(),
                test_predictions=np.squeeze(test_predictions.cpu().numpy(), axis=1),
                test_labels=test_labels.cpu().numpy(),
            )

    threshold_true = 0.5
    val_labels = np.where(val_labels > threshold_true, 1, 0)
    test_labels = np.where(test_labels > threshold_true, 1, 0)

    if validation:
        print("=" * 50)
        print("Validation starting")
        _, _, _, best_threshold = validation(
            val_predictions, val_labels, n_thresholds=100
        )
        print("Found best threshold to be {}".format(best_threshold))

    if test:
        print("=" * 50)
        print("Testing starting")
        f1, precision, recall, support = test(
            test_predictions, test_labels, best_threshold
        )
        summary_type = "mean"
        results = np.column_stack(
            (
                summary_stats(x, type=summary_type)
                for x in [f1, precision, recall, support]
            )
        )
        print("Summary statistics are based on the {}.".format(summary_type))
        print(
            "Results (mid-point, lower, upper) for threshold = {}:".format(
                best_threshold
            )
        )
        for i, measure in enumerate(["F1-score", "Precision", "Recall", "Support"]):
            print("{}: {}".format(measure, results[:, i]))
        print("Saving results to {}".format(results_file))
        np.savetxt(
            results_file,
            results,
            delimiter=" ",
            header="Threshold: {}\nf1_score  precision  recall  support".format(
                best_threshold
            ),
        )


if __name__ == "__main__":
    dir_models = "../stuff/FinalModels/"

    noPV_percentages = dict(zip(os.listdir(dir_models), [0 for _ in range(1000)]))
    noPV_percentages[
        "Adam_e_3_20percentnoPV_BCEwithweights_epochs_80_rescheduledat50_toe4"
    ] = 0.2
    noPV_percentages[
        "Adam_e_3_100percentnoPV_BCEwithweights_epochs_80_rescheduledat50_toe4"
    ] = 1
    print(noPV_percentages)

    # for model in noPV_percentages.keys():
    model = "Adam_e_4_withoutnoPV_BCEwithweights_epochs_100_noscheduler"
    main(
        model_name=model,
        prop_noPV=noPV_percentages[model],
        load_data=True,
        validation=True,
        test=True,
        verbose=False,
    )
