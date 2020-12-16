import torch
import numpy as np
import os
import sklearn.metrics

from model.unet_model import UNet
from helpers import summary_stats
from visualisation import plot_precision_recall_f1
from pipeline import load_data

METRICS = {
    "accuracy" : sklearn.metrics.accuracy_score,
    "balanced_accuracy" : sklearn.metrics.balanced_accuracy_score,
    "average_precision" : sklearn.metrics.average_precision_score,
    "neg_brier_score" : sklearn.metrics.brier_score_loss,
    "f1" : sklearn.metrics.f1_score,
    "f1_weighted" : lambda true, pred: sklearn.metrics.f1_score(true, pred, zero_division=1, average="weighted"),
    "neg_log_loss" : sklearn.metrics.log_loss,
    "precision" : sklearn.metrics.precision_score,
    "precision_weighted" : lambda true, pred: sklearn.metrics.precision_score(true, pred, zero_division=1, average="weighted"),
    "recall" : sklearn.metrics.recall_score,
    "recall_weighted" : lambda true, pred: sklearn.metrics.recall_score(true, pred, zero_division=1, average="weighted"),
    "jaccard" : sklearn.metrics.jaccard_score,
    "jaccard_weighted" : lambda true, pred: sklearn.metrics.jaccard_score(true, pred, average="weighted"),
    "roc_auc" : sklearn.metrics.roc_auc_score,
    # "roc_auc_ovr" : sklearn.metrics.roc_auc_score,
    # "roc_auc_ovo" : sklearn.metrics.roc_auc_score,
    # "roc_auc_ovr_weighted" : sklearn.metrics.roc_auc_score,
    # "roc_auc_ovo_weighted" : sklearn.metrics.roc_auc_score,
}

## VALIDATION
def find_best_threshold(predictions, labels, n_thresholds, plot=True):
    """Determine the best threshold given validation set and
    visualise results (precision-recall curve and F1-score against thresholds).
    This is based on the sklearn.metrics.precision_recall_curve function
    but adapred so that thresholds can be the same for each image (that way
    summary statistics can be computed).

    Inputs:
    ========
    predictions : ndarray
        model predictions on validation set
    labels : ndarray
        true labels corresponding to predictions on validation set
    n_thresholds : int
        number of thresholds to test for
    plot : bool, optional
        whether to plot results or not

    Returns:
    ========
    precision, recall, f1_scores : ndarray
        Evaluation measures for each threshold, for each image
    best_thresh : float
        The threshold which maximizes some value
    """
    # Don't keep 0 or 1 because they are uninteresting
    # and they cause issues down the line
    thresholds = np.linspace(0, 1, n_thresholds)[1:-1]
    n_thresholds = len(thresholds)
    pred_probas = 1 / (1 + np.exp(-predictions))

    precision = np.zeros((len(predictions), n_thresholds))
    recall = np.zeros((len(predictions), n_thresholds))
    f1_scores = np.zeros((len(predictions), n_thresholds))
    n = len(pred_probas[0].flatten())

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
        predicted_1 = n - thresholds_idx
        predicted_0 = n - predicted_1
        actually_1 = tps[0]
        actually_0 = n - actually_1
        # Positives - True positives
        fns = actually_1 - tps
        # Negatives - False negatives
        tns = actually_0 - fns

        with np.errstate(divide="ignore", invalid="ignore"):
            prec_1 = tps / predicted_1
        #     prec_0 = tns / predicted_0
        # prec_1 = np.nan_to_num(prec_1, 0)
        # prec_0 = np.nan_to_num(prec_0, 0)
        # prec = (actually_1 * prec_1 + actually_0 * prec_0) / n
        prec = np.nan_to_num(prec_1, 1)
        # If you never predict 1 your precision is bad
        # But I need the precision-recall curve to make sense
        # (i.e. that precision = 1 when recall = 0)
        # and the F1-score to be defined
        # (i.e. that precision and recall aren't both 0)
        prec[limit_idx] = 1
        precision[i] = prec
        with np.errstate(divide="ignore", invalid="ignore"):
            rec_1 = tps / actually_1
        #     rec_0 = tns / actually_0
        # rec_1 = np.nan_to_num(rec_1, 0)
        # rec_0 = np.nan_to_num(rec_0, 0)
        # # The recall weighted by support seems to be the same as the accuracy
        # rec = (actually_1 * rec_1 + actually_0 * rec_0) / n
        rec = np.nan_to_num(rec_1, 1)
        recall[i] = rec
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = 2 * (prec * rec) / (prec + rec)
        f1_scores[i] = np.nan_to_num(f1, 0)

    summary_type = "median"
    lower_bound = 25
    prec_summary = summary_stats(precision, type=summary_type, lower_bound=lower_bound)
    rec_summary = summary_stats(recall, type=summary_type, lower_bound=lower_bound)
    f1_summary = summary_stats(f1_scores, type=summary_type, lower_bound=lower_bound)

    # Estimating what the threshold should be set to
    f1_lower, f1_mid, f1_upper = (row for row in f1_summary)
    # This measure penalises uncertain choices (maximize (mean - spread))
    idx_best = np.argmax(f1_mid - (f1_upper - f1_lower))
    # idx_best = np.argmax(f1_mid)
    if plot:
        plot_precision_recall_f1(
            thresholds, prec_summary, rec_summary, f1_summary, idx_best=idx_best
        )

    return precision, recall, f1_scores, thresholds[idx_best]


## TESTING (once threshold is chosen)
def test_model(test_predictions, test_labels, threshold, concat, *args):
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
    concat : bool
        Whether to compute each metric once, on the concatenation of the whole test set.
    args : strings
        Metrics to use for testing: the list of accepted strings can be found at
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter.


    Returns:
    ========
    results : ndarray
        Test results; rows correspond to datapoints
        and columns correspond to metrics (in the order they were passed)
    """
    scorings = [METRICS[s] for s in args]
    pred_probas = 1 / (1 + np.exp(-test_predictions))
    test_predictions = np.where(pred_probas > threshold, 1, 0)


    if concat:
        results = np.zeros((1, len(args)))
        true = test_labels.flatten()
        pred = test_predictions.flatten()
        for i, scoring in enumerate(scorings):
            results[0, i] = scoring(true, pred)

    else:
        results = np.zeros((len(test_predictions), len(args)))
        for i, (true, pred) in enumerate(zip(test_labels, test_predictions)):
            # Could probably flatten the two at the beginning, like I do in concat
            true = true.flatten()
            pred = pred.flatten()
            for j, scoring in enumerate(scorings):
                results[i, j] = scoring(true, pred)

    return results


def main(
    model_name,
    from_file=True,
    to_file=False,
    validation=True,
    test=["precision", "recall", "f1", "accuracy", "jaccard"],
    concat=False,
    plot=True,
):
    """
    Inputs:
    ========
    model_name : str
        Which model to do things with. This is assumed to be both the name of the directory in which parameters are stored, and the name of the parameters file.
    from_file : bool
        Whether data should be loaded from a file; otherwise it will be generated.
        The file should:
            - be in the same directory as the model parameters
            - be called "data.npz"
            - contain 4 arrays: "val_predictions", "val_labels", "test_predictions" and "test_labels".
        Labels are expected to be floats and will be thresholded.
        Predictions are expected to be raw (not probabilities).
    to_file : bool
        If data is generated (not loaded from a file), whether to save to a file in the model directory, according to the form described in from_file.
        Irrelevant when from_file is set to True.
    validation : bool
        Whether to go through validation steps (to find the best threshold).
    test : list of str
        Which metrics to use for testing (evaluate the model with a given threshold).
        The results are stored in a txt file called "test_results.txt" in the model directory.
        If empty then testing is skipped.
    concat : bool
        During testing, whether to compute each metric once, on the concatenation of the whole test set.
    plot : bool
        Whether to show plots during run.
    """
    model_dir = os.path.join(dir_models, model_name)
    params_file = os.path.join(model_dir, model_name)
    new_section = "=" * 50

    print("Importing model parameters from {}".format(params_file))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model = model.to(device)
    model.load_state_dict(torch.load(params_file, map_location=torch.device("cpu")))

    # File where data are stored, or will be if they aren't already
    data_file = os.path.join(model_dir, "data.npz")
    print(new_section)
    if from_file:
        print("Loading data")
        arrays = np.load(data_file)
        val_predictions, val_labels, test_predictions, test_labels = arrays.values()
    else:
        print("Generating data")
        dir_data_validation = os.path.join(dir_data, "validation")
        dir_data_test = os.path.join(dir_data, "test")

        _, validation_dl, test_dl = load_data(
            dir_data_validation=dir_data_validation,
            dir_data_test=dir_data_test,
            prop_noPV_training=0,  # Has no impact
            min_rescale_images=0,  # Has no impact
            batch_size=100,  # All of them
        )

        model.eval()
        with torch.no_grad():
            # Get images and labels from both DataLoaders
            val_images, val_labels = next(iter(validation_dl))
            test_images, test_labels = next(iter(test_dl))
            val_images = val_images.to(device, dtype=torch.float32)
            test_images = test_images.to(device, dtype=torch.float32)
            # Make predictions (predictions are not probabilities at this stage)
            print("Running model on data")
            val_predictions = model(val_images)
            test_predictions = model(test_images)
            # Convert to numpy arrays for computing
            val_predictions = np.squeeze(val_predictions.cpu().numpy())
            val_labels = np.squeeze(val_labels.cpu().numpy())
            test_predictions = np.squeeze(test_predictions.cpu().numpy())
            test_labels = np.squeeze(test_labels.cpu().numpy())
            # Save to file as numpy arrays
            if to_file:
                print("Saving results to file")
                np.savez_compressed(
                    data_file,
                    val_predictions=val_predictions,
                    val_labels=val_labels,
                    test_predictions=test_predictions,
                    test_labels=test_labels,
                )

    threshold_true = 0.5
    val_labels = np.where(val_labels > threshold_true, 1, 0)
    test_labels = np.where(test_labels > threshold_true, 1, 0)

    if validation:
        print(new_section)
        print("Validation starting")
        _, _, _, best_threshold = find_best_threshold(
            val_predictions, val_labels, n_thresholds=101, plot=plot
        )
        print(f"Found best threshold to be {best_threshold:.4f}")

    if test:
        print(new_section)
        print("Testing starting with metrics:")
        print(", ".join(test))
        results = test_model(test_predictions, test_labels, best_threshold, concat, *test)
        print(results)
        summary_type = "median"
        results_file = os.path.join(model_dir, "test_{}results.txt".format('concat_' if concat else ''))
        # results = np.column_stack([summary_stats(x, type=summary_type) for x in [f1, precision, recall]])
        if concat:
            results_summary = np.transpose(results)
            print("Results:")
        else:
            results_summary = np.transpose(summary_stats(results, type=summary_type))
            print(f"Summary statistics are based on the {summary_type}")
            print("Results (lower, mid-point, upper):")
        print(f"\tBest threshold = {best_threshold:.4f}")
        for i, measure in enumerate(test):
            print("\t{}: {}".format(measure, results_summary[i, :]))
        print(new_section)
        print("Saving results to {}".format(results_file))
        np.savetxt(
            results_file,
            results_summary,
            fmt="%.4f",
            delimiter=" ",
            header=f"Threshold: {best_threshold:.3f}\n{'  '.join(test)}",
        )

    print("\n")


if __name__ == "__main__":
    # dir_models = "/home/auguste/allFinalModels/"
    # dir_data = "/raid/machinelearning_course/data"
    dir_models = "../stuff/models_data/"
    dir_data = "../data/data/"

    test = ["precision", "recall", "f1", "accuracy", "jaccard"]

    # for model in os.listdir(dir_models):
    model = "Adam_e_4_withoutnoPV_BCEwithweights_epochs_100_noscheduler"
    main(
        model_name=model,
        from_file=True,  # Should probably be put to a filename
        to_file=True,
        validation=True,
        test=test,
        concat=True,
        plot=False,
    )
