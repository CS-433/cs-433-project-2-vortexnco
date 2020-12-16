import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from itertools import product
from load_data import load_data

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
        result[i, j] = COMPARE_MAP_01[tuple(comp_array[:, i, j])]
    return result

def show_full_comparisonTestGenerator(model, threshold_prediction = 0.9,
                        dir_data_training = "../data/train",
                        dir_data_validation = "../data/validation",
                        dir_data_test= "../data/test"):
    """
    Creates a generator for plots vizualizing the results of the model.

    Parameters
    ----------
    model : TYPE
        Model to use.
    threshold_prediction : float, optional
        Threshold to use after the model predicts probabilities. The default is 0.9.
    dir_data_training : TYPE, optional
        Directory of Train data. The default is "../data/train".
    dir_data_validation : TYPE, optional
        Directory of Validation data. The default is "../data/validation".
    dir_data_test : TYPE, optional
        Directory of Test data. The default is "../data/test".

    Returns
    -------
    None.

    """
    _, _, test_dl =  load_data(
        dir_data_training,
        dir_data_validation,
        dir_data_test,
        prop_noPV_training = 0.0, #dummy value since only used in Train
        min_rescale_images = 0.6, #dummy value since only used in Train
        batch_size = 1
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for images, labels in test_dl:
        model.eval()
        with torch.no_grad():
            images = images.to(device, dtype=torch.float32)
            predictions = model(images)
        
        #Plotting first image of batch
        i=0
        fig, axs = plt.subplots(2,2, figsize = (8,8))

        #Showing Aerial Image
        image_numpy = images[i].cpu().numpy().transpose((1,2,0))        
        axs[0,0].imshow(image_numpy)
        axs[0,0].set_title("Image")
        
        #Sgowing True label
        label_numpy = labels[i].cpu().numpy()
        axs[0,1].imshow(label_numpy)
        axs[0,1].set_title("True label")
        
        #transforming output of model to probabilities
        predicted_numpy = np.squeeze(predictions.cpu().numpy()[i])
        predicted_numpy = 1/(1 + np.exp(-predicted_numpy)) 
        
        
        #Thresholding prediction probabilities to make a decision
        axs[1,0].imshow(np.where(predicted_numpy>threshold_prediction, 1., 0.))
        axs[1,0].set_title("Prediction")
        

        #Comparing label to decision 
        show_label_comparison(label_numpy, np.where(predicted_numpy>threshold_prediction, 1, 0), axs[1,1])
        fig.tight_layout()
        fig.show()
        yield

def show_label_comparison(true_label, predicted_label, ax):
    """
    Plots an array annotated with TP, FP, TN and FN.

    Inputs:
    ========
    true_label : ndarray
        True label for the image; only contains 1s and 0s.
    predicted_label : ndarray
        Prediction from the model for the image; only contains 1s and 0s.
    ax :
        Axe object on which to plot the comparison

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
    ax.legend(handles=[TP, FP, TN, FN], bbox_to_anchor=(1.05, 1), loc="upper left")


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
