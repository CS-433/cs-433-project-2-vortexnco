import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import random

from model.unet_model import UNet
from PIL import Image
from itertools import product
from helpers import *
from sklearn.metrics import precision_recall_fscore_support
# precision_score, recall_score, f1_score
from AvailableRooftopDataset import AvailableRooftopDataset
from torch.utils.data import DataLoader


dataFolder = "../data"
imageFolder = os.path.join(dataFolder, "PV")
labelFolder = os.path.join(dataFolder, "labels")
# test_filename = "DOP25_LV03_1301_11_2015_1_15_497500.0_119062.5.png"

map_rgb = {0: [0, 255, 0], 1: [0, 0, 0], 2: [255, 0, 0], 3: [255, 215, 0]}

# If color values are binary
COMPARE_MAP_01 = {
    (2, 0): 0,
    (0, 0): 1,
    (1, 1): 2,
    (1, -1): 3,
}

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
    for i, j in product(range(height), range(width)):
        comparison_rgb[i, j, :] = map_rgb[comparison[i, j]]

    plt.imshow(comparison_rgb)
    TP = mpatches.Patch(color="green", label="TP")
    TN = mpatches.Patch(color="black", label="TN")
    FP = mpatches.Patch(color="red", label="FP")
    FN = mpatches.Patch(color=[255 / 255, 215 / 255, 0], label="FN")
    plt.legend(handles=[TP, FP, TN, FN], bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


def eval_model(model, val_set):
    """Returns evaluation measures over a validation set."""
    precision = np.zeros(len(val_set))
    recall = np.zeros(len(val_set))
    f1 = np.zeros(len(val_set))
    support = np.zeros(len(val_set))
    for i, image, true_label in enumerate(val_set):
        precision[i], recall[i], f1[i], support[i] = precision_recall_fscore_support(
            true_label, model(image)
        )
    return f1, precision, recall, support
    # For different threshold probabilities
    # Predictions passed as probabilities
    # precision_recall_curve(y_true, probas_pred, *)
    # roc_curve


if __name__ == "__main__":
    label_files = random.sample(os.listdir(labelFolder), 2)
    image_files = [get_image_file(f) for f in label_files]
    label_files = [os.path.join(labelFolder, f) for f in label_files]
    image_files = [os.path.join(imageFolder, f) for f in image_files]
    labels = [
        np.array(Image.open(label_file).convert("L")) for label_file in label_files
    ]

    # Import model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model = model.to(device)
    model.load_state_dict(torch.load("../stuff/Adam_e_3_reschedulat100toe4_epochs_200",map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(roof_dataloader_train))
        images = images.to(device, dtype=torch.float32)
        predictions = model(images)

    # I'd like to do that on the full dataset
    image_numpy = images[i].cpu().numpy().transpose((1, 2, 0))
    # plt.imshow(image_numpy)
    # plt.show()

    label_numpy = labels[i].cpu().numpy()
    # plt.imshow(label_numpy)
    # plt.show()

    # Transforming output of model to probabilities
    predicted_numpy = np.squeeze(predictions.cpu().numpy()[i])
    predicted_numpy = 1 / (1 + np.exp(-predicted_numpy))
    # plt.imshow(predicted_numpy)
    # plt.show()

    # Thresholding prediction probabilities to make a decision
    threshold_prediction = 0.9
    pred = np.where(label_numpy > threshold_prediction, 1.0, 0.0)
    # Label needs to be thresholded because of transforms
    threshold_true_label = 0.5
    true = np.where(label_numpy > threshold_true_label, 1.0, 0.0)
    # plt.imshow(np.where(predicted_numpy>threshold_prediction, 1., 0.))
    # plt.show()

    show_label_comparison(true, pred)
    # plt.imshow(labels[0])
    # plt.show()
    # plt.imshow(labels[1])
    # plt.show()
    # show_label_comparison(labels[0], labels[1])
