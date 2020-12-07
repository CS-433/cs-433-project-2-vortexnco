import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import random

from PIL import Image
from itertools import product
from helpers import *


dataFolder = "../data"
imageFolder = os.path.join(dataFolder, "images")
labelFolder = os.path.join(dataFolder, "labels")
test_filename = "DOP25_LV03_1301_11_2015_1_15_497500.0_119062.5.png"

# POS = 255
# NEG = 0
# true_value = 255
map_rgb = {
    "tp" : [0, 255, 0]
    "tn" : [0, 0, 0]
    "fp" : [255, 0, 0]
    "fn" : [255, 215, 0]
}

# If color values are binary
COMPARE_MAP_01 = {
    (2,0)  : "tp",
    (0,0)  : "tn",
    (1,1)  : "fp",
    (1,-1) : "fn"
}

# If color values are 3 bytes
# COMPARE_MAP_uint8 = {
#     (254, 0): true_positive,
#     (0, 0): true_negative,
#     (255, 255): false_positive,
#     (255, 1): false_negative,
# }


def compare_labels(true_label, predicted_label):
    """Outputs an array annotated as TP, FP, TN or FN"""
    height, width = true_label.shape
    comp_array = np.array([predicted_label + true_label, predicted_label - true_label])
    f = lambda i, j: COMPARE_MAP_01[tuple(comp_array[:, i, j])]

    result = np.fromfunction(f, (height, width), dtype=str)
    return result
    #result = np.empty((3, height, width), dtype=int)
    #for i, j in product(range(height), range(height)):
        #result[:, i, j] = f(i, j)

    #return result.transpose((1,2,0))


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
    # Recover comparison array and convert it to RGB
    comparison_rgb = np.empty((height, width, 3), dtype=int)
    for i, j in product(range(height), range(width)):
        comparison_rgb[i, j, :] = f(i, j)

    plt.imshow(comparison_rgb)
    TP = mpatches.Patch(color="green", label="TP")
    TN = mpatches.Patch(color="black", label="TN")
    FP = mpatches.Patch(color="red", label="FP")
    FN = mpatches.Patch(color=[255 / 255, 215 / 255, 0], label="FN")
    plt.legend(handles=[TP, FP, TN, FN], bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


def eval_model(model, val_set):


if __name__ == "__main__":
    label_files = random.sample(os.listdir(labelFolder), 2)
    image_files = [get_image_file(f) for f in label_files]
    label_files = [os.path.join(labelFolder, f) for f in label_files]
    image_files = [os.path.join(imageFolder, f) for f in image_files]
    labels = [
        np.array(Image.open(label_file).convert("L")) for label_file in label_files
    ]

    plt.imshow(labels[0])
    plt.show()
    plt.imshow(labels[1])
    plt.show()
    show_label_comparison(labels[0], labels[1])
