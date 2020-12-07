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
true_positive = [0, 255, 0]
true_negative = [0, 0, 0]
false_positive = [255, 0, 0]
false_negative = [255, 215, 0]

# If color values are binary
COMPARE_MAP_01 = {
    (2,0)  : true_positive,
    (0,0)  : true_negative,
    (1,1)  : false_positive,
    (1,-1) : false_negative
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

    result = np.empty((3, height, width), dtype=int)
    for i, j in product(range(height), range(height)):
        result[:, i, j] = f(i, j)

    return result.transpose((1,2,0))

def show_label_comparison(true_label, predicted_label):
    comparison = compare_labels(true_label, predicted_label)
    plt.imshow(comparison)
    TP = mpatches.Patch(color="green", label="TP")
    TN = mpatches.Patch(color="black", label="TN")
    FP = mpatches.Patch(color="red", label="FP")
    FN = mpatches.Patch(color=[255 / 255, 215 / 255, 0], label="FN")
    plt.legend(handles=[TP, FP, TN, FN], bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


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
