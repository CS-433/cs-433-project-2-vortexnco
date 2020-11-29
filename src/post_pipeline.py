import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os 
import random

from PIL import Image
from itertools import product
from helpers import *

dataFolder = "../data"
test_filename = "DOP25_LV03_1301_11_2015_1_15_497500.0_119062.5.png"

def compare_labels(true_label, predicted_label):
    (height, width) = true_label.shape
    #array with annotated arreas for TP, FP, TN, FN
    array = np.zeros((height, width, 3), dtype=int)  
    
    #true_value = np.array([255,255,255]) #TODO change maybe in 1 (RGB -> B&W)
    true_value = 255
    true_positive = [0,255,0]
    false_positive = [255,0,0]
    false_negative = [255,215,0]
    
    #TODO might need to remove : if numpy array are just HxW and not HxWx3
    for h in range(height):
        for w in range(width):
            if true_label[h,w] == predicted_label[h,w]:
                if true_label[h,w] == true_value:
                    array[h,w] = true_positive
            else:
                if true_label[h,w] == true_value:
                    array[h,w] = false_negative
                else:
                    array[h,w] = false_positive
    return array

if __name__ =="__main__":
    filenames = random.sample(os.listdir(os.path.join(dataFolder, "PV")), 2)
    label_files = [os.path.join(dataFolder, "PV", "labels", get_label_file(filename)) for filename in filenames]
    labels = [np.array(Image.open(label_file).convert('L')) for label_file in label_files]
    
    plt.imshow(labels[0])
    plt.show()
    plt.imshow(labels[1])
    plt.show()
    true_value = np.array([1,1,1], dtype=np.uint8)
    array = compare_labels(labels[0], labels[1])
    plt.imshow(array)
    TP = mpatches.Patch(color='green', label='TP')
    TN = mpatches.Patch(color='black', label='TN')
    FP = mpatches.Patch(color='red', label='FP')
    FN = mpatches.Patch(color=[255/255,215/255,0], label='FN')
    plt.legend(handles=[TP, FP, TN, FN], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
