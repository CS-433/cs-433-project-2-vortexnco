# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:57:29 2020

@author: Alexander
"""

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import Image
import random

dataFolder = "../data"
test_filename = "DOP25_LV03_1301_11_2015_1_15_497500.0_119062.5.png"



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
"""
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(make_grid(images[:4, : ,:, :]))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
"""

def get_groudtruth_file(filename_image):
    filename, file_extension = os.path.splitext(filename_image)
    filename_label = filename + "_label"
    return filename_label+file_extension


def compare_labels(true_label, predicted_label):
    array = np.zeros_like(true_label)
    (height, width, channels) = array.shape
    
    true_value = np.array([255,255,255])
    
    true_positive = [0,255,0]
    false_positive = [255,0,0]
    false_negative = [255,215,0]
    
    for h in range(height):
        for w in range(width):
            if np.equal(true_label[h,w,:], predicted_label[h,w,:]).all():
                if np.equal(true_label[h,w,:], true_value).all():
                    array[h,w,:] = true_positive
            else:
                if np.equal(true_label[h,w,:], true_value).all():
                    array[h,w,:] = false_negative
                else:
                    array[h,w,:] = false_positive
    return array

    
def aa(filename):
    img_file = os.path.join(dataFolder, "PV", filename)
    label_filename = get_groudtruth_file(filename)
    label_file = os.path.join(dataFolder, "PV", "labels", label_filename)
    
    img = Image.open(img_file)
    label = Image.open(label_file)
    
    arr = np.array(img)
    plt.imshow(arr)
    plt.show()
    arr_label = np.array(label)
    print(arr_label.shape)
    plt.imshow(arr_label)
    plt.show()
    # plt.imshow(img)

if __name__ =="__main__":
    filenames = random.sample(os.listdir(os.path.join(dataFolder, "PV")), 2)
    label_files = [os.path.join(dataFolder, "PV", "labels", get_groudtruth_file(filename)) for filename in filenames]
    labels = [np.array(Image.open(label_file)) for label_file in label_files]
    plt.imshow(labels[0])
    plt.show()
    plt.imshow(labels[1])
    plt.show()
    true_value = np.array([1,1,1]).astype(np.uint8)
    array = compare_labels(labels[0], labels[1])
    plt.imshow(array)
    plt.show()
