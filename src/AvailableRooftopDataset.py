import os
import random

from torch.utils.data import Dataset
from skimage import io, transform
import torch
import numpy as np


class AvailableRooftopDataset(Dataset):
    """Available Rooftop Dataset."""

    def __init__(self, dir_images, dir_labels, transform=None):
        """
        Args:
            dir_images (string): Directory with all the images.
            dir_labels (string): Directory with all the labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir_images = dir_images
        self.dir_labels = dir_labels
        self.transform = transform

        # Get the list of image/label name from images/labels directory, except dotfile
        self.images_name = [image_name for image_name in os.listdir(dir_images) if image_name[0] != '.']
        self.labels_name = [label_name for label_name in os.listdir(dir_labels) if label_name[0] != '.']

        # Create an image -> label dict
        self.image_label_dict = {}

        # Iterate through all the images' name to add them in the dict
        for image_full_name in self.images_name:
            image_name, image_extension = os.path.splitext(image_full_name)

            # Find the label of the image, if there is one
            label_name_associated = None
            for label_full_name in self.labels_name:
                if image_name in label_full_name:
                    label_name_associated = label_full_name

            # If no label associated, then it should be a black label
            if (not label_name_associated):
                label_name_associated = 'DEFAULT'

            self.image_label_dict[image_full_name] = label_name_associated

        # Shuffle the images' name to avoid having an order when retrieving the images in __getitem__
        random.shuffle(self.images_name)


    def __len__(self):
        return len(self.image_label_dict)


    def __getitem__(self, idx):
        # In case we use random_split
        if torch.is_tensor(idx):
             idx = idx.tolist()

        image_name = self.images_name[idx]
        label_name = self.image_label_dict[image_name]
        
        image_path = os.path.join(self.dir_images, image_name)
        if label_name != 'DEFAULT':
            label_path = os.path.join(self.dir_labels, label_name)
        
        # Retrieve the image and transpose from (HxWxC) -> (CxHxW)
        image = io.imread(image_path)
        image = image.transpose(2, 0, 1)

        label = np.zeros((1, 250, 250))
        if label_name != 'DEFAULT':
            # Retrieve the label and transpose from (HxWxC) -> (CxHxW)
            label = io.imread(label_path, as_gray=True)
            label = label[np.newaxis,:]

        sample = {'image': image, 'label': label}
        
        # Apply the transforms if any
        if self.transform:
            sample = self.transform(sample)

        return sample
